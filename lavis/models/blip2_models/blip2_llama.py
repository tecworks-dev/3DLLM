'''
Author: Diantao Tu
Date: 2023-04-16 13:46:07
'''
"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn

from lavis.common.registry import registry
from lavis.models.blip2_models.blip2 import Blip2Base, disabled_train
from transformers import AutoTokenizer, AutoModelForCausalLM
# import lavis.models.blip2_models.llama as llama
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
import time

@registry.register_model("blip2_llama")
class Blip2Llama(Blip2Base):
    """
    BLIP2 Llama model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "blip2_3d_stage2": "configs/models/blip2_3d/blip2_3d_stage2.yaml",
        "blip2_3d_caption": "configs/models/blip2_3d/blip2_3d_caption.yaml",
    }

    def __init__(
        self,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        point_cloud_encoder_model = None,
        freeze_point_cloud_encoder = True,
        max_cloud_size = 10000,
        num_query_token=32,
        llama_model_path="",
        prompt="",
        max_txt_len=32,
        qformer_encoder_layer=12,
        point_cloud_encoder_pretrain_model_path = None,
    ):
        super().__init__()

        # self.tokenizer = self.init_tokenizer()

        
        self.cloud_encoder, self.ln_cloud = self.init_cloud_encoder(
            point_cloud_encoder_model, max_cloud_size, drop_path_rate, use_grad_checkpoint, point_cloud_encoder_pretrain_model_path
        )
        if freeze_point_cloud_encoder:
            for name, param in self.cloud_encoder.named_parameters():
                param.requires_grad = False
            self.cloud_encoder = self.cloud_encoder.eval()
            self.cloud_encoder.train = disabled_train
            logging.info("freeze point cloud encoder")

        self.num_query_token = num_query_token
        # TODO : 不要固定为384 需要根据point cloud encoder的输出来确定
        self.Qformer, self.query_tokens = self.init_Qformer(
            self.num_query_token, self.cloud_encoder.enc_channels[-1], encoder_layer=qformer_encoder_layer
        )

        # 把一部分网络固定删去了, 不理解为什么
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        t1 = time.time()
        # 张老师的学生提供的 llama 读取代码
        # config = llama.LLaMAConfig.from_pretrained(llama_model_path)
        # self.llama_tokenizer = llama.LLaMATokenizer.from_pretrained(llama_model_path)
        # self.llama_model = llama.LLaMAForCausalLM.from_pretrained(llama_model_path, torch_dtype=torch.float16, config=config, state_dict=None)
        # transformers 的读取代码
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_model_path, torch_dtype=torch.float16)
        # 这里设置为True, 是为了避免 model.generate() 函数报错, 不知道为什么要这么做
        self.llama_model.config.use_cache = True
        # self.llama_model = None
        logging.info("load llama model spend time: {:.4f} s".format(time.time() - t1))
         
        if self.llama_model is not None:
            for name, param in self.llama_model.named_parameters():
                param.requires_grad = False
        # 给的 llama-Chinese 模型中使用的是 </s> 作为结束符
        self.eos_token_id = self.llama_tokenizer("</s>", add_special_tokens=False).input_ids[0]

        # 增加 pad token 不然二阶段会报错 ValueError: Asking to pad but the tokenizer does not have a padding token.
        self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token 

        self.llama_proj = nn.Linear(self.Qformer.config.hidden_size, 
                                  self.llama_model.config.hidden_size if self.llama_model is not None else 5120 
        )

        self.max_txt_len = max_txt_len

        # front_prompt + prompt + query + end_prompt + input_text
        self.front_prompt = "以下是一个描述任务的指令，请写一个完成该指令的适当回复。\n\n ### 指令:\n"
        self.end_prompt = "\n\n### 回复:"
        self.prompt = "请描述一下这个三维点云。"



        # 对预先设定的prompt进行一些处理，最终 forward 时用的输入是 query + prompt + input_text(mask 掉)
        
        self.prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt")
        # prompt_token 会自动加一个 bos token, 所以这里要减去1
        input_ids = self.prompt_tokens.input_ids[:, 1:]
        self.prompt_attention_mask = self.prompt_tokens.attention_mask[:, 1:]
        self.prompt_embedings = self.llama_model.model.embed_tokens(input_ids)

        assert self.prompt_attention_mask.shape[1] == self.prompt_embedings.shape[1],  \
                "prompt_attention_mask.shape : {} prompt_embedings.shape : {}".format(self.prompt_attention_mask.shape, self.prompt_embedings.shape)
        self.prompt_length = self.prompt_attention_mask.sum(1)
        logging.info("prompt {}, prompt_length : {}".format(self.prompt, self.prompt_length))

        # self.prompt = prompt
        # prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt")
        # self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, samples):
        image = samples["cloud"]
        device = image["coord"].device
        with self.maybe_autocast():
            # fake_cloud_encoder_result = torch.rand(image["coord"].shape[0], 256, 384).to(device)        # [batch_size, 256, 384]
            # image_embeds = self.ln_cloud(fake_cloud_encoder_result)
            image_embeds = self.ln_cloud(self.cloud_encoder(image))
            
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        # 这一行就相当于是把 learnable query 和 image 特征进行了交叉注意力
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        # query_output.last_hidden_state [B , 32, 768] -> [B, 32, 5120]
        inputs_llama = self.llama_proj(query_output.last_hidden_state)
        atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)  # [B, 32]

        self.llama_tokenizer.padding_side = "left"

        # text = [t + "\n" for t in samples["text_input"]]
        text = [self.prompt + t + "\n" for t in samples["text_input"]]

        llama_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        # llama_tokenizer 会自动在输入的文本前面加上 bos token, 所以这里要减去1
        llama_tokens.input_ids = llama_tokens.input_ids[:, 1:]
        llama_tokens.attention_mask = llama_tokens.attention_mask[:, 1:]

        # 把padding 的位置设置为 -100, 这样就不会计算loss
        # targets 尺寸为 [B, N]
        targets = llama_tokens.input_ids.masked_fill(
            llama_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        # 如果有一个固定的 prompt, 那么就把这个prompt对应的位置的  也设置为-100
        # 有点疑惑, llama_tokens 是根据输入的文字生成的, 并不包含 self.prompt， 所以如果把前N个位置认为是self.prompt，就会导致后续出错
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        # [B, 32]
        empty_targets = (
            torch.ones(atts_llama.size(), dtype=torch.long).to(device).fill_(-100)
        )
        # [B, 32] cat [B, N]  -> [B, 32+N]
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llama_model.model.embed_tokens(llama_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_llama, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_llama, llama_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        # output_text = self.llama_tokenizer.decode(outputs.logits[0][self.num_query_token:].argmax(1))
        output_text = self.llama_tokenizer.batch_decode(outputs.logits[:, self.num_query_token:].argmax(2), 
                                                        skip_special_tokens=True, clean_up_tokenization_spaces=True)
        # output_text 应该是一个长度等于 batch 的list
        return {"loss": loss, "output_text": output_text}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=1,
        top_p=0.95,
        top_k=50,
        no_repeat_ngram_size=6,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=0.7,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - cloud (dict) : A dictionary containing the following keys:
                    - coord (Tensor): The coordinates of the points in the point cloud. The shape is [B, N, 3].
                    - color (Tensor): The colors of the points in the point cloud. The shape is [B, N, 3].
                text_input (str): prompt text, it will be used in each cloud in the batch
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["cloud"]
        device = image["coord"].device
        with self.maybe_autocast():
            # fake_cloud_encoder_result = torch.rand(image["coord"].shape[0], 256, 384).to(device)        # [batch_size, 256, 384]
            # image_embeds = self.ln_cloud(fake_cloud_encoder_result)

            image_embeds = self.ln_cloud(self.cloud_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            inputs_opt = self.llama_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)

            if "text_input" in samples.keys():
                prompt = samples["text_input"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image_embeds.size(0)

            llama_tokens = self.llama_tokenizer(prompt, return_tensors="pt").to(device)
            input_embeds = self.llama_model.model.embed_tokens(llama_tokens.input_ids)
            input_embeds = torch.cat([inputs_opt, input_embeds], dim=1)         # 把 learnable query 和 prompt embedding 结果拼接起来
            attention_mask = torch.cat([atts_opt, llama_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = input_embeds.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = input_embeds.repeat_interleave(num_beams, dim=0)


            # generated_ids = self.llama_model.generate(
            #     llama_tokens.input_ids,
            #     inputs_embeds=query_embeds,
            #     attention_mask=attention_mask,
            #     max_new_tokens=max_length,
            #     do_sample = use_nucleus_sampling,
            #     num_beams = num_beams,
            #     top_k = top_k,
            #     top_p = top_p,
            #     temperature = temperature,
            #     no_repeat_ngram_size = no_repeat_ngram_size,
            # )
            # results = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            # return results

            outputs = self.llama_model.generate(
                # input_ids=input_ids,
                inputs_embeds=query_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            prompt_length = llama_tokens.input_ids.shape[1]
            output_text = self.llama_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
            )
            output_text = [text.strip() for text in output_text]
            return output_text


    @torch.no_grad()
    def generate_with_hidden_prompt(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=1,
        top_p=0.95,
        top_k=50,
        no_repeat_ngram_size=6,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=0.7,
    ):
        front_prompt = "以下是一个描述任务的指令，请写一个完成该指令的适当回复。\n\n ### 指令:\n"
        end_prompt = "\n\n### 回复:"
        image = samples["cloud"]
        device = image["coord"].device
        with self.maybe_autocast():
            # fake_cloud_encoder_result = torch.rand(image["coord"].shape[0], 256, 384).to(device)        # [batch_size, 256, 384]
            # image_embeds = self.ln_cloud(fake_cloud_encoder_result)

            image_embeds = self.ln_cloud(self.cloud_encoder(image))
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
    
            # Q-Former 的 learnable query 映射到 llama 的特征空间后得到的特征以及 attention mask
            inputs_query = self.llama_proj(query_output.last_hidden_state)
            atts_query = torch.ones(inputs_query.size()[:-1], dtype=torch.long).to(device)

            if "text_input" in samples.keys():
                prompt = samples["text_input"]
            else:
                prompt = self.prompt

            # 对起始的token进行处理
            front_tokens = self.llama_tokenizer(front_prompt, return_tensors="pt").to(device)
            front_embeds = self.llama_model.model.embed_tokens(front_tokens.input_ids)

            # 对输入的文字prompt进行处理，因为每个prompt都会添加一个 bos token, 所以需要把bos token去掉
            prompt = [prompt] * image_embeds.size(0)
            prompt_tokens = self.llama_tokenizer(prompt, return_tensors="pt").to(device)
            prompt_tokens.input_ids = prompt_tokens.input_ids[:, 1:]
            prompt_tokens.attention_mask = prompt_tokens.attention_mask[:, 1:]
            prompt_embeds = self.llama_model.model.embed_tokens(prompt_tokens.input_ids)
            

            # 对于结尾的 token 需要额外处理, 因为每个prompt都会添加一个 bos token, 而结尾的prompt要和前面的tokenizer结果拼接起来
            # 所以需要把结尾的prompt的bos token去掉
            end_tokens = self.llama_tokenizer(end_prompt, return_tensors="pt").to(device)
            end_tokens.input_ids = end_tokens.input_ids[:, 1:]
            end_tokens.attention_mask = end_tokens.attention_mask[:, 1:]
            end_embeds = self.llama_model.model.embed_tokens(end_tokens.input_ids)

            # 最后进行拼接  front + query + prompt + end
            input_embeds = torch.cat([front_embeds, inputs_query, prompt_embeds, end_embeds], dim=1)
            attention_mask = torch.cat([front_tokens.attention_mask, atts_query, prompt_tokens.attention_mask, end_tokens.attention_mask], dim=1)


            if use_nucleus_sampling:
                input_embeds = input_embeds.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                input_embeds = input_embeds.repeat_interleave(num_beams, dim=0)


            # generated_ids = self.llama_model.generate(
            #     llama_tokens.input_ids,
            #     inputs_embeds=query_embeds,
            #     attention_mask=attention_mask,
            #     max_new_tokens=max_length,
            #     do_sample = use_nucleus_sampling,
            #     num_beams = num_beams,
            #     top_k = top_k,
            #     top_p = top_p,
            #     temperature = temperature,
            #     no_repeat_ngram_size = no_repeat_ngram_size,
            # )
            # results = self.llama_tokenizer.batch_decode(generated_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            # return results

            outputs = self.llama_model.generate(
                # input_ids=input_ids,
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                do_sample=use_nucleus_sampling,
                top_p=top_p,
                temperature=temperature,
                num_beams=num_beams,
                max_new_tokens=max_length,
                min_length=min_length,
                eos_token_id=self.eos_token_id,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                num_return_sequences=num_captions,
            )

            output_text = self.llama_tokenizer.batch_decode(
                outputs, skip_special_tokens=True, spaces_between_special_tokens=False
            )
            output_text = [text.strip() for text in output_text]
            return output_text
        

    @classmethod
    def from_config(cls, cfg):
        num_query_token = cfg.get("num_query_token")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)

        prompt = cfg.get("prompt", "")
        max_txt_len = cfg.get("max_txt_len", 32)

        pritrained_llama_model_path = cfg.get("pretrained_llama_path", "")
        qformer_encoder_layer = cfg.get("qformer_encoder_layer", 12)
        point_cloud_encoder_model = cfg.get("point_cloud_encoder_model", "")
        point_cloud_encoder_pretrain_model_path = cfg.get("point_cloud_encoder_model_path", None)
        freeze_point_cloud_encoder = cfg.get("freeze_cloud_encoder", True)

        model = cls(
            point_cloud_encoder_model = point_cloud_encoder_model,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            freeze_point_cloud_encoder = freeze_point_cloud_encoder,
            num_query_token=num_query_token,
            llama_model_path=pritrained_llama_model_path,
            prompt=prompt,
            max_txt_len=max_txt_len,
            qformer_encoder_layer=qformer_encoder_layer,
            point_cloud_encoder_pretrain_model_path = point_cloud_encoder_pretrain_model_path,
        )

        load_finetuned = cfg.get("load_finetuned", False)
        load_pretrained = cfg.get("load_pretrained", False)
        if(load_finetuned):
            logging.info("load fintuned blip2_llama model from {}".format(cfg.get("finetuned", None)))
            model.load_checkpoint_from_config(cfg)
        elif(load_pretrained):
            logging.info("load pretrained blip2_llama model from {}".format(cfg.get("pretrained", None)))
            model.load_checkpoint_from_config(cfg)

        return model
