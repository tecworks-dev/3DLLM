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
from lavis.models.blip2_models.modeling_opt import OPTForCausalLM, OPTConfig
from transformers import AutoTokenizer
from transformers import LlamaConfig, LlamaModel, LlamaForCausalLM
import time

@registry.register_model("blip2_llama")
class Blip2Llama(Blip2Base):
    """
    BLIP2 Llama model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "blip2_3d_stage2": "configs/models/blip2_3d/blip2_3d_stage2.yaml",
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

        self.tokenizer = self.init_tokenizer()

        
        self.cloud_encoder, self.ln_cloud = self.init_cloud_encoder(
            point_cloud_encoder_model, max_cloud_size, drop_path_rate, use_grad_checkpoint, point_cloud_encoder_pretrain_model_path
        )
        if freeze_point_cloud_encoder:
            for name, param in self.cloud_encoder.named_parameters():
                param.requires_grad = False
            self.cloud_encoder = self.cloud_encoder.eval()
            self.cloud_encoder.train = disabled_train
            logging.info("freeze point cloud encoder")

        # TODO : 不要固定为384 需要根据point cloud encoder的输出来确定
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, 384, encoder_layer=qformer_encoder_layer
        )

        # 把一部分网络固定删去了, 不理解为什么
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None

        t1 = time.time()
        self.llama_tokenizer = AutoTokenizer.from_pretrained(llama_model_path, use_fast=False)
        self.llama_model = LlamaForCausalLM.from_pretrained(llama_model_path, torch_dtype=torch.float16)

        # self.llama_model = LlamaForCausalLM(config=LlamaConfig())
        # self.llama_model = None
        logging.info("load llama model spend time: {:.4f} s".format(time.time() - t1))
         

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        self.eos_token_id = self.llama_tokenizer(
            "\n", add_special_tokens=False
        ).input_ids[0]

        # self.opt_proj = nn.Linear(
        #     self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        # )

        self.opt_proj = nn.Linear(self.Qformer.config.hidden_size, 
                                  self.llama_model.config.hidden_size if self.llama_model is not None else 5120 
        )

        self.max_txt_len = max_txt_len
        self.prompt = prompt
        prompt_tokens = self.llama_tokenizer(self.prompt, return_tensors="pt")
        self.prompt_length = prompt_tokens.attention_mask.sum(1)

    def forward(self, samples):
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

        inputs_opt = self.opt_proj(query_output.last_hidden_state)
        atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)

        self.llama_tokenizer.padding_side = "right"

        text = [t + "\n" for t in samples["text_input"]]

        llama_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
        ).to(device)

        targets = llama_tokens.input_ids.masked_fill(
            llama_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )
        if self.prompt:
            targets[:, : self.prompt_length] = -100  # do not apply loss to the prompt

        empty_targets = (
            torch.ones(atts_opt.size(), dtype=torch.long).to(device).fill_(-100)
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        inputs_embeds = self.llama_model.model.embed_tokens(llama_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_opt, inputs_embeds], dim=1)
        attention_mask = torch.cat([atts_opt, llama_tokens.attention_mask], dim=1)

        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        loss = outputs.loss

        return {"loss": loss}

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=5,
        max_length=30,
        min_length=1,
        top_p=0.9,
        repetition_penalty=1.0,
        length_penalty=1.0,
        num_captions=1,
        temperature=1,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
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

            inputs_opt = self.opt_proj(query_output.last_hidden_state)
            atts_opt = torch.ones(inputs_opt.size()[:-1], dtype=torch.long).to(device)

            if "prompt" in samples.keys():
                prompt = samples["prompt"]
            else:
                prompt = self.prompt

            prompt = [prompt] * image_embeds.size(0)

            opt_tokens = self.llama_tokenizer(prompt, return_tensors="pt").to(device)
            input_ids = opt_tokens.input_ids
            attention_mask = torch.cat([atts_opt, opt_tokens.attention_mask], dim=1)

            if use_nucleus_sampling:
                query_embeds = inputs_opt.repeat_interleave(num_captions, dim=0)
                num_beams = 1
            else:
                query_embeds = inputs_opt.repeat_interleave(num_beams, dim=0)

            outputs = self.llama_model.generate(
                input_ids=input_ids,
                query_embeds=query_embeds,
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

            prompt_length = opt_tokens.input_ids.shape[1]
            output_text = self.llama_tokenizer.batch_decode(
                outputs[:, prompt_length:], skip_special_tokens=True
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

        if cfg.get("load_finetuned", False) or cfg.get("load_pretrained", False):
            model.load_checkpoint_from_config(cfg)

        return model
