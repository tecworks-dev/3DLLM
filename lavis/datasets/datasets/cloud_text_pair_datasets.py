'''
Author: Diantao Tu
Date: 2023-04-15 20:28:14
'''

import os
from collections import OrderedDict
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image

def cloud_processor(cloud:torch.Tensor) -> torch.Tensor:
    '''
    对输入的点云进行处理, 可以是降采样, 数据增强等等
    Args:
        cloud: (N, 3) tensor
    Returns:
        cloud: (N, 3) tensor
    '''


    return cloud

def text_processor(text:str, max_length:int=-1) -> str:
    '''
    对输入的文本进行处理, 主要是去除空格, 换行符等等
    max_length: 最大长度, 超过的部分截断, -1表示无限制 
    '''

    return text

class CloudTextPairDataset():
    def __init__(self, vis_processor, text_processor, text_prompt:str=""):
        self.text_prompt = text_prompt
        self.vis_processor = vis_processor  # 其实是对点云进行处理, 只是名字叫做 vis_processor
        self.text_processor = text_processor
        
        

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        image_path = os.path.join(self.vis_root, ann["image"])
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        caption = self.text_processor(ann["caption"])

        return {"image": image, "text_input": caption}

    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": os.path.basename(ann["image"]),
                "caption": ann["caption"],
                "image": sample["image"],
            }
        )