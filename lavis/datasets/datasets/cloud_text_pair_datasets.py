'''
Author: Diantao Tu
Date: 2023-04-15 20:28:14
'''

import os
from collections import OrderedDict
import torch

from lavis.datasets.datasets.base_dataset import BaseDataset
from PIL import Image
from typing import List, Tuple, Dict
import json

def load_point_cloud(path:str) -> Dict[str, torch.Tensor]:
    """
    从文件中读取点云
    path: 点云路径,绝对路径
    return: 点云, shape: (N, 3)
    """
    # cloud = torch.zeros((2048, 3))
    #TODO: 以后需要完成关于不同格式的点云的读取，目前torch.load应该只能读取pth格式的点云
    cloud = torch.load(path)
    return cloud

def load_pairs(path:str) -> List[Tuple[str, str]]:
    """
    从文件中读取点云和对应的caption
    path: 文件路径, 绝对路径
    return: List[List[str, str]], 每个元素是一个List, 第一个元素是点云路径, 第二个元素是对应的caption
    """
    pairs = []
    with open(path, "r") as f:
        init_pairs = json.load(f)
        for key in init_pairs:
            pairs.append([key, init_pairs[key]])
    
    return pairs

# 虽然这里继承了 BaseDataset, 但这个类覆写了__getitem__和__len__方法, 这里继承的主要目的是之后dataloader建立的时候不报错
# 由于时间关系, 我没有看dataloader是怎么建立的, 所以就直接继承了BaseDataset
class CloudTextPairDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, text_prompt:str="", path:str=""):
        """
        点云和对应的caption的数据集, 里面存储了点云和对应的caption, 如果点云对应于N个caption, 那么就会有N个元素具有相同的点云不同的caption
        vis_processor: 对点云进行处理的函数, 叫做 vis_processor 只是为了和原本的代码保持一致
        text_processor: 对caption进行处理的函数
        text_prompt: caption的前缀, 最终的caption是 text_prompt + caption
        path: 存储点云和对应的caption的文件的路径, 绝对路径
        """
        super().__init__()
        self.text_prompt = text_prompt
        self.vis_processor = vis_processor  # 其实是对点云进行处理, 只是名字叫做 vis_processor
        self.text_processor = text_processor
        self.cloud_text_pairs = load_pairs(path)
        # self.cloud_text_pairs = [["test.pcd", "frist test caption"],["test2.pcd", "second test caption"]]
        
        

    def __getitem__(self, index):

        pair = self.cloud_text_pairs[index]
        cloud = load_point_cloud(pair[0])
        caption = pair[1]

        cloud = self.vis_processor(cloud)
        caption = self.text_processor(caption)

        return {"cloud": cloud, "text_input": caption, "cloud_path": pair[0]}

        image = torch.ones((3, 224, 224))

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
    
    def __len__(self):
        return len(self.cloud_text_pairs)


if __name__ == "__main__":
    path = "/data3/rmq/points_text_datasets/S3DIS/text_embed/ZN-s3dis_text_absolute.json"
    cloud_text_pairs = load_pairs(path)
    print("true")