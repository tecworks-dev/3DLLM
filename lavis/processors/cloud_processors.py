'''
Author: Diantao Tu
Date: 2023-04-15 20:51:25
'''
from lavis.common.registry import registry
from lavis.processors.base_processor import BaseProcessor
from lavis.processors.randaugment import RandomAugment
from omegaconf import OmegaConf
from torchvision import transforms
from torchvision.transforms.functional import InterpolationMode


@registry.register_processor("chinese_caption")
class ChineseCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        # super().__init__()
        self.prompt = prompt            # 对每个caption添加的前缀
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)
    
    def pre_caption(self, caption):
        # TODO 对中文caption进行预处理
        return caption


@registry.register_processor("cloud_train")
class CloudTrainProcessor(BaseProcessor):
    def __init__(self, max_size:int):
        super().__init__()
        self.max_size = max_size

    # TODO 对点云进行裁剪, 降采样, 数据增强等等
    def __call__(self, point_cloud):
        return point_cloud 
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_size = cfg.get("max_size", 2048)

        return cls(max_size=max_size)
    