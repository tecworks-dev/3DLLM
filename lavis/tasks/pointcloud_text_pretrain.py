'''
Author: Diantao Tu
Date: 2023-04-15 19:16:02
'''
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("pointcloud_text_pretrain")
class PointCloudTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    def evaluation(self, model, data_loader, cuda_enabled=True):
        pass