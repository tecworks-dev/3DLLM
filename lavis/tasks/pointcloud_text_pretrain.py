'''
Author: Diantao Tu
Date: 2023-04-15 19:16:02
'''
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask
import torch
import torch.distributed as dist
from lavis.common.dist_utils import get_rank, get_world_size, is_main_process, is_dist_avail_and_initialized
from lavis.common.logger import MetricLogger
from lavis.datasets.data_utils import prepare_sample
from typing import Dict, List, Tuple

@registry.register_task("pointcloud_text_pretrain")
class PointCloudTextPretrainTask(BaseTask):
    def __init__(self):
        super().__init__()

    # 对于第一阶段的训练来说, 验证没有什么直观的方法, 还得是计算loss, 只不过是在验证集上计算
    def evaluation(self, model, data_loader, cuda_enabled=True):
        # return
        total_steps = len(data_loader)

        if not hasattr(data_loader, "__next__"):
            # convert to iterator if not already
            data_loader = iter(data_loader)

        metric_logger = MetricLogger(delimiter="  ")
        header = "Evaluation"
        # TODO make it configurable
        print_freq = 10

        results = []

        for samples in metric_logger.log_every(range(total_steps), print_freq, header):
            samples = next(data_loader)
            samples = prepare_sample(samples, cuda_enabled=cuda_enabled)
            with torch.cuda.amp.autocast(enabled=True):
                # eval_output = self.train_step(model=model, samples=samples)
                loss = model(samples)["loss"]
            results.append(loss)

        if is_dist_avail_and_initialized():
            dist.barrier()
            for r in results:
                dist.all_reduce(r, op=dist.ReduceOp.SUM)
                r /= get_world_size()

        return results
    
    def after_evaluation(self, val_result:List[torch.Tensor], split_name, epoch) -> Dict[str, float]:
        metrics = dict()
        # 计算val_result的平均值
        val_result = torch.stack(val_result)
        val_result = val_result.mean()
        # 在runner_base 中, 会比较agg_metrics的值, 来决定最好的模型
        # 但是里面认为 agg_metrics 是一个正数, 而且越大越好, 而实际上这里取得是loss, 是一个越小越好的正数
        # 为了符合评估的要求, 这里用一个很大的正数减去loss, 使得loss越小, agg_metrics越大
        val_result = 100000 - val_result
        metrics["agg_metrics"] = val_result.item()

        # 保存结果
        if is_main_process():
            result_dir = registry.get_path("result_dir")
            log = "Epoch: {}, split: {}, val_result: {}".format(epoch, split_name, 100000-val_result.item())
            with open(result_dir + "/log.txt", "a") as f:
                f.write(log + "\n")
            print(log)
        return metrics
        