'''
Author: Diantao Tu
Date: 2023-04-17 23:25:05
'''
import torch
from PIL import Image
import requests
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry

import logging 

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

# 从命令行读取参数
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", help="")
args = parser.parse_args()


device = torch.device(args.device)

model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_llama", model_type="blip2_3d_caption", is_eval=True, device=device
)


cloud_path = "/data3/rmq/points_text_datasets/S3DIS/s3dis_processed/Area_1/hallway_4.pth"
logging.info("loading cloud")
cloud = torch.load(cloud_path)
logging.info(cloud["coord"].shape)

logging.info("pre-processing cloud")
cloud = vis_processors["eval"](cloud)
logging.info(cloud["coord"].shape)

for k in cloud.keys():
    if(isinstance(cloud[k], torch.Tensor)):
        cloud[k] = cloud[k].to(device)
        cloud[k] = cloud[k].unsqueeze(0)

result = model.generate({"cloud":cloud}, max_length=50)

print(result)