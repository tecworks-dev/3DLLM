'''
Author: Diantao Tu
Date: 2023-04-24 18:24:23
'''
import torch 
from lavis.models import load_model_and_preprocess
from lavis.common.registry import registry
from flask import request
from flask_api import FlaskAPI
from typing import Dict
import numpy as np

app = FlaskAPI(__name__)

device = torch.device("cuda:2")
model, vis_processors, _ = load_model_and_preprocess(
    name="blip2_llama", model_type="blip2_3d_caption", is_eval=True, device=device
)
del _
model = model.to(device)

def load_cloud(cloud_path:str) -> Dict:
    cloud = torch.load(cloud_path)

    if(isinstance(cloud, tuple)):
        cloud = {"coord": cloud[0], "color": cloud[1], "semantic_gt": cloud[2]}
        cloud["color"] = ((cloud["color"] + 1) * 127.5).astype(np.uint8)
        cloud["color"] = cloud["color"].astype(np.float64)
        cloud["coord"] = cloud["coord"].astype(np.float64)
        # 把 coord 中的值归一化到 [-5, 5] 之间
        max_value = np.max(cloud["coord"])
        min_value = np.min(cloud["coord"])
        final_value = max(abs(max_value), abs(min_value))
        cloud["coord"] = cloud["coord"] / final_value  * 5.0
    
    for k in cloud.keys():
        if(isinstance(cloud[k], torch.Tensor)):
            cloud[k] = cloud[k].to(device)
            cloud[k] = cloud[k].unsqueeze(0)
    return cloud

@app.route("/caption", methods=['GET','POST'])
def caption():
    if(request.method == 'POST'):
        data = request.data
        result = {"data_keys": list(data.keys())}
        if "cloud_path" not in data:
            return {"error": "cloud_path not found"}, 400
        cloud_path = data["cloud_path"]
        cloud = load_cloud(cloud_path)
        print("cloud_device", cloud["coord"].device)
        cloud = vis_processors["eval"](cloud)
        print("cloud_device", cloud["coord"].device)
        # print("model_device", model.device)
        caption = model.generate_with_hidden_prompt({"cloud":cloud, "text_input": "请描述一下这个三维场景。"}, max_length=100, num_beams=1, max_sentences = 2)
        result["caption"] = caption
        return result
    elif(request.method == 'GET'):
        return {"error": "nothing to GET"}

@app.route("/test")
def test():
    return {"test": "test_test"}

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)