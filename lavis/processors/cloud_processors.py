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

import scipy.ndimage
import scipy.interpolate
import scipy.stats
import numpy as np
import torch
from collections.abc import Sequence, Mapping

class PointCloudAugmentation():
    def __init__(self,):
        pass
    
    def RandomSample(self, data_dict, max_num = 80000):
        if "coord" in data_dict.keys():
            if data_dict["coord"].shape[0] > max_num:
                choice = np.random.choice(data_dict["coord"].shape[0], max_num, replace=False)
                for key in data_dict.keys():
                    data_dict[key] = data_dict[key][choice]
        return data_dict
    
    def CenterShift(self, data_dict, apply_z=True):
        if "coord" in data_dict.keys():
            x_min, y_min, z_min = data_dict["coord"].min(axis=0)
            x_max, y_max, _ = data_dict["coord"].max(axis=0)
            if apply_z:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, z_min]
            else:
                shift = [(x_min + x_max) / 2, (y_min + y_max) / 2, 0]
            data_dict["coord"] -= shift
        return data_dict

    def RandomScale(self, data_dict, scale=None, anisotropic=False):
        if "coord" in data_dict.keys():
            scale = np.random.uniform(scale[0], scale[1], 3 if anisotropic else 1)
            data_dict["coord"] *= scale
        return data_dict

    def RandomFlip(self, data_dict, p=0.5):
        if "coord" in data_dict.keys():
            if np.random.rand() < p:
                data_dict["coord"][:, 0] = -data_dict["coord"][:, 0]
            if np.random.rand() < p:
                data_dict["coord"][:, 1] = -data_dict["coord"][:, 1]
        return data_dict
    
    def RandomJitter(self, data_dict, sigma=0.01, clip=0.05):
        assert (clip > 0)
        if "coord" in data_dict.keys():
            jitter = np.clip(sigma * np.random.randn(data_dict["coord"].shape[0], 3), -clip, clip)
            data_dict["coord"] += jitter
        return data_dict

    def ChromaticAutoConstrast(self, data_dict, p=0.2, blend_factor=None):
        if "color" in data_dict.keys() and np.random.rand() < p:
            lo = np.min(data_dict["color"], 0, keepdims=True)
            hi = np.max(data_dict["color"], 0, keepdims=True)
            scale = 255 / (hi - lo)
            contrast_feat = (data_dict["color"][:, :3] - lo) * scale
            blend_factor = np.random.rand() if blend_factor is None else blend_factor
            data_dict["color"][:, :3] = (1 - blend_factor) * data_dict["color"][:, :3] + blend_factor * contrast_feat
        return data_dict

    def ChromaticTranslation(self, data_dict, p=0.95, ratio=0.05):
        if "color" in data_dict.keys() and np.random.rand() < p:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * ratio
            data_dict["color"][:, :3] = np.clip(tr + data_dict["color"][:, :3], 0, 255)
        return data_dict
    
    def ChromaticJitter(self, data_dict, p=0.95, std=0.005):
        if "color" in data_dict.keys() and np.random.rand() < p:
            noise = np.random.randn(data_dict["color"].shape[0], 3)
            noise *= std * 255
            data_dict["color"][:, :3] = np.clip(noise + data_dict["color"][:, :3], 0, 255)
        return data_dict

    def fnv_hash_vec(self, arr):
        assert arr.ndim == 2
        # Floor first for negative coordinates
        arr = arr.copy()
        arr = arr.astype(np.uint64, copy=False)
        hashed_arr = np.uint64(14695981039346656037) * np.ones(arr.shape[0], dtype=np.uint64)
        for j in range(arr.shape[1]):
            hashed_arr *= np.uint64(1099511628211)
            hashed_arr = np.bitwise_xor(hashed_arr, arr[:, j])
        return hashed_arr
    
    def ravel_hash_vec(self, arr):
        assert arr.ndim == 2
        arr = arr.copy()
        arr -= arr.min(0)
        arr = arr.astype(np.uint64, copy=False)
        arr_max = arr.max(0).astype(np.uint64) + 1

        keys = np.zeros(arr.shape[0], dtype=np.uint64)
        # Fortran style indexing
        for j in range(arr.shape[1] - 1):
            keys += arr[:, j]
            keys *= arr_max[j + 1]
        keys += arr[:, -1]
        return keys

    def Voxelize(self, data_dict, voxel_size=0.05, hash_type="fnv", mode='train', 
                 keys=("coord", "normal", "color", "label"), return_inverse=False, 
                 return_discrete_coord=False, return_min_coord=False):
        hash = self.fnv_hash_vec if hash_type == "fnv" else self.ravel_hash_vec
        assert mode in ["train", "val", "test"]
        
        assert "coord" in data_dict.keys()
        discrete_coord = np.floor(data_dict["coord"] / np.array(voxel_size)).astype(np.int)
        min_coord = discrete_coord.min(0) * np.array(voxel_size)
        discrete_coord -= discrete_coord.min(0)
        key = hash(discrete_coord)
        idx_sort = np.argsort(key)
        key_sort = key[idx_sort]
        _, inverse, count = np.unique(key_sort, return_inverse=True, return_counts=True)

        if mode == "train" or mode == "val":
            idx_select = np.cumsum(np.insert(count, 0, 0))[0:-1] + np.random.randint(0, count.max(), count.size) % count
            idx_unique = idx_sort[idx_select]
            if return_discrete_coord:
                data_dict["discrete_coord"] = discrete_coord[idx_unique]
            if return_inverse:
                data_dict["mask"] = np.zeros_like(inverse)
                data_dict["mask"][idx_unique] = 1
                data_dict["inverse"] = np.zeros_like(inverse)
                data_dict["inverse"][idx_sort] = inverse
                data_dict["length"] = np.array(inverse.shape)
            if self.return_min_coord:
                data_dict["min_coord"] = min_coord.reshape([1, 3])
            for key in self.keys:
                data_dict[key] = data_dict[key][idx_unique]
            return data_dict
        elif mode == "test":
            data_part_list = []
            for i in range(count.max()):
                idx_select = np.cumsum(np.insert(count, 0, 0)[0:-1]) + i % count
                idx_part = idx_sort[idx_select]
                data_part = dict(index=idx_part)
                for key in self.keys:
                    data_part[key] = data_dict[key][idx_part]
                if self.return_discrete_coord:
                    data_part["discrete_coord"] = discrete_coord[idx_part]
                if self.return_inverse:
                    data_part["inverse"] = np.zeros_like(inverse)
                    data_part["inverse"][idx_sort] = inverse
                    data_part["length"] = np.array(inverse.shape)
                if self.return_min_coord:
                    data_part["min_coord"] = min_coord.reshape([1, 3])
                data_part_list.append(data_part)
            return data_part_list
        else:
            raise NotImplementedError

    def SphereCrop(self, data_dict, point_max=80000, sample_rate=None, mode="random"):
        assert mode in ["random", "center", "all"]
        point_max = int(sample_rate * data_dict["coord"].shape[0]) \
            if sample_rate is not None else point_max
        assert "coord" in data_dict.keys()
        if mode == "all":
            if "index" not in data_dict.keys():
                data_dict["index"] = np.arange(data_dict["coord"].shape[0])
            data_part_list = []
            if data_dict["coord"].shape[0] > point_max:
                coord_p, idx_uni = np.random.rand(data_dict["coord"].shape[0]) * 1e-3, np.array([])
                while idx_uni.size != data_dict["index"].shape[0]:
                    init_idx = np.argmin(coord_p)
                    dist2 = np.sum(np.power(data_dict["coord"] - data_dict["coord"][init_idx], 2), 1)
                    idx_crop = np.argsort(dist2)[:point_max]

                    data_crop_dict = dict()
                    if "coord" in data_dict.keys():
                        data_crop_dict["coord"] = data_dict["coord"][idx_crop]
                    if "discrete_coord" in data_dict.keys():
                        data_crop_dict["discrete_coord"] = data_dict["discrete_coord"][idx_crop]
                    if "normal" in data_dict.keys():
                        data_crop_dict["normal"] = data_dict["normal"][idx_crop]
                    if "color" in data_dict.keys():
                        data_crop_dict["color"] = data_dict["color"][idx_crop]
                    data_crop_dict["weight"] = dist2[idx_crop]
                    data_crop_dict["index"] = data_dict["index"][idx_crop]
                    data_part_list.append(data_crop_dict)

                    delta = np.square(1 - data_crop_dict["weight"] / np.max(data_crop_dict["weight"]))
                    coord_p[idx_crop] += delta
                    idx_uni = np.unique(np.concatenate((idx_uni, data_crop_dict["index"])))
            else:
                data_crop_dict = data_dict.copy()
                data_crop_dict["weight"] = np.zeros(data_dict["coord"].shape[0])
                data_crop_dict["index"] = data_dict["index"]
                data_part_list.append(data_crop_dict)
            return data_part_list
        # mode is "random" or "center"
        elif data_dict["coord"].shape[0] > point_max:
            if mode == "random":
                center = data_dict["coord"][np.random.randint(data_dict["coord"].shape[0])]
            elif mode == "center":
                center = data_dict["coord"][data_dict["coord"].shape[0] // 2]
            else:
                raise NotImplementedError
            idx_crop = np.argsort(np.sum(np.square(data_dict["coord"] - center), 1))[:point_max]
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx_crop]
            if "origin_coord" in data_dict.keys():
                data_dict["origin_coord"] = data_dict["origin_coord"][idx_crop]
            if "discrete_coord" in data_dict.keys():
                data_dict["discrete_coord"] = data_dict["discrete_coord"][idx_crop]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx_crop]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx_crop]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx_crop]
            if "label" in data_dict.keys():
                data_dict["label"] = data_dict["label"][idx_crop] \
                    if len(data_dict["label"]) != 1 else data_dict["label"]
        return data_dict

    def NormalizeColor(self, data_dict):
        if "color" in data_dict.keys():
            data_dict["color"] = data_dict["color"] / 127.5 - 1
        return data_dict

    def ToTensor(self, data_dict):
        if isinstance(data_dict, Mapping):
            result = {sub_key: self(item) for sub_key, item in data_dict.items()}
            return result
        elif isinstance(data_dict, Sequence):
            result = [self(item) for item in data_dict]
            return result
        elif isinstance(data_dict, torch.Tensor):
            return data_dict
        elif isinstance(data_dict, str):
            return data_dict
        elif isinstance(data_dict, int):
            return torch.LongTensor([data_dict])
        elif isinstance(data_dict, float):
            return torch.FloatTensor([data_dict])
        elif isinstance(data_dict, np.ndarray) and np.issubdtype(data_dict.dtype, np.int):
            return torch.from_numpy(data_dict).long()
        elif isinstance(data_dict, np.ndarray) and np.issubdtype(data_dict.dtype, np.floating):
            return torch.from_numpy(data_dict).float()
        else:
            raise TypeError(f'type {type(data_dict)} cannot be converted to tensor.')
        
    def Collect(self,data_dict, keys, offset_keys_dict=None, **kwargs):
        if offset_keys_dict is None:
            offset_keys_dict = dict(offset="coord")
        data = dict()
        if isinstance(keys, str):
            keys = [keys]
        for key in keys:
            data[key] = data_dict[key]
        for key, value in offset_keys_dict.items():
            data[key] = torch.tensor([data_dict[value].shape[0]])
        for name, keys in kwargs.items():
            name = name.replace("keys", "")
            data[name] = torch.cat([data_dict[key] for key in keys], 0)
        return data

    def Copy(self, data_dict, keys_dict=None):
        if keys_dict is None:
            keys_dict = dict(coord="origin_coord", label="origin_label")
        for key, value in keys_dict.items():
            if isinstance(data_dict[key], np.ndarray):
                data_dict[value] = data_dict[key].copy()
            elif isinstance(data_dict[key], torch.Tensor):
                data_dict[value] = data_dict[key].clone().detach()
            else:
                data_dict[value] = copy.deepcopy(data_dict[key])
        return data_dict


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
        # 把开头、结尾的空格去掉
        caption = caption.strip()
        # 把回车符号删掉
        caption = caption.replace("\r", "")
        # 把中间的空格都去掉
        caption = caption.replace(" ", "")

        return caption


@registry.register_processor("cloud_train")
class CloudTrainProcessor(BaseProcessor):
    def __init__(self, max_size:int):
        super().__init__()
        self.max_size = max_size

    # TODO 对点云进行裁剪, 降采样, 数据增强等等
    def __call__(self, point_cloud):

        coord = point_cloud["coord"]
        color = point_cloud["color"]
        if "semantic_gt" in point_cloud.keys():
            label = point_cloud["semantic_gt"].reshape(-1)
        else:
            label = np.zeros(coord.shape[0])
        pcd_dict = dict(coord=coord, color=color, label=label)

        # 接下来是对点云进行数据增强
        pcd_dict = PointCloudAugmentation.CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = PointCloudAugmentation.RandomScale(scale=[0.9, 1.1])(pcd_dict)
        pcd_dict = PointCloudAugmentation.RandomFlip(p=0.5)(pcd_dict)
        pcd_dict = PointCloudAugmentation.RandomJitter(sigma=0.005, clip=0.02)(pcd_dict)
        pcd_dict = PointCloudAugmentation.ChromaticAutoContrast(p=0.2, blend_factor=None)(pcd_dict)
        pcd_dict = PointCloudAugmentation.ChromaticTranslation(p=0.95, ratio=0.05)(pcd_dict)
        pcd_dict = PointCloudAugmentation.ChromaticJitter(p=0.95, std=0.05)(pcd_dict)
        pcd_dict = PointCloudAugmentation.Voxelize(voxel_size=0.04, hash_type='fnv', mode='train',
                    keys=("coord", "color", "label"), return_discrete_coord=True)(pcd_dict)
        pcd_dict = PointCloudAugmentation.SphereCrop(point_max=100000, mode='random')(pcd_dict)

        # 随机下采样是我自己实现的，而对于其他的数据变换方式是原来的PointTransformer2的代码
        # pcd_dict = PointCloudAugmentation.RandomSample(max_num=65536)(pcd_dict)
        pcd_dict = PointCloudAugmentation.CenterShift(apply_z=False)(pcd_dict)
        pcd_dict = PointCloudAugmentation.NormalizeColor()(pcd_dict)
        pcd_dict = PointCloudAugmentation.ToTensor()(pcd_dict)
        pcd_dict = PointCloudAugmentation.Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)

        return pcd_dict 
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_size = cfg.get("max_size", 2048)

        return cls(max_size=max_size)
    

@registry.register_processor("cloud_test")
class CloudTestProcessor(BaseProcessor):
    def __init__(self, max_size:int):
        super().__init__()
        self.max_size = max_size

    # TODO 对点云进行裁剪, 降采样等等
    def __call__(self, point_cloud):

        coord = point_cloud["coord"]
        color = point_cloud["color"]
        if "semantic_gt" in point_cloud.keys():
            label = point_cloud["semantic_gt"].reshape(-1)
        else:
            label = np.zeros(coord.shape[0])
        pcd_dict = dict(coord=coord, color=color, label=label)

        # 接下来是对点云进行数据增强
        pcd_dict = PointCloudAugmentation.CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = PointCloudAugmentation.Copy(keys_dict={"coord": "origin_coord", "label": "origin_label"}(pcd_dict))
        pcd_dict = PointCloudAugmentation.Voxelize(voxel_size=0.04, hash_type='fnv', mode='train',
                    keys=("coord", "color", "label"), return_discrete_coord=True)(pcd_dict)
        pcd_dict = PointCloudAugmentation.CenterShift(apply_z=False)(pcd_dict)
        pcd_dict = PointCloudAugmentation.NormalizeColor()(pcd_dict)
        pcd_dict = PointCloudAugmentation.ToTensor()(pcd_dict)
        pcd_dict = PointCloudAugmentation.Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)

        return point_cloud 
    
    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        max_size = cfg.get("max_size", 2048)

        return cls(max_size=max_size)
    