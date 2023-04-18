# 1.第一步，需要修改blip2.py的第70行，即点云模型的encoder定义部分

## 1) 对于第一次训练的小模型S3DIS，网络结构如下**

```python
            cloud_encoder = PointTransformerV2(in_channels=6,
                                             num_classes=13,
                                             patch_embed_depth=2,
                                             patch_embed_channels=48,
                                             patch_embed_groups=6,
                                             patch_embed_neighbours=16,
                                             enc_depths=(2, 6, 2),
                                             enc_channels=(96, 192, 384),
                                             enc_groups=(12, 24, 48),
                                             enc_neighbours=(16, 16, 16),
                                             dec_depths=(1, 1, 1),
                                             dec_channels=(48, 96, 192),
                                             dec_groups=(6, 12, 24),
                                             dec_neighbours=(16, 16, 16),
                                             grid_sizes=(0.1, 0.2, 0.4),
                                             attn_qkv_bias=True,
                                             pe_multiplier=False,
                                             pe_bias=True,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.3,
                                             enable_checkpoint=False,
                                             unpool_backend="interp",
                                             num_features=256,
                                             checkpoint_path=pretrained_model_path,)
```

## 2) 对于第二次训练的模型scannet，甚至以后其他更多的数据，都采用如下网络结构（但是需注意sacannet是20个语义类别，其他的数据集类别可能不一样，有时候需要修改一下其中的num_classes）：

```python
            cloud_encoder = PointTransformerV2(in_channels=6,
                                             num_classes=20,
                                             patch_embed_depth=1,
                                             patch_embed_channels=48,
                                             patch_embed_groups=6,
                                             patch_embed_neighbours=8,
                                             enc_depths=(2, 2, 6, 2),
                                             enc_channels=(96, 192, 384, 512),
                                             enc_groups=(12, 24, 48, 64),
                                             enc_neighbours=(16, 16, 16, 16),
                                             dec_depths=(1, 1, 1, 1),
                                             dec_channels=(48, 96, 192, 384),
                                             dec_groups=(6, 12, 24, 48),
                                             dec_neighbours=(16, 16, 16, 16),
                                             grid_sizes=(0.06, 0.15, 0.375, 0.9375),
                                             attn_qkv_bias=True,
                                             pe_multiplier=False,
                                             pe_bias=True,
                                             attn_drop_rate=0.,
                                             drop_path_rate=0.3,
                                             enable_checkpoint=False,
                                             unpool_backend="map",
                                             num_features=256,
                                             checkpoint_path=pretrained_model_path,)
```



# 2.第二步，scannet点云的预处理可以修改一下，即cloud_processors.py，大致修改为如下情况

## 1）对于训练集
```python
        # 接下来是对点云进行数据增强
        # 随机下采样是我自己实现的，而对于其他的数据变换方式是原来的PointTransformer2的代码
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)        
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = RandomDropout(dropout_ratio=0.2, dropout_application_ratio=0.2)(pcd_dict)
        pcd_dict = RandomRotate(angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5)(pcd_dict)
        pcd_dict = RandomRotate(angle=[-1/64, 1/64], axis='x', p=0.5)(pcd_dict)
        pcd_dict = RandomRotate(angle=[-1/64, 1/64], axis='y', p=0.5)(pcd_dict)
        pcd_dict = RandomScale(scale=[0.9, 1.1])(pcd_dict)
        pcd_dict = RandomFlip(p=0.5)(pcd_dict)
        pcd_dict = RandomJitter(sigma=0.005, clip=0.02)(pcd_dict)
        pcd_dict = ElasticDistortion(distortion_params=[[0.2, 0.4], [0.8, 1.6]])(pcd_dict)
        pcd_dict = ChromaticAutoContrast(p=0.2, blend_factor=None)(pcd_dict)
        pcd_dict = ChromaticTranslation(p=0.95, ratio=0.05)(pcd_dict)
        pcd_dict = ChromaticJitter(p=0.95, std=0.05)(pcd_dict)        
        pcd_dict = CenterShift(apply_z=False)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)
```

## 2）对于测试集
```python
        # 接下来是对点云进行数据增强
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)        
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)
```


# 3.第三步，有几种数据增强是之前的代码中未涉及到的，现补充(直接粘贴到cloud_processors.py的开头部分即可)

## 1）RandomDrop
```python
class RandomDropout(object):
    def __init__(self, dropout_ratio=0.2, dropout_application_ratio=0.5):
        """
            upright_axis: axis index among x,y,z, i.e. 2 for z
        """
        self.dropout_ratio = dropout_ratio
        self.dropout_application_ratio = dropout_application_ratio

    def __call__(self, data_dict):
        if random.random() < self.dropout_application_ratio:
            n = len(data_dict["coord"])
            idx = np.random.choice(n, int(n * (1 - self.dropout_ratio)), replace=False)
            if "coord" in data_dict.keys():
                data_dict["coord"] = data_dict["coord"][idx]
            if "color" in data_dict.keys():
                data_dict["color"] = data_dict["color"][idx]
            if "normal" in data_dict.keys():
                data_dict["normal"] = data_dict["normal"][idx]
            if "strength" in data_dict.keys():
                data_dict["strength"] = data_dict["strength"][idx]
            if "instance" in data_dict.keys():
                data_dict["instance"] = data_dict["instance"][idx]
            if "label" in data_dict.keys():
                data_dict["label"] = data_dict["label"][idx] \
                    if len(data_dict["label"]) != 1 else data_dict["label"]
        return data_dict
```

## 2）RandomRotate
```python
class RandomRotate(object):
    def __init__(self,
                 angle=None,
                 center=None,
                 axis='z',
                 always_apply=False,
                 p=0.5):
        self.angle = [-1, 1] if angle is None else angle
        self.axis = axis
        self.always_apply = always_apply
        self.p = p if not self.always_apply else 1
        self.center = center

    def __call__(self, data_dict):
        if random.random() > self.p:
            return data_dict
        angle = np.random.uniform(self.angle[0], self.angle[1]) * np.pi
        rot_cos, rot_sin = np.cos(angle), np.sin(angle)
        if self.axis == 'x':
            rot_t = np.array([[1, 0, 0], [0, rot_cos, -rot_sin], [0, rot_sin, rot_cos]])
        elif self.axis == 'y':
            rot_t = np.array([[rot_cos, 0, rot_sin], [0, 1, 0], [-rot_sin, 0, rot_cos]])
        elif self.axis == 'z':
            rot_t = np.array([[rot_cos, -rot_sin, 0], [rot_sin, rot_cos, 0], [0, 0, 1]])
        else:
            raise NotImplementedError
        if "coord" in data_dict.keys():
            if self.center is None:
                x_min, y_min, z_min = data_dict["coord"].min(axis=0)
                x_max, y_max, z_max = data_dict["coord"].max(axis=0)
                center = [(x_min + x_max) / 2, (y_min + y_max) / 2, (z_min + z_max) / 2]
            else:
                center = self.center
            data_dict["coord"] -= center
            data_dict["coord"] = np.dot(data_dict["coord"], np.transpose(rot_t))
            data_dict["coord"] += center
        if "normal" in data_dict.keys():
            data_dict["normal"] = np.dot(data_dict["normal"], np.transpose(rot_t))
        return data_dict

```

## 3）ElasticDistortion
```python
class ElasticDistortion(object):
    def __init__(self, distortion_params=None):
        self.distortion_params = [[0.2, 0.4], [0.8, 1.6]] if distortion_params is None else distortion_params

    @staticmethod
    def elastic_distortion(coords, granularity, magnitude):
        """
        Apply elastic distortion on sparse coordinate space.
        pointcloud: numpy array of (number of points, at least 3 spatial dims)
        granularity: size of the noise grid (in same scale[m/cm] as the voxel grid)
        magnitude: noise multiplier
        """
        blurx = np.ones((3, 1, 1, 1)).astype('float32') / 3
        blury = np.ones((1, 3, 1, 1)).astype('float32') / 3
        blurz = np.ones((1, 1, 3, 1)).astype('float32') / 3
        coords_min = coords.min(0)

        # Create Gaussian noise tensor of the size given by granularity.
        noise_dim = ((coords - coords_min).max(0) // granularity).astype(int) + 3
        noise = np.random.randn(*noise_dim, 3).astype(np.float32)

        # Smoothing.
        for _ in range(2):
            noise = scipy.ndimage.filters.convolve(noise, blurx, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blury, mode='constant', cval=0)
            noise = scipy.ndimage.filters.convolve(noise, blurz, mode='constant', cval=0)

        # Trilinear interpolate noise filters for each spatial dimensions.
        ax = [
            np.linspace(d_min, d_max, d)
            for d_min, d_max, d in zip(coords_min - granularity, coords_min + granularity *
                                       (noise_dim - 2), noise_dim)
        ]
        interp = scipy.interpolate.RegularGridInterpolator(ax, noise, bounds_error=False, fill_value=0)
        coords += interp(coords) * magnitude
        return coords

    def __call__(self, data_dict):
        if "coord" in data_dict.keys() and self.distortion_params is not None:
            if random.random() < 0.95:
                for granularity, magnitude in self.distortion_params:
                    data_dict["coord"] = self.elastic_distortion(data_dict["coord"], granularity, magnitude)
        return data_dict
```

## RandomSample函数更新
```python
class RandomSample(object):
    def __init__(self, max_num = 80000):
        self.max_num = max_num
    def __call__(self, data_dict):
        if "coord" in data_dict.keys():
            if data_dict["coord"].shape[0] > self.max_num:
                choice = np.random.choice(data_dict["coord"].shape[0], self.max_num, replace=False)
                for key in data_dict.keys():
                    data_dict[key] = data_dict[key][choice]
            else:
                #若不够这么多点，则随机的重复采样
                choice = np.random.choice(data_dict["coord"].shape[0], self.max_num - data_dict["coord"].shape[0], replace=True)
                for key in data_dict.keys():
                    data_dict[key] = np.concatenate([data_dict[key], data_dict[key][choice]], axis=0)
        return data_dict
```

# 4.以前的对于S3DIS数据集的预处理方式（暂存一下）

## 1）对于训练集
```python
        # 接下来是对点云进行数据增强
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = RandomScale(scale=[0.9, 1.1])(pcd_dict)
        pcd_dict = RandomFlip(p=0.5)(pcd_dict)
        pcd_dict = RandomJitter(sigma=0.005, clip=0.02)(pcd_dict)
        pcd_dict = ChromaticAutoContrast(p=0.2, blend_factor=None)(pcd_dict)
        pcd_dict = ChromaticTranslation(p=0.95, ratio=0.05)(pcd_dict)
        pcd_dict = ChromaticJitter(p=0.95, std=0.05)(pcd_dict)
        # pcd_dict = Voxelize(voxel_size=0.04, hash_type='fnv', mode='train',
        #             keys=("coord", "color", "label"), return_discrete_coord=True)(pcd_dict)
        # pcd_dict = SphereCrop(point_max=100000, mode='random')(pcd_dict)

        # 随机下采样是我自己实现的，而对于其他的数据变换方式是原来的PointTransformer2的代码
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)
        pcd_dict = CenterShift(apply_z=False)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)
```

## 2）对于测试集
```python
        # 接下来是对点云进行数据增强
        pcd_dict = CenterShift(apply_z=True)(pcd_dict)
        pcd_dict = Copy(keys_dict={"coord": "origin_coord", "label": "origin_label"})(pcd_dict)
        # pcd_dict = Voxelize(voxel_size=0.04, hash_type='fnv', mode='train',
        #             keys=("coord", "color", "label"), return_discrete_coord=True)(pcd_dict)
        pcd_dict = RandomSample(max_num=self.max_size)(pcd_dict)
        pcd_dict = CenterShift(apply_z=False)(pcd_dict)
        pcd_dict = NormalizeColor()(pcd_dict)
        pcd_dict = ToTensor()(pcd_dict)
        pcd_dict = Collect(keys=("coord", "color", "label"), feat_keys=["coord", "color"])(pcd_dict)
```
