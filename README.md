# 3DLLM

## 数据集路径和账号

### 山东服务器

```txt
ip:172.16.1.131
port:22
```

#### Point Transformer和Q Former环境

tdt(环境名称pointcept)、 rmq(环境名称pointformer)

#### 数据集

```txt
/data3/rmq/points_text_datasets (有S3DIS、ScanNet)
```

#### 代码

```txt
/home/rmq/PointTextCombine
```

### 自动化所集群

好像不能debug

```txt
ip:172.18.36.7
port:22
```

#### 账号

```txt
账号1:mengqi_rong8
密码:123456

账号2:tang_xc_11
密码:1111
```

#### 环境设置

> 配置环境时需要在admin下面，即运行salloc.sh之前。**运行程序前许先执行sh salloc.sh 然后执行ssh gpu8**
>
> salloc.sh 里的内容salloc -N 1 --cpus-per-task=1 --gres=gpu:8 -p td_ssh --nodelist gpu8

#### 数据存储

```txt
/public/public_data/3DLLM
```

# 运行代码

## 第一阶段训练
```[bash]
cd 3DLLM
python -m torch.distributed.run --nproc_per_node=4 train_3d.py --cfg-path lavis/projects/blip2_3d/train/pretrain_stage1.yaml
```
配置文件 `pretrain_stage1.yaml` 中定义了一些基础的参数，包括学习率，使用的数据集等等。在不同的电脑上跑，需要进行一些不同的修改。
1. 进入 `lavis/configs/models/blip2_3d/blip2_3d_stage1.yaml`，这是第一阶段的模型的配置文件。这里有一项为 `point_cloud_encoder_model_path` 表示的是预训练的三维特征提取网络的存储路径，需要相应修改。
2. 进入 `lavis/configs/datasets/point_cloud/defaults_cap.yaml`，这是数据集的配置文件。里面的 `train.storage` 是数据集的具体内容，需要进行相应的修改。
3. 如果想要在第一阶段就使用验证集、测试集，需要先在基础配置文件 `pretrain_stage1.yaml` 中启用这两个集合
   ```
    # test_splits: ["test"]   ->  test_splits: ["test"]
    # valid_splits: ["val"]   ->  valid_splits: ["val"]
   ```
   然后进入数据集的配置文件 `defaults_cap.yaml` 中，把 train, val 相应的数据存储路径输入进去