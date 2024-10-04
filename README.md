#3DLLM

## Dataset path and account

### Shandong Server

```txt
ip:172.16.1.131
port:22
```

#### Point Transformer and Q Former Environment

tdt(environment name pointcept), rmq(environment name pointformer)

#### Dataset

```txt
/data3/rmq/points_text_datasets (with S3DIS, ScanNet)
```

#### Code

```txt
/home/rmq/PointTextCombine
```

### Automation Institute Cluster

Can't seem to debug

```txt
ip:172.18.36.75
port:22
```

#### account

```txt
Account 1:mengqi_rong8
Password:123456

Account 2:tang_xc_11
Password:1111
```

#### Environment Setup

> You need to be in admin when configuring the environment, that is, before running salloc.sh. **Before running the program, you can first execute sh salloc.sh and then execute ssh gpu8**
>
> Contents of salloc.sh salloc -N 1 --cpus-per-task=1 --gres=gpu:8 -p td_ssh --nodelist gpu8

#### Data Storage

[dataset.md](https://github.com/rongmq8802/3DLLM/blob/main/dataset.md)

#### Project Modification

[codechange.md](https://github.com/rongmq8802/3DLLM/blob/main/CodeChange.md)


#### Code Flow

[Code flow.md](https://github.com/rongmq8802/3DLLM/blob/main/%E4%BB%A3%E7%A0%81%E6%B5%81%E7%A8%8B.md)

## Tool Code

### :cat: llama original weight converted to huggin face format

#### Create the environment

```bash
pip install transformers
pip install accelerate
pip install protobuf==3.20.0
```

#### use

```bash
python transform.py --input_dir "llama_weight_dir" --model_size "X"B --output_dir "output_dir"
```

# Run the code

## First stage training
```[bash]
cd 3DLLM
python -m torch.distributed.run --nproc_per_node=4 train_3d.py --cfg-path lavis/projects/blip2_3d/train/pretrain_stage1.yaml
```
The configuration file `pretrain_stage1.yaml` defines some basic parameters, including learning rate, dataset used, etc. Running on different computers requires some different modifications.
1. Enter `lavis/configs/models/blip2_3d/blip2_3d_stage1.yaml`, which is the configuration file of the first stage model. There is an item called `point_cloud_encoder_model_path`, which represents the storage path of the pre-trained 3D feature extraction network and needs to be modified accordingly.
2. Enter `lavis/configs/datasets/point_cloud/defaults_cap.yaml`, which is the configuration file of the dataset. The `train.storage` in it is the specific content of the dataset and needs to be modified accordingly.
3. If you want to use the validation set and test set in the first stage, you need to enable these two sets in the basic configuration file `pretrain_stage1.yaml`
   ```
    # test_splits: ["test"] -> test_splits: ["test"]
    # valid_splits: ["val"] -> valid_splits: ["val"]
   ```
   Then enter the dataset configuration file `defaults_cap.yaml` and enter the corresponding data storage path of train and val
