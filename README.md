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
