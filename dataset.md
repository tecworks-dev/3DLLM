# 数据集合说明

数据集路径：`/public/public_data/3DLLM/`

## S3dis

point cloud: `/public/public_data/3DLLM/S3DIS/`

text: `/public/public_data/3DLLM/s3dis_text/`

## scannet

point cloud: `/public/public_data/3DLLM/scannet_frames_25k/`

text: `/public/public_data/3DLLM/scannet_text/`

## struture3d

point cloud:

text:

## 已完成

1.structured3d 全景图片转换成ply点云

2.构造layout数据集，示例：

There are 10 rooms, incluing 1 living room, 1 kitchen, 3 bedrooms, 2 bathrooms, 1 balcony, 1 undefined room, and there are 8 doors,and there are 10 windows,bedroom and living room have contections.living room has contections to the outside.living room and bedroom have contections.bedroom and living room have contections.living room and bathroom have contections.balcony and living room have contections.living room and kitchen have contections.bedroom and bathroom have contections.living room and balcony have contections.living room has 1 window, kitchen has 2 windows, bedroom has 3 windows, bathroom has 2 windows, balcony has 1 window, undefined has 1 window, 


3.在S3dis数据集添加房间类型描述：

A room with a table, chairs, and a clock on the wall. a room filled with desks and chairs with books on them. a classroom with a desk with a computer and chairs. a row of chairs sitting in front of a window. a white wall with a picture of a man on it. **this is a conferenceRoom**.

4.完成英文数据集至中文数据集的翻译

## 未完成
