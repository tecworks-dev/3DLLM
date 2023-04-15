目前的代码进行了以下修改：  
1. `lavis/models/blip2_models` 是使用三维点云作为输入的代码，原本的 blip2 代码被放到了同级目录的 `blip2_models_reserve` 中，理论上老代码应该不会运行



TODO:
1. 数据集构造
builder = registry.get_builder_class(name)(dataset_config)  
dataset = builder.build_datasets()
注册processor用于预处理数据，包括cloud_processor 和中文的 text_processor
建立一个假数据测试
2. Point Transformer 构建 
3. llama 模型