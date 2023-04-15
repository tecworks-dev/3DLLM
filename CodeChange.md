目前的代码进行了以下修改：  
1. `lavis/models/blip2_models` 是使用三维点云作为输入的代码，原本的 blip2 代码被放到了同级目录的 `blip2_models_reserve` 中，理论上老代码应该不会运行
2. 