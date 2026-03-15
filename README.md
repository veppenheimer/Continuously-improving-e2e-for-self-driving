------------2026 3.15改动------------



更改模型权重保存策略，不再选择loss最小的pth，添加validation loop
添加early stopping 机制，防止过拟合，添加可视化Learning Curve，添加推理FPS统计
更改数据集划分策略，添加验证集。
对数据采用balanced sampler，不再使用原始数据的全部或者直接下采样硬删除部分直行数据，该部分在datasets中实现，而不在划分数据集时下采样
