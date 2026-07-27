# verl-das 仓库SFT微调使用说明
## 环境配置
需搭配torch2.7.1及以上版本的torch及配套使用，若按[用户指南](../../docs/user_guide.md)推荐的镜像，需在[光合社区](https://developer.sourcefind.cn/)下载torch2.7.1及其配套组件。

## 数据处理
sft数据处理可参考[用户指南](../../docs/user_guide.md)中数据处理部分。

## 运行任务
执行sft训练，在配置脚本中的参数后，执行以下命令
```shell
bash run_qwen3_8b_sft.sh
```

## 注意事项
1. deepseek3原始权重为fp8格式，需先使用官方提供脚本将权重转换为bf16格式；
2. 权重转换完成后，需将模型文件夹中config.json文件内以下量化相关参数删除：
```shell
  "quantization_config": {
    "activation_scheme": "dynamic",
    "fmt": "e4m3",
    "quant_method": "fp8",
    "weight_block_size": [
      128,
      128
    ]
  },
```

3. 提供的deepseek3 sft示例为单机减层demo，需将模型文件夹中config.json文件内模型层数改为3层：
```shell
    "num_hidden_layers": 3,
```