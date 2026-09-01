<h1 align="center">
  <strong>
  verl-das is a patched RL training library based on 
  <a href="https://github.com/volcengine/verl" target="_blank"><strong>verl</strong></a> 
  and 
  <a href="https://github.com/NVIDIA/Megatron-LM" target="_blank"><strong>Megatron-LM</strong></a>.
  <strong>
  <br>
</h1>

## 简介
verl-das：基于HCU生态的强化学习训练套件，旨在为HCU生态合作伙伴提供强化学习训练解决方案。

## 快速开始
在HCU设备上快速上手使用verl-das详见[用户指南](docs/user_guide.md)。

## 支持特性
|Training Algorithm    | Model family          | `vllm` | `sglang` | Train backend   |
|:-------------------: | :-------------------: | :----: | :------: | :-------------: |
|grpo                  | Qwen2.5-0.5B          | ✓      |          | FSDP            |
|grpo                  | Qwen2.5-VL-7B         | ✓      |          | FSDP2, Megatron |
|grpo                  | Qwen3-8B              | ✓      |          | FSDP, Megatron  |
|grpo                  | Qwen3-235B            | ✓      |          | Megatron        |
|grpo                  | Qwen3-VL-8B           | ✓      |          | FSDP2, Megatron |
|grpo                  | Qwen3.5-9B            | ✓      |          | Megatron        |
|grpo                  | Qwen3.5-27B           | ✓      |          | FSDP2, Megatron |
|grpo                  | Qwen3.5-35B-A3B       | ✓      |          | FSDP2, Megatron |
|grpo                  | DeepSeek-7B           | ✓      |          | FSDP            |
|grpo                  | Moonlight-16B-A3B     | ✓      |          | Megatron        |
|grpo                  | Seed-OSS-36B-Base     | ✓      |          | FSDP2           |
|fully_async_policy    | Qwen2.5-0.5B          | ✓      |          | FSDP2           |
|fully_async_policy    | Qwen2.5-7B            | ✓      |          | FSDP2           |
|one_step_off_policy   | Qwen3-0.6B            | ✓      | ✓        | FSDP2           |
|on_policy_distillation| Qwen3-8B              | ✓      |          | FSDP, Megatron  |
|on_policy_distillation| Qwen3-VL-8B           | ✓      |          | FSDP            |
|on_policy_distillation| Qwen3.5-4B            | ✓      |          | FSDP2           |
|multi_on_policy_distillation| Qwen3-VL-8B     | ✓      |          | FSDP, VeOmni    |

## License

HCU adaptations, modifications, and original contributions by Hygon Information Technology Co., Ltd. are licensed under the Apache License, Version 2.0.

Modified by Hygon Information Technology Co., Ltd.

Original copyright notices and license terms from the upstream verl project are retained. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.