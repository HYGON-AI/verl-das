# 基于verl-das启动GRPO训练示例

## 1、环境准备
### 1.1 [光源社区](https://developer.sourcefind.cn/servicelist/detail?post_id=61036870-b3c7-11f0-9989-acde48001122&active=Overview)拉取HCU基础docker镜像

### 1.2 创建容器

参考命令如下（请按需修改，正确挂载工作目录，修改正确镜像名称）：
```shell
docker run -it \
       --name verl-das_test \
       --shm-size=64G \
       --device=/dev/kfd \
       --device=/dev/mkfd \
       --device=/dev/dri \
       --cap-add=SYS_PTRACE \
       --security-opt seccomp=unconfined \
       --ulimit memlock=-1:-1 \
       --ipc=host \
       --network=host \
       --workdir=/workspace \
       --privileged \
       -v /opt/hyhal:/opt/hyhal:ro \
       -v /home:/home \
       REPOSITORY:TAG \
       /bin/bash
```

### 1.3 拉取代码仓
```shell
cd /workspace

git clone --recurse-submodules https://github.com/HYGON-AI/verl-das.git
```
如代码仓无法正常clone，可离线下载对应分支的`verl-das`，`Megatron-LM`, `verl`仓库，解压后按`verl-das`仓库目录位置放置。

### 1.4 安装依赖
```shell
cd /workspace/verl-das
pip install -r requirements.txt
```

## 2、数据处理
### 2.1 数据集下载

以gsm8k数据集为例，[点击下载](https://huggingface.co/datasets/openai/gsm8k/tree/main)数据集。

假设下载后文件目录如下
```
workspace
|—— data
|   |—— gsm8k
|   |   |—— main
|   |   |   |—— test-00000-of-00001.parquet
|   |   |   |—— train-00000-of-00001.parquet
```

### 2.2 数据集处理

执行如下命令
```shell
cd /workspace/verl-das/verl

python examples/data_preprocess/gsm8k.py --local_dataset_path /workspace/data/gsm8k --local_save_dir /workspace/data/gsm8k
```


## 3、模型准备

以Qwen2.5-0.5B模型为例

```shell
pip install modelscope

cd /workspace

modelscope download --model Qwen/Qwen2.5-0.5B-Instruct --local_dir ./Qwen2.5-0.5B-Instruct
```

或者[点击链接](https://modelscope.cn/models/Qwen/Qwen2.5-0.5B-Instruct/files)，手动下载模型文件，上传至`/workspace`路径下。


## 4、启动任务

以Qwen2.5-0.5B-Instruct模型GRPO任务为例

### 4.1 参数填写

（1）配置hostfile文件

进入`/workspace/verl-das/examples/grpo_trainer`，填写`hostfile`文件，示例如下（多个节点填写多行）

```shell
127.0.0.1 slots=1

```

可以填写节点ip或者节点主机名。

（2）根据任务填写`run.sh`脚本

进入`/workspace/verl-das/examples/grpo_trainer`，根据上述文件路径，填写`run.sh`文件中如下变量：

```shell
export NET_TYPE="" # please choose one of {mlnx, shca}.
PORT="" # The port which you set in your docker using /usr/sbin/sshd -p xxx
HOSTFILE="./hostfile"
DATA_PATH="/workspace/data"
HF_MODEL_PATH="/workspace"
MCORE_MODEL_PATH="/workspace"
PROFILING="" # If you want to profiling, please choose one of {torch}
```

### 4.2 执行脚本

```shell
docker exec -it "your container name" bash
cd /workspace/verl-das/examples/grpo_trainer

bash run.sh --model_name qwen2_5_0.5b
```
