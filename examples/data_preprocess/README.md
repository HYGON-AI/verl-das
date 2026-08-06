# Dataset Preprocessing
```bash
CURRENT_DIR=$( cd "$( dirname "$0" )" && pwd )
VERL_PATH=$( dirname $( dirname ${CURRENT_DIR}))
export GLOG_minloglevel=3
export PYTHONPATH=${VERL_PATH}:${VERL_PATH}/third_party/verl:${VERL_PATH}/third_party/Megatron-LM:$PYTHONPATH
```

## BytedTsinghua-SIA/AIME-2024
```python
python ${VERL_PATH}/third_party/verl/examples/data_preprocess/aime2024_multiturn_w_tool.py \
        --local_dataset_path /path/to/BytedTsinghua-SIA/AIME-2024 \
        --local_save_dir /path/to/aime-2024
```

## BytedTsinghua-SIA/DAPO-Math-17k
```python
python ${VERL_PATH}/third_party/verl/examples/data_preprocess/dapo_multiturn_w_tool.py \
        --local_dataset_path /path/to/BytedTsinghua-SIA/DAPO-Math-17k \
        --local_save_dir /path/to/dapo-math-17k
```

## DigitalLearningGmbH/MATH-lighteval
```python
python ${VERL_PATH}/third_party/verl/examples/data_preprocess/math_dataset.py \
        --local_dataset_path /path/to/DigitalLearningGmbH/MATH-lighteval \
        --local_save_dir /path/to/math
```

## hiyouga/geometry3k
```python
python ${VERL_PATH}/third_party/verl/examples/data_preprocess/geo3k.py \
        --local_dataset_path /path/to/hiyouga/geometry3k \
        --local_save_dir /path/to/geo3k
```

## math-ai/aime24、math-ai/aime25
```python
python3 ${VERL_PATH}/third_party/verl/recipe/open_math_reasoning/prepare_eval_dataset.py \
        --local_dataset_path /path/to/dataset \
        --local_save_dir /path/to/aime_test
```

## nvidia/OpenMathReasoning
```python
python3 ${VERL_PATH}/third_party/verl/recipe/open_math_reasoning/prepare_nvidia-OpenMathReasoning_sft.py \
        --local_dataset_path /path/to/nvidia/OpenMathReasoning \
        --local_save_dir /path/to/open_math_reasoning
```

# openai/gsm8k
```python
python ${VERL_PATH}/third_party/verl/examples/data_preprocess/gsm8k.py \
        --local_dataset_path /path/to/openai/gsm8k \
        --local_save_dir /path/to/gsm8k
```
