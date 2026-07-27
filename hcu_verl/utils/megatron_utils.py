# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect

from megatron.core import mpu, tensor_parallel
from megatron.core.distributed import DistributedDataParallel as DDP
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.enums import ModelType
from megatron.core.transformer import TransformerConfig
from megatron.core.transformer.module import Float16Module

from verl.utils.device import get_device_id, get_device_name
from verl.utils.megatron_utils import get_model_config

def get_model(
    model_provider_func,
    model_type=ModelType.encoder_or_decoder,
    wrap_with_ddp=True,
    use_distributed_optimizer=True,
    transformer_config=None,
    override_ddp_config=None,
):
    """Build the model."""
    # Build model.
    if (
        mpu.get_pipeline_model_parallel_world_size() > 1
        and mpu.get_virtual_pipeline_model_parallel_world_size() is not None
    ):
        assert model_type != getattr(ModelType, "encoder_and_decoder", None), (
            "Interleaved schedule not supported for model with both encoder and decoder"
        )
        model = []
        has_vp_stage = inspect.signature(mpu.is_pipeline_first_stage).parameters.get("vp_stage", None) is not None
        for i in range(mpu.get_virtual_pipeline_model_parallel_world_size()):
            mpu.set_virtual_pipeline_model_parallel_rank(i)
            # Set pre_process and post_process only after virtual rank is set.
            extra_kwargs = {} if not has_vp_stage else {"ignore_virtual": False, "vp_stage": i}
            pre_process = mpu.is_pipeline_first_stage(**extra_kwargs)
            post_process = mpu.is_pipeline_last_stage(**extra_kwargs)
            this_model = model_provider_func(pre_process=pre_process, post_process=post_process, vp_stage=i)
            this_model.model_type = model_type
            model.append(this_model)
        mpu.set_virtual_pipeline_model_parallel_rank(0)
    else:
        pre_process = mpu.is_pipeline_first_stage()
        post_process = mpu.is_pipeline_last_stage()
        add_encoder = True
        add_decoder = True
        assert model_type != getattr(ModelType, "encoder_and_decoder", None), (
            "Model type encoder_and_decoder is not supported"
        )
        if model_type == getattr(ModelType, "encoder_and_decoder", None):
            if mpu.get_pipeline_model_parallel_world_size() > 1:
                assert mpu.get_pipeline_model_parallel_split_rank() is not None, (
                    "Split rank needs to be specified for model with both encoder and decoder"
                )
                rank = mpu.get_pipeline_model_parallel_rank()
                split_rank = mpu.get_pipeline_model_parallel_split_rank()
                world_size = mpu.get_pipeline_model_parallel_world_size()
                pre_process = rank == 0 or rank == split_rank
                post_process = (rank == (split_rank - 1)) or (rank == (world_size - 1))
                add_encoder = mpu.is_pipeline_stage_before_split()
                add_decoder = mpu.is_pipeline_stage_after_split()
            model = model_provider_func(
                pre_process=pre_process, post_process=post_process, add_encoder=add_encoder, add_decoder=add_decoder
            )
        else:
            model = model_provider_func(pre_process=pre_process, post_process=post_process)
        model.model_type = model_type

    if not isinstance(model, list):
        model = [model]

    # Set tensor model parallel attributes if not set.
    # Only parameters that are already tensor model parallel have these
    # attributes set for them. We should make sure the default attributes
    # are set for all params so the optimizer can use them.
    for model_module in model:
        for param in model_module.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(param)

    # Print number of parameters.
    if mpu.get_data_parallel_rank() == 0:
        print(
            " > number of parameters on (tensor, pipeline) model parallel rank ({}, {}): {}".format(
                mpu.get_tensor_model_parallel_rank(),
                mpu.get_pipeline_model_parallel_rank(),
                sum([sum([p.nelement() for p in model_module.parameters()]) for model_module in model]),
            ),
            flush=True,
        )

    # GPU allocation.
    if transformer_config is None or (not transformer_config.use_cpu_initialization):
        for model_module in model:
            model_module.to(f"{get_device_name()}:{get_device_id()}")

    # Fp16 conversion.
    config: TransformerConfig = get_model_config(model[0])
    config.fp8 = None
    tfconfig: TransformerConfig = model[0].config
    if config.fp16 or config.bf16:  # the ModelParallelConfig in GPTModel
        model = [Float16Module(config, model_module) for model_module in model]

    if wrap_with_ddp:
        ddp_models = []
        ddp_config_dict = {
            "use_distributed_optimizer": use_distributed_optimizer,
            "grad_reduce_in_fp32": True,
            "overlap_grad_reduce": False,
        }
        if override_ddp_config is not None:
            ddp_config_dict.update(override_ddp_config)
        ddp_config = DistributedDataParallelConfig(**ddp_config_dict)
        for model_chunk_idx, model_chunk in enumerate(model):
            ddp_model = DDP(
                config=tfconfig,
                module=model_chunk,
                disable_bucketing=(model_chunk_idx > 0),
                ddp_config=ddp_config,
            )
            ddp_models.append(ddp_model)
        model = ddp_models
        # # Broadcast params from data parallel src rank to other data parallel ranks.
        # # if args.data_parallel_random_init:
        for model_module in model:
            model_module.broadcast_params()
    return model