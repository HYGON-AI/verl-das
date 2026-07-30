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

from megatron.training import print_rank_0

class VerlAdaptation:

    _patch_info_collection = {}
    _args = None

    @classmethod
    def execute(cls):
        """
        Execute adaptations.
        """
        CoreAdaptation().execute()
        VerlAdaptation.apply()
    
    @classmethod
    def register(cls, orig_func_name, new_func=None, force_patch=False, create_dummy=False, apply_wrapper=False, remove_origin_wrappers=False):
        """
        Register adaptations into collection.
        """
        if orig_func_name not in cls._patch_info_collection:
            from .patch_utils import Patch
            cls._patch_info_collection[orig_func_name] = Patch(
                orig_func_name,
                new_func,
                create_dummy,
                apply_wrapper=apply_wrapper,
                remove_origin_wrappers=remove_origin_wrappers
            )
        else:
            cls._patch_info_collection.get(orig_func_name).set_patch_func(
                new_func,
                force_patch,
                apply_wrapper=apply_wrapper,
                remove_origin_wrappers=remove_origin_wrappers
            )

    @classmethod
    def apply(cls):
        """
        Apply adaptations.
        """
        for patch in cls._patch_info_collection.values():
            patch.apply_patch()

    @classmethod
    def post_execute(cls):
        """
        Execute after other adaptations.
        """
        pass


class CoreAdaptation:
    def execute(self):
        self.verl_hcu_wrapper()
        self.megatron_hcu_wrapper()
    
    def verl_hcu_wrapper(self):
        # verl.trainer
        from ..trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
        VerlAdaptation.register('verl.trainer.constants_ppo.PPO_RAY_RUNTIME_ENV',
                                PPO_RAY_RUNTIME_ENV)

        # verl.utils
        from ..utils.flops_counter import _DEVICE_FLOPS
        VerlAdaptation.register('verl.utils.flops_counter._DEVICE_FLOPS',
                                _DEVICE_FLOPS)

        # TODO: This patch will be removed in next verl version
        from ..utils.megatron_utils import get_model
        VerlAdaptation.register('verl.utils.megatron_utils.get_model',
                                get_model)

        # TODO: This patch will be removed in vllm >= v0.19.1
        # verl.workers.rollout.vllm_rollout
        from ..workers.rollout.vllm_rollout.utils import get_device_uuid
        VerlAdaptation.register('verl.workers.rollout.vllm_rollout.utils.get_device_uuid',
                                get_device_uuid)

    def megatron_hcu_wrapper(self):
        # megatron core
        from ..core.dist_checkpointing.strategies.filesystem_async import preload_tensors
        VerlAdaptation.register('megatron.core.dist_checkpointing.strategies.filesystem_async.FileSystemWriterAsync.preload_tensors',
                                preload_tensors)


VerlAdaptation.execute()
print_rank_0("[HCU_ADAPT] Patch has been applied in worker")
