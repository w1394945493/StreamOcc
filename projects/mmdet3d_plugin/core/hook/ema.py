# Copyright (c) OpenMMLab. All rights reserved.
# modified from megvii-bevdepth.
import math
import os
import io
from copy import deepcopy

import torch
from mmcv.runner import load_state_dict
from mmcv.runner.dist_utils import master_only
from mmcv.runner.hooks import HOOKS, Hook
from mmcv.runner.hooks import CheckpointHook

from projects.mmdet3d_plugin.core.hook.utils import is_parallel

__all__ = ['ModelEMA']


class ModelEMA:
    """Model Exponential Moving Average from https://github.com/rwightman/
    pytorch-image-models Keep a moving average of everything in the model
    state_dict (parameters and buffers).

    This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/
    ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training
    schemes to perform well.
    This class is sensitive where it is initialized in the sequence
    of model init, GPU assignment and distributed training wrappers.
    """

    def __init__(self, model, decay=0.9999, updates=0):
        """
        Args:
            model (nn.Module): model to apply EMA.
            decay (float): ema decay reate.
            updates (int): counter of EMA updates.
        """
        # Create EMA(FP32)
        self.ema_model = deepcopy(model).eval()
        self.ema = self.ema_model.module.module if is_parallel(
            self.ema_model.module) else self.ema_model.module
        self.updates = updates
        # decay exponential ramp (to help early epochs)
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for p in self.ema.parameters():
            p.requires_grad_(False)

    def update(self, trainer, model):
        # Update EMA parameters
        with torch.no_grad():
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.module.state_dict() if is_parallel(
                model) else model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1.0 - d) * msd[k].detach()


@HOOKS.register_module()
class MEGVIIEMAHook(Hook):
    """EMAHook used in BEVDepth.

    Modified from https://github.com/Megvii-Base
    Detection/BEVDepth/blob/main/callbacks/ema.py.
    """

    def __init__(self, init_updates=0, decay=0.9990, split_iter=10548, final_decay=[0.9950], decay_step = None,resume=None,synch_main=False,init_update_num=10560):
        super().__init__()
        self.init_updates = init_updates
        self.resume = resume
        self.decay = decay
        # Keep accepting split_iter for old configs; checkpoint_config now owns
        # the saving schedule for both the training weights and EMA weights.
        self.final_decay = final_decay
        self.decay_step = decay_step
        self.synch_main = synch_main
        self.init_update_num = init_update_num
        
    def before_run(self, runner):
        from torch.nn.modules.batchnorm import SyncBatchNorm

        self.checkpoint_hook = next(
            (hook for hook in runner.hooks if isinstance(hook, CheckpointHook)),
            None,
        )
        bn_model_list = list()
        bn_model_dist_group_list = list()
        for model_ref in runner.model.modules():
            if isinstance(model_ref, SyncBatchNorm):
                bn_model_list.append(model_ref)
                bn_model_dist_group_list.append(model_ref.process_group)
                model_ref.process_group = None
        runner.ema_model = ModelEMA(runner.model, self.decay)

        for bn_model, dist_group in zip(bn_model_list,
                                        bn_model_dist_group_list):
            bn_model.process_group = dist_group
        runner.ema_model.updates = self.init_updates

        if self.resume is not None:
            runner.logger.info(f'resume ema checkpoint from {self.resume}')
            cpt = torch.load(self.resume, map_location='cpu')
            load_state_dict(runner.ema_model.ema, cpt['state_dict'])
            runner.ema_model.updates = cpt['updates']
            
            if self.synch_main:
                runner.ema_model.updates = self.init_update_num
                main_model_state_dict = runner.ema_model.ema.state_dict()
                runner.model.load_state_dict(main_model_state_dict, strict=True)
                runner.logger.info('Main model weights updated from EMA checkpoint.')


    def after_train_iter(self, runner):
        runner.ema_model.update(runner, runner.model.module)
        hook = self.checkpoint_hook
        # Match CheckpointHook's completed-step convention and final-step rule.
        # Update EMA first so both files describe the same completed iteration.
        if hook is not None and not hook.by_epoch:
            if hook.every_n_iters(runner, hook.interval) or (
                hook.save_last and hook.is_last_iter(runner)
            ):
                self.save_checkpoint(runner)
        if self.decay_step is not None:
            if runner.iter in self.decay_step:
                self.decay = self.final_decay.pop(0)
                runner.logger.info(f'EMA decay rate changed to {self.decay}')
            
    def after_train_epoch(self, runner):
        hook = self.checkpoint_hook
        if hook is not None and hook.by_epoch:
            if hook.every_n_epochs(runner, hook.interval) or (
                hook.save_last and hook.is_last_epoch(runner)
            ):
                self.save_checkpoint(runner)

    @master_only
    def save_checkpoint(self, runner):
        hook = self.checkpoint_hook
        # Reuse the ordinary checkpoint destination, naming template and retention
        # policy. The suffix keeps EMA files separate from resumable checkpoints.
        template = hook.args.get(
            'filename_tmpl', 'epoch_{}.pth' if hook.by_epoch else 'iter_{}.pth'
        )
        stem, extension = os.path.splitext(template)
        template = stem + '_ema' + extension
        current = runner.epoch + 1 if hook.by_epoch else runner.iter + 1
        state_dict = runner.ema_model.ema.state_dict()
        ema_checkpoint = {
            # Epoch runners have already advanced iter before after_train_epoch.
            'iter': runner.iter if hook.by_epoch else runner.iter + 1,
            'state_dict': state_dict,
            'updates': runner.ema_model.updates
        }
        save_path = hook.file_client.join_path(hook.out_dir, template.format(current))
        with io.BytesIO() as buffer:
            torch.save(ema_checkpoint, buffer)
            hook.file_client.put(buffer.getvalue(), save_path)
        # Use the same interval-based cleanup as MMCV, touching only EMA files.
        if hook.max_keep_ckpts > 0 and hook.interval > 0:
            for step in range(current - hook.max_keep_ckpts * hook.interval,
                              0, -hook.interval):
                old_path = hook.file_client.join_path(hook.out_dir, template.format(step))
                if not hook.file_client.isfile(old_path):
                    break
                hook.file_client.remove(old_path)
        runner.logger.info(f'Saving ema checkpoint at {save_path}')
