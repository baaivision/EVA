# ------------------------------------------------------------------------------------------------
# Modified from:
# https://github.com/facebookresearch/d2go/blob/main/d2go/modeling/ema.py
# https://github.com/IDEA-Research/detrex/blob/main/detrex/modeling/ema.py
# ------------------------------------------------------------------------------------------------


import copy
import itertools
import logging
from contextlib import contextmanager
from typing import List

import torch
from detectron2.engine.train_loop import HookBase


logger = logging.getLogger(__name__)


class EMAState(object):
    def __init__(self):
        self.state = {}
    
    @classmethod
    def FromModel(cls, model: torch.nn.Module, device: str = ""):
        ret = cls()
        ret.save_from(model, device)
        return ret
    
    def save_from(self, model: torch.nn.Module, device: str = ""):
        """Save model state from `model` to this object"""
        for name, val in self.get_model_state_iterator(model):
            val = val.detach().clone()
            self.state[name] = val.to(device) if device else val
    
    def apply_to(self, model: torch.nn.Module):
        """Apply state to `model` from this object"""
        with torch.no_grad():
            for name, val in self.get_model_state_iterator(model):
                assert (
                    name in self.state
                ), f"Name {name} not existed, available names {self.state.keys()}"
                val.copy_(self.state[name])

    @contextmanager
    def apply_and_restore(self, model):
        old_state = EMAState.FromModel(model, self.device)
        self.apply_to(model)
        yield old_state
        old_state.apply_to(model)
    
    def get_ema_model(self, model):
        ret = copy.deepcopy(model)
        self.apply_to(ret)
        return ret
    
    @property
    def device(self):
        if not self.has_inited():
            return None
        return next(iter(self.state.values())).device
    
    def to(self, device):
        for name in self.state:
            self.state[name] = self.state[name].to(device)
        return self
    
    def has_inited(self):
        return self.state

    def clear(self):
        self.state.clear()
        return self

    def get_model_state_iterator(self, model):
        param_iter = model.named_parameters()
        buffer_iter = model.named_buffers()
        return itertools.chain(param_iter, buffer_iter)

    def state_dict(self):
        return self.state

    def load_state_dict(self, state_dict, strict: bool = True):
        self.clear()
        for x, y in state_dict.items():
            self.state[x] = y
        return torch.nn.modules.module._IncompatibleKeys(
            missing_keys=[], unexpected_keys=[]
        )

    def __repr__(self):
        ret = f"EMAState(state=[{','.join(self.state.keys())}])"
        return ret


class EMAUpdater(object):
    """Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and
    buffers). This is intended to allow functionality like
    https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    Note:  It's very important to set EMA for ALL network parameters (instead of
    parameters that require gradient), including batch-norm moving average mean
    and variance.  This leads to significant improvement in accuracy.
    For example, for EfficientNetB3, with default setting (no mixup, lr exponential
    decay) without bn_sync, the EMA accuracy with EMA on params that requires
    gradient is 79.87%, while the corresponding accuracy with EMA on all params
    is 80.61%.
    Also, bn sync should be switched on for EMA.
    """

    def __init__(self, state: EMAState, decay: float = 0.999, device: str = ""):
        self.decay = decay
        self.device = device

        self.state = state

    def init_state(self, model):
        self.state.clear()
        self.state.save_from(model, self.device)

    def update(self, model):
        with torch.no_grad():
            ema_param_list = []
            param_list = []
            for name, val in self.state.get_model_state_iterator(model):
                ema_val = self.state.state[name]
                if self.device:
                    val = val.to(self.device)
                if val.dtype in [torch.float32, torch.float16]:
                    ema_param_list.append(ema_val)
                    param_list.append(val)
                else:
                    ema_val.copy_(ema_val * self.decay + val * (1.0 - self.decay))
            self._ema_avg(ema_param_list, param_list, self.decay)

    def _ema_avg(
        self,
        averaged_model_parameters: List[torch.Tensor],
        model_parameters: List[torch.Tensor],
        decay: float,
    ) -> None:
        """
        Function to perform exponential moving average:
        x_avg = alpha * x_avg + (1-alpha)* x_t
        """
        torch._foreach_mul_(averaged_model_parameters, decay)
        torch._foreach_add_(
            averaged_model_parameters, model_parameters, alpha=1 - decay
        )


def _remove_ddp(model):
    from torch.nn.parallel import DistributedDataParallel

    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def may_build_model_ema(cfg, model):
    if not cfg.train.model_ema.enabled:
        return
    model = _remove_ddp(model)
    assert not hasattr(
        model, "ema_state"
    ), "Name `ema_state` is reserved for model ema."
    model.ema_state = EMAState()
    logger.info("Using Model EMA.")


def may_get_ema_checkpointer(cfg, model):
    if not cfg.train.model_ema.enabled:
        return {}
    model = _remove_ddp(model)
    return {"ema_state": model.ema_state}


def get_model_ema_state(model):
    """Return the ema state stored in `model`"""
    model = _remove_ddp(model)
    assert hasattr(model, "ema_state")
    ema = model.ema_state
    return ema


def apply_model_ema(model, state=None, save_current=False):
    """Apply ema stored in `model` to model and returns a function to restore
    the weights are applied
    """
    model = _remove_ddp(model)

    if state is None:
        state = get_model_ema_state(model)

    if save_current:
        # save current model state
        old_state = EMAState.FromModel(model, state.device)
    state.apply_to(model)

    if save_current:
        return old_state
    return None


@contextmanager
def apply_model_ema_and_restore(model, state=None):
    """Apply ema stored in `model` to model and returns a function to restore
    the weights are applied
    """
    model = _remove_ddp(model)

    if state is None:
        state = get_model_ema_state(model)

    old_state = EMAState.FromModel(model, state.device)
    state.apply_to(model)
    yield old_state
    old_state.apply_to(model)


class EMAHook(HookBase):
    def __init__(self, cfg, model):
        model = _remove_ddp(model)
        assert cfg.train.model_ema.enabled
        assert hasattr(
            model, "ema_state"
        ), "Call `may_build_model_ema` first to initilaize the model ema"
        self.model = model
        self.ema = self.model.ema_state
        self.device = cfg.train.model_ema.device or cfg.model.device
        self.ema_updater = EMAUpdater(
            self.model.ema_state, decay=cfg.train.model_ema.decay, device=self.device
        )

    def before_train(self):
        if self.ema.has_inited():
            self.ema.to(self.device)
        else:
            self.ema_updater.init_state(self.model)

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        if not self.model.train:
            return
        self.ema_updater.update(self.model)