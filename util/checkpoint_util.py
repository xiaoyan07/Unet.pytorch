import torch
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

GLOBAL_STEP = 'global_step'
OPTIMIZER = 'optimizer'
NET = 'net'


class CheckPoint(object):
    def __init__(self,
                 filepath):
        self._fpath = filepath
        self._cache = None
        self._train_state = TrainState()

    def read(self):
        if self._cache is None:
            self._cache = torch.load(self._fpath)
            self._train_state = TrainState.from_state_dict(self._cache)
        return self._cache

    def write(self, state_dict):
        return torch.save(state_dict, self._fpath)

    def get_global_step(self):
        if self._cache is None:
            self.read()
        if GLOBAL_STEP in self._cache:
            return self._cache[GLOBAL_STEP]
        else:
            logger.warning('{} is not existed in {}. default to 0'.format(GLOBAL_STEP, self._fpath))
            return 0

    def restore_train_state(self, model=None, optimizer=None, global_step=None):
        if model:
            model.load_state_dict(self._train_state.get_model_state_dict())
        if optimizer:
            optimizer.load_state_dict(self._train_state.get_optimizer_state_dict())
        if global_step:
            global_step.load_state_dict(self._train_state.get_global_step_state_dict())


class GlobalStep(object):
    def __init__(self,
                 init_step=0):
        self._global_step = init_step

    def state_dict(self):
        return {
            GLOBAL_STEP: self._global_step
        }

    def load_state_dict(self, state_dict):
        gs = state_dict[GLOBAL_STEP]
        self._global_step = gs

    def step(self):
        self._global_step += 1

    @property
    def value(self):
        return self._global_step


class TrainState(object):
    def __init__(self):
        self._default_state_dict = {
            GLOBAL_STEP: 0,
            NET: {},
            OPTIMIZER: {}
        }

    def state_dict(self):
        return self._default_state_dict

    def update(self, state_dict):
        for k, v in state_dict.items():
            self._default_state_dict[k] = v

    def update_model(self, model):
        self._default_state_dict.update({
            NET: model.state_dict(),
        })

    def update_optimizer(self, optimizer):
        self._default_state_dict.update({
            OPTIMIZER: optimizer.state_dict()
        })

    def update_globalstep(self, global_step):
        if isinstance(global_step, int):
            gstep = global_step
        elif isinstance(global_step, GlobalStep):
            gstep = global_step.value
        else:
            raise ValueError()
        self._default_state_dict.update({
            GLOBAL_STEP: gstep,
        })

    def get_model_state_dict(self):
        return self._default_state_dict[NET]

    def get_optimizer_state_dict(self):
        return self._default_state_dict[OPTIMIZER]

    def get_global_step_state_dict(self):
        return self._default_state_dict[GLOBAL_STEP]

    @staticmethod
    def from_state_dict(state_dict):
        trainstate = TrainState()

        for k, _ in trainstate.state_dict():
            if k in state_dict:
                trainstate._default_state_dict[k] = state_dict[k]
        return trainstate

    @staticmethod
    def from_instances(model=None, optimizer=None, global_step=None):
        trainstate = TrainState()
        if model:
            trainstate.update_model(model)
        if optimizer:
            trainstate.update_optimizer(optimizer)
        if global_step:
            trainstate.update_globalstep(global_step)
        return trainstate
