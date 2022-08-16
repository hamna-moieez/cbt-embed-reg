import torch
from embeddingreg import embedding


class DefaultConfig(object):
    LR = 0.001
    L1_REG = 0
    IS_INCREMENTAL = True

    ITERS = 1
    EPOCHS = 2
    BATCH_SIZE = 64
    IS_CONVOLUTIONAL = True

    NEXT_TASK_LR = None
    NEXT_TASK_EPOCHS = None

    # EWC_SAMPLE_SIZE = 250
    # EWC_IMPORTANCE = 1000
    USE_CL = True

    CL_TEC = embedding
    CL_PAR = {'sample_size': 250, 'penalty_importance': 1e+3}

    USE_TENSORBOARD = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    MODEL_NAME = ''
    SAVE_PATH = '.'

    RUN_NAME = 'default'
    LOSS = 'cross_entropy'
    OPTIMIZER = 'SGD'

    def __str__(self):
        fields = [a for a in dir(self) if not a.startswith('__')]
        s = 'CONFIG PARAMETERS\n'
        for f in fields:
            s += f+': '+str(getattr(self, f))+'\n'
        return s

class Embedding(DefaultConfig):
    CL_TEC = embedding
    # CL_PAR = {'margin': 0.5}
    # super.CL_TEC_PARAMETERS['margin'] = 0.5