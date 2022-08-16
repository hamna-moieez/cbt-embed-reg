import torch

config = {

    'task': {

        'num_tasks': 3,
        'is_conv': False,

    },

    'cont_learn_tec': {

        'importance': 1000,
        'sample_size': 250

    },

    'opt': {

        'lr': 1e-3,
        'l1_reg': 1e-4,
        'iters': 1,
        'batch_size': 64,
        'epochs': 10
    },

    'other': {

        'enable_tensorboard': True,
        'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        'model_name': '', # To be modified dinamically
        'run_name': 'mnist' # To be modified dinamically

    },

}