from .lr_scheduler import GradualWarmupScheduler
from .radam import RAdam
import torch.optim


def get_optimizer(parameters, cfg):
    """
    Get optimizer by name
      - SGD: Typical Stochastic Gradient Descent. Default is SGD with nesterov momentum of 0.9.
      - ADAM: Adam optimizer
      - RADAM: Rectified Adam
    """
    if cfg.SOLVER.OPTIMIZER == "SGD":
        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, parameters),
            lr=cfg.SOLVER.BASE_LR,
            momentum=0.9,
            nesterov=True,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER == "ADAM":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, parameters),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )
    elif cfg.SOLVER.OPTIMIZER == "RADAM":
        from optim.radam import RAdam

        optimizer = RAdam(
            filter(lambda p: p.requires_grad, parameters),
            lr=cfg.SOLVER.BASE_LR,
            weight_decay=cfg.SOLVER.WEIGHT_DECAY,
        )

    return optimizer


def get_scheduler(optimizer, cfg):
    """
    Get scheduler by name
      - STATIC: No learning rate scheduling
      - EXP_DECAY: Decay the learning rate exponentially by rate of lr_gamma
      - COSINE: Scheduler the learning rate per cosine annealing.
    """
    if cfg.SOLVER.LR_SCHEDULER == "STATIC":
        scheduler = None
    elif cfg.SOLVER.LR_SCHEDULER == "EXP_DECAY":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, cfg.SOLVER.LR_GAMMA
        )
    elif cfg.SOLVER.LR_SCHEDULER == "COSINE":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, cfg.SOLVER.NUM_EPOCHS
        )
    elif cfg.SOLVER.LR_SCHEDULER == "MULTISTEP":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, cfg.SOLVER.LR_MILESTONES, cfg.SOLVER.MULTISTEP_GAMMA
        )

    return scheduler
