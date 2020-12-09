from yacs.config import CfgNode as CN

_C = CN()

# System configs
_C.SYSTEM = CN()
_C.SYSTEM.DEVICE_IDS = ("0",)
_C.SYSTEM.NUM_GPUS = 1
_C.SYSTEM.MULTI_GPU = False
_C.SYSTEM.WORKERS = 0
_C.SYSTEM.PIN_MEMORY = True
_C.SYSTEM.SEED = 42

# Datasets configs
_C.DATASETS = CN()
_C.DATASETS.TRAIN = ''
_C.DATASETS.TEST = ''
_C.DATASETS.COLOR_MAP = 'RGB'   # 'RGB' or 'L'
_C.DATASETS.IMG_WIDTH = 336
_C.DATASETS.IMG_HEIGHT = 336
_C.DATASETS.MEAN = (0.485, 0.456, 0.406)
_C.DATASETS.STD = (0.229, 0.224, 0.225)
_C.DATASETS.MAX_NUM_DATA = 0
_C.DATASETS.TRAIN_TRANSFORM_TYPES = ('Resize', 'RandomHorizontalFlip', 'ToTensor', 'Normalize')
_C.DATASETS.TEST_TRANSFORM_TYPES = ('Resize', 'ToTensor', 'Normalize')
_C.DATASETS.TEST_SIZE = 0.1
_C.DATASETS.RANDOM_ROTATION = 0.5

# Model configs
_C.MODEL = CN()
_C.MODEL.BACKBONE = 'ResNet18'
_C.MODEL.BATCH_NORM = True
_C.MODEL.PRETRAINED = ''

# Solver configs
_C.SOLVER = CN()
_C.SOLVER.NUM_EPOCHS = 50
_C.SOLVER.BATCH_SIZE_PER_DEVICE = 64

_C.SOLVER.OPTIMIZER = 'RADAM'
_C.SOLVER.SAVEDIR = './result'
_C.SOLVER.CKPT_DIR = './checkpoint'
_C.SOLVER.PRINT_INTERVAL = 10
_C.SOLVER.VALID_INTERVAL = 1
_C.SOLVER.SAVE_INTERVAL = 1
_C.SOLVER.TB_LOG_INTERVAL = 10

_C.SOLVER.LR_SCHEDULER = 'STATIC'   # 'STATIC', 'EXP_DECAY', 'MULTISTEP', 'COSINE'
_C.SOLVER.LR_MILESTONES = (10,)
_C.SOLVER.MULTISTEP_GAMMA = 0.1
_C.SOLVER.BASE_LR = 0.001
_C.SOLVER.MOMENTUM = 0.9
_C.SOLVER.LR_GAMMA = 0.99

# warm up the learning rate
_C.SOLVER.LR_WARMUP = False
_C.SOLVER.WARUP_EPOCHS = 10
_C.SOLVER.LR_MULTIPLIER = 10

# For ADAM and RADAM
_C.SOLVER.BETA1 = 0.5
_C.SOLVER.WEIGHT_DECAY = 0.0005
_C.SOLVER.STEPS = (30000,)
_C.SOLVER.CHECKPOINT_PERIOD = 1

def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _C.clone()