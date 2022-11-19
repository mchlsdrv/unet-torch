import pathlib
from torch import nn

__author__ = 'mchlsdrv@gmail.com'

# PATHS
TRAIN_DATA_DIR = pathlib.Path('../../../data/yeast/batch1')
TEST_DATA_DIR = pathlib.Path('../../../data/yeast/batch2')
INFERENCE_DATA_DIR = pathlib.Path('../../../data/yeast/batch2/images')

TRAIN_CONTINUE = False
OUTPUT_TS = ''
CHECKPOINT_DIR = pathlib.Path(f'./output/{OUTPUT_TS}/checkpoints/')

DELETE_ON_FINISH = False

# > AWS
AWS_INPUT_BUCKET_NAME = 'nanosct-service'
AWS_INPUT_BUCKET_SUBDIR = 'input/'
AWS_INPUT_REGION = 'us-east-1'

# - OUTPUT -
OUTPUT_DIR = pathlib.Path('./output')

# > AWS
AWS_OUTPUT_BUCKET_NAME = 'nanosct-service'
AWS_OUTPUT_BUCKET_SUBDIR = 'output/'
AWS_OUTPUT_REGION = 'us-east-1'

TEMP_DIR = pathlib.Path('./temp')

CONFIGS_DIR_PATH = pathlib.Path('./configs')

MODEL_CONFIGS_DIR_PATH = CONFIGS_DIR_PATH

# CONTROL VARIABLES
DEBUG_LEVEL = 0
PROFILE = False

# CONSTANTS
EPSILON = 1e-7

# DATA
DATA_FORMAT = 'tiff'
PREPROCESS_IMAGE = True
IMAGE_HEIGHT = 128
IMAGE_WIDTH = 128
MASK_BINARY = True
MASK_ZERO_BOUNDARY = False
INFO_BAR_HEIGHT = 70
IN_CHANNELS = 1
OUT_CHANNELS = 1

# SHUFFLE
SHUFFLE = True

# TRAINING
LOAD_MODEL = False
CHECKPOINT_FILE = pathlib.Path('/home/sidorov/Projects/nanoscout-models/segmentation/code/pytorch/unet/1k_epchs/checkpoints/best_val_loss_chkpt.pth.tar')
EPOCHS = 100
BATCH_SIZE = 16
PIN_MEMORY = True
NUM_WORKERS = 2
# LOSS_FN = nn.CrossEntropyLoss()  # => for more than 1 prediction class
LOSS_FN = nn.BCEWithLogitsLoss()  # => for 1 prediction class

# - OPTIMIZER
# > Arguments
OPTIMIZER = 'adam'
OPTIMIZER_LR = 1e-4
OPTIMIZER_LR_DECAY = 0.001
OPTIMIZER_BETA_1 = 0.9
OPTIMIZER_BETA_2 = 0.999
OPTIMIZER_WEIGHT_DECAY = 0.01
OPTIMIZER_MOMENTUM = 0.1
OPTIMIZER_DAMPENING = 0.01
OPTIMIZER_MOMENTUM_DECAY = 0.01
OPTIMIZER_RHO = 0.9
OPTIMIZER_AMSGRAD = False

# > Parameters
OPTIMIZER_MAXIMIZE = False
OPTIMIZER_EPS = 1e-8
OPTIMIZER_FOREACH = None

# - REGULARIZER
KERNEL_REGULARIZER_TYPE = 'l2'
KERNEL_REGULARIZER_L1 = 0.01
KERNEL_REGULARIZER_L2 = 0.01
KERNEL_REGULARIZER_FACTOR = 0.01
KERNEL_REGULARIZER_MODE = 'rows'

FALSE_POSITIVES_WEIGHT = .9
FALSE_NEGATIVES_WEIGHT = 1.2

IOU_THRESHOLD = 0.3
TRAIN_SCORE_THRESHOLD = 0.8
INFERENCE_SCORE_THRESHOLD = 0.85
N_TOP_PREDS = 100

#  Variables
VALIDATION_BATCH_SIZE = 10
VAL_PROP = .2
N_LOGS = 5

# > CALLBACKS

# - LR Reduce
REDUCE_LR_ON_PLATEAU_FACTOR = 0.5
REDUCE_LR_ON_PLATEAU_MIN = 1e-8

# - Model Checkpoint
MODEL_CHECKPOINT = True
MODEL_CHECKPOINT_FILE_TEMPLATE = 'my_checkpoint.pth.tar'  # <- may be used in case we want to save all teh check points, and not only the best
MODEL_CHECKPOINT_MONITOR = 'val_loss'
MODEL_CHECKPOINT_VERBOSE = 1
MODEL_CHECKPOINT_SAVE_BEST_ONLY = True
MODEL_CHECKPOINT_MODE = 'auto'
MODEL_CHECKPOINT_SAVE_WEIGHTS_ONLY = True
MODEL_CHECKPOINT_SAVE_FREQ = 'epoch'
