DEBUG = False

# -------------------- MAIN - (switch for testing specific solutions) --------------------
USE_CNN = False

# -------------------- TESTING DATASET --------------------
# DATASET = 'CIFAR'
DATASET = 'MNIST'

# -------------------- DEFAULT DATASET PARAMETERS --------------------
TRAIN_SIZE = 5000
INPUT_DIMENSION = 784
OUTPUT_DIMENSION = 10
BATCH_SIZE = 128

# -------------------- CONVOLUTION PARAMETERS --------------------
FILTER_COUNT = 32
DROPOUT = 0.5
KERNEL_SIZE = 3
POOL_SIZE = 2

# -------------------- SPECIFIC DATASET PARAMETERS --------------------
if DATASET == 'MNIST':
    TRAIN_SIZE = 6000
    IMG_DIMENSION = 28
    if USE_CNN:
        INPUT_DIMENSION = [IMG_DIMENSION, 1]
    else:
        INPUT_DIMENSION = IMG_DIMENSION * IMG_DIMENSION
    OUTPUT_DIMENSION = 10
    BATCH_SIZE = 128

if DATASET == 'CIFAR':
    TRAIN_SIZE = 50000
    IMG_DIMENSION = 32
    if USE_CNN:
        INPUT_DIMENSION = [IMG_DIMENSION, 3]
    else:
        INPUT_DIMENSION = IMG_DIMENSION * IMG_DIMENSION * 3
    OUTPUT_DIMENSION = 10
    BATCH_SIZE = 128

# -------------------- EVOLUTION PARAMETERS --------------------
POPS = 15
NGEN = 15
MUTPB = 0.1
CXPB = 0.5
BLOAT_LIMIT = 17
ELITISM_TUNE_UP = 0

# -------------------- FITNESS FUNCTION PARAMETERS --------------------
SCORE_CONST = 100000
SCORE_CONST_LAYER = 100
SCORE_CONST_NEURON = 0.01

# -------------------- PHENOTYPE PROPERTIES --------------------
NEURON_COUNT = 100
ACTIVATION_FUNCTION = 'relu'
MAX_NEURON_THRESHOLD = 10000
MIN_NEURON_THRESHOLD = NEURON_COUNT

# -------------------- KERAS --------------------
ACTIVATION_FUNCTION_FOR_EXIT = 'softmax'
OPTIMIZER = 'adam'
LOSS_FUNCTION = 'categorical_crossentropy'

K_EPOCHS = 25
K_CLASS_COUNT = OUTPUT_DIMENSION
K_VERBOSE = 0  # 0 = silent || 1 = progress bar || 2 = show epoch

BASE_LINK = "C:\\Users\\Tobias\\PycharmProjects\\master\\src"

GENOTYPE_PATH = BASE_LINK + '\\output\\Genotype.txt'
PHENOTYPE_PATH = BASE_LINK + '\\output\\Phenotype.txt'
GENGRAPH_PATH = BASE_LINK + '\\output\\gen.png'
KERASGRAPH_PATH = BASE_LINK + '\\output\\keras.png'
