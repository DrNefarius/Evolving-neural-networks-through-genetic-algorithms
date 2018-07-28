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
POPULATION_SIZE = 15
NUMBER_OF_GENERATIONS = 15
MUTATION_PROBABILITY = 0.1
CROSSOVER_PROBABILITY = 0.5
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

LEARN_EPOCH_COUNT = 25
OUTPUT_CLASS_COUNT = OUTPUT_DIMENSION
VERBOSE = 0

# -------------------- OUTPUT FILES --------------------
# OUTPUT = '' # for local pycharm run
OUTPUT = "C:\\Users\\Tobias\\PycharmProjects\\dolezal\\src"  # for server run

OUTPUT_MUTATION_CHANGE = OUTPUT + '\\output\\mutchanges.txt'
OUTPUT_ACCURACY = OUTPUT + '\\output\\accuracy.txt'
OUTPUT_GENOTYPE_TREE = OUTPUT + '\\output\\Genotype.txt'
OUTPUT_PHENOTYPE_TREE = OUTPUT + '\\output\\Phenotype.txt'
OUTPUT_MODEL = OUTPUT + '\\output\\model_'
OUTPUT_GRAPH = OUTPUT + '\\output\\graph.png'
OUTPUT_TRAINTEST = OUTPUT + '\\output\\trainAndTest.png'
