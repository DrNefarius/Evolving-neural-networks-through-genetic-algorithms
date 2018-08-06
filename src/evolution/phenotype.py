from src import constants


class Phenotype(object):
    """
    Represents a phenotype. Saves all relevant information for building a Keras model.
    Also has some functions, that are used by some operators.
    Needs to be initialized with an index, that represents where in the to be created model this phenotype
    needs to be appplied.
    Also saves references to inputs and outputs of this phenotype.
    """

    def __init__(self, index, neurons=constants.MIN_NEURONS):
        self.inputs = []
        self.outputs = []
        self.index = index

        self.activation_function = constants.ACTIVATION_FUNCTION
        self.max_threshold = constants.MAX_NEURONS
        self.min_threshold = constants.MIN_NEURONS
        if neurons < constants.MIN_NEURONS:
            self.neurons = self.min_threshold
        else:
            self.neurons = neurons

        if constants.USE_CNN:
            self.dropout = constants.DROPOUT_START
            self.maxPooling = False
            self.filter_count = constants.FILTER_COUNT_MIN
            self.kernel_size = constants.KERNEL_SIZE_MIN
            self.pool_size = constants.POOL_SIZE_MIN

    def __eq__(self, other):
        return self.index == other.index

    def add_input(self, node):
        if node not in self.inputs:
            self.inputs.append(node)

    def add_output(self, node):
        if node not in self.outputs:
            self.outputs.append(node)

    def copy_inputs(self, node):
        for i in node.inputs:
            self.add_input(i)
            i.add_output(self)

    def swap_input_node(self, old, new):
        self.inputs.remove(old)
        self.inputs.append(new)

    def multiply_neuron_count(self, multi):
        count = self.neurons * multi
        if count < self.max_threshold:
            self.neurons = count
        else:
            self.neurons = self.max_threshold

    def divide_neuron_count(self, div):
        count = int(self.neurons / div)
        if count < constants.MIN_NEURONS:
            self.neurons = self.min_threshold
        else:
            self.neurons = count

    def set_activation(self, activation):
        self.activation_function = activation

    def copy_outputs(self, node):
        for o in node.outputs:
            self.add_output(o)
            o.add_input(self)

    def set_type(self, type):
        self.type = type

    def get_type(self):
        return self.type
