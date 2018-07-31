from src import constants


class Phenotype(object):

    def __init__(self, index, neuroncount=constants.NEURONS):
        self.inputs = []
        self.outputs = []
        self.index = index

        self.activation_function = constants.ACTIVATION_FUNCTION
        self.maxThreshold = constants.MAX_NEURONS
        self.minThreshold = constants.MIN_NEURONS
        self.layer_type = constants.ACTIVATION_FUNCTION
        if neuroncount < constants.MIN_NEURONS:
            self.neuron_count = self.minThreshold
        else:
            self.neuron_count = neuroncount

        if (constants.USE_CNN):
            self.dropout = constants.DROPOUT
            self.maxPooling = True
            self.filter_count = constants.FILTER_COUNT
            self.kernel_size = constants.KERNEL_SIZE
            self.pool_size = constants.POOL_SIZE

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
        count = self.neuron_count * multi
        if count < self.maxThreshold:
            self.neuron_count = count
        else:
            self.neuron_count = self.maxThreshold

    def divide_neuron_count(self, div):
        count = int(self.neuron_count / div)
        if count < constants.MIN_NEURONS:
            self.neuron_count = self.minThreshold
        else:
            self.neuron_count = count

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
