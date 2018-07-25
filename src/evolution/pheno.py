from src import parameters


class Pheno(object):

    def __init__(self, index, neuroncount=parameters.NEURON_COUNT):
        self.inputs = []
        self.outputs = []
        self.index = index

        self.activation_function = parameters.ACTIVATION_FUNCTION
        self.maxThreshold = parameters.MAX_NEURON_THRESHOLD
        self.minThreshold = parameters.MIN_NEURON_THRESHOLD
        self.layer_type = parameters.ACTIVATION_FUNCTION
        if neuroncount < parameters.MIN_NEURON_THRESHOLD:
            self.neuron_count = self.minThreshold
        else:
            self.neuron_count = neuroncount

        if (parameters.USE_CONVOLUTION_NN):
            self.dropout = parameters.DROUPOUT
            self.maxPooling = True
            self.filter_count = parameters.FILTER_COUNT
            self.kernel_size = parameters.KERNEL_SIZE
            self.pool_size = parameters.POOL_SIZE

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
        if count < parameters.MIN_NEURON_THRESHOLD:
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
