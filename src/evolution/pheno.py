from src import parameters


class pheno(object):

    def __init__(self, index, neuroncount=parameters.NEURON_COUNT):
        self.inputs = []
        self.outputs = []
        self.index = index

        # DEFINE LAYER
        self.act_func = parameters.ACTIVATION_FUNCTION
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

    def addInput(self, node):
        if node not in self.inputs:
            self.inputs.append(node)

    def addOutput(self, node):
        if node not in self.outputs:
            self.outputs.append(node)

    def copyInputs(self, node):
        for i in node.inputs:
            self.addInput(i)
            i.addOutput(self)

    def swapInputNode(self, old, new):
        self.inputs.remove(old)
        self.inputs.append(new)

    def multiplyNeuronCount(self, multi):
        count = self.neuron_count * multi
        if count < self.maxThreshold:
            self.neuron_count = count
        else:
            self.neuron_count = self.maxThreshold

    def divideNeuronCount(self, div):
        count = int(self.neuron_count / div)
        if count < parameters.MIN_NEURON_THRESHOLD:
            self.neuron_count = self.minThreshold
        else:
            self.neuron_count = count

    def setActivation(self, activation):
        self.act_func = activation

    def copyOutputs(self, node):
        for o in node.outputs:
            self.addOutput(o)
            o.addInput(self)

    def setType(self, type):
        self.type = type

    def getType(self):
        return self.type
