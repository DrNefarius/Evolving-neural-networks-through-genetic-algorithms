'''
    This class is used for creating a library that keeps a reference to all existing genetic operators
'''
from src.genotype.operator import Operator
from src.phenotype.pnode import Pnode
from src import parameters


class OperatorLib(object):

    def __init__(self):
        self.operators = []
        self.addOperators()

    def getOperators(self):
        return self.operators

    def addOperators(self):
        """
            Call all the operators, this call adds the nodes into the individual
        """
        self.addSEQ()
        self.addPAR()
        self.addDOUB()
        self.addHALF()

        if parameters.USE_CONVOLUTION_NN:
            self.addPOOL()
            self.addDDRrop20()
            # self.addDDRrop50()
            self.addDoubF()
            self.addKerS()
            self.addPoolS()
        else:
            # add actiavtion variations
            self.addSOFTMAX()
            self.addELU()
            self.addSOFTPLUS()
            self.addSOFTSIGN()
            self.addRELU()
            self.addTANH()
            self.addSIGMOID()
            self.addHSIGMOID()


    # ---- DEFINE OPERATORS ----
    # To define a new operator add function and call it in addOperators function, everything else is automatic

    # ---- CONVOLUTIONAL OPERATORS ----
    # each function in this section defines an operator

    def addPOOL(self):
        Op = Operator('MAX_P', 1)  # ENABLES MAX POOLING

        def func(node, index):
            node.maxPooling = True
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addDDRrop20(self):
        Op = Operator('DROP_20', 1)

        def func(node, index):
            node.dropout = 0.2
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addDDRrop50(self):
        Op = Operator('DROP_50', 1)

        def func(node, index):
            node.dropout = 0.5
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addDoubF(self):
        Op = Operator('DOUB_F', 1)  # DOUBLE FILTER COUNT

        def func(node, index):
            node.filter_count *= 2
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addKerS(self):
        Op = Operator('KER_S', 1)  # KERNEL SIZE +1

        def func(node, index):
            # node.kernel_size += 1
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addPoolS(self):
        Op = Operator('POOL_S', 1)  # POOL SIZE +1

        def func(node, index):
            # node.pool_size += 1
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    # ---- COMMON OPERATORS ----

    def addSEQ(self):
        Op = Operator('SEQ', 2)

        def func(node, index):
            next = Pnode(index, node.neuron_count)
            next.addInput(node)
            next.copyOutputs(node)
            for n in node.outputs:
                n.inputs.remove(node)
            node.outputs = []
            node.addOutput(next)
            return node, next  # return LEFT, RIGHT

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addPAR(self):
        Op = Operator('PAR', 2)

        def func(node, index):
            next = Pnode(index, node.neuron_count)
            next.copyInputs(node)
            next.copyOutputs(node)
            return node, next  # return LEFT, RIGHT

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addDOUB(self):
        Op = Operator('DOUB', 1)

        def func(node, index):
            node.multiplyNeuronCount(2)
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addHALF(self):
        Op = Operator('HALF', 1)

        def func(node, index):
            node.divideNeuronCount(2)
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    # ---- ACTIVATION FUNCTION OPERATORS ----

    def addSOFTMAX(self):
        Op = Operator('SOFTMAX', 1)

        def func(node, index):
            node.setActivation('softmax')
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addELU(self):
        Op = Operator('ELU', 1)

        def func(node, index):
            node.setActivation('elu')
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addSOFTPLUS(self):
        Op = Operator('SOFTPLUS', 1)

        def func(node, index):
            node.setActivation('softplus')
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addSOFTSIGN(self):
        Op = Operator('SOFTSIGN', 1)

        def func(node, index):
            node.setActivation('softsign')
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addRELU(self):
        Op = Operator('RELU', 1)

        def func(node, index):
            node.setActivation('relu')
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addTANH(self):
        Op = Operator('TANH', 1)

        def func(node, index):
            node.setActivation('tanh')
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addSIGMOID(self):
        Op = Operator('SIGMOID', 1)

        def func(node, index):
            node.setActivation('sigmoid')
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)

    def addHSIGMOID(self):
        Op = Operator('HSIGMOID', 1)

        def func(node, index):
            node.setActivation('hard_sigmoid')
            return node

        Op.setPhenoFunc(func)
        self.operators.append(Op)


    # ---- CONVOLUTION OPERATORS ----

    def getOperator(self, name):
        for n in self.operators:
            if n.name == name:
                return n
