from operators.operator import Operator
from src.evolution.phenotype import Phenotype
from src import constants


class OperatorLib(object):

    def __init__(self):
        self.operators = []
        self.populate()

    def get_operators(self):
        return self.operators

    def populate(self):
        self.addSEQ()
        self.addPAR()
        self.addDOUB()
        self.addHALF()

        if constants.USE_CNN:
            self.addMAX_P()
            self.addDROP_20()
            self.addDROP_50()
            self.addDOUB_F()
            self.addKER_S()
            self.addPOOL_S()
        else:
            self.addSOFTMAX()
            self.addELU()
            self.addSOFTPLUS()
            self.addSOFTSIGN()
            self.addRELU()
            self.addTANH()
            self.addSIGMOID()
            self.addHSIGMOID()

    # Standard operators

    def addSEQ(self):
        operator = Operator('SEQ', 2)

        def func(node, index):
            next = Phenotype(index, node.neuron_count)
            next.add_input(node)
            next.copy_outputs(node)
            for n in node.outputs:
                n.inputs.remove(node)
            node.outputs = []
            node.add_output(next)
            return node, next  # return LEFT, RIGHT

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addPAR(self):
        operator = Operator('PAR', 2)

        def func(node, index):
            next = Phenotype(index, node.neuron_count)
            next.copy_inputs(node)
            next.copy_outputs(node)
            return node, next  # return LEFT, RIGHT

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addDOUB(self):
        operator = Operator('DOUB', 1)

        def func(node):
            node.multiply_neuron_count(2)
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addHALF(self):
        operator = Operator('HALF', 1)

        def func(node):
            node.divide_neuron_count(2)
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    # Convolutional operators

    def addMAX_P(self):
        operator = Operator('MAX_P', 1)

        def func(node):
            node.maxPooling = True
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addDROP_20(self):
        operator = Operator('DROP_20', 1)

        def func(node):
            node.dropout = 0.2
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addDROP_50(self):
        operator = Operator('DROP_50', 1)

        def func(node):
            node.dropout = 0.5
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addDOUB_F(self):
        operator = Operator('DOUB_F', 1)

        def func(node):
            node.filter_count *= 2
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addKER_S(self):
        operator = Operator('KER_S', 1)

        def func(node):
            node.kernel_size += 1
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addPOOL_S(self):
        operator = Operator('POOL_S', 1)

        def func(node):
            node.pool_size += 1
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    # activation function operators for

    def addSOFTMAX(self):
        operator = Operator('SOFTMAX', 1)

        def func(node):
            node.set_activation('softmax')
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addELU(self):
        operator = Operator('ELU', 1)

        def func(node):
            node.set_activation('elu')
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addSOFTPLUS(self):
        operator = Operator('SOFTPLUS', 1)

        def func(node):
            node.set_activation('softplus')
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addSOFTSIGN(self):
        operator = Operator('SOFTSIGN', 1)

        def func(node):
            node.set_activation('softsign')
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addRELU(self):
        operator = Operator('RELU', 1)

        def func(node):
            node.set_activation('relu')
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addTANH(self):
        operator = Operator('TANH', 1)

        def func(node):
            node.set_activation('tanh')
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addSIGMOID(self):
        operator = Operator('SIGMOID', 1)

        def func(node):
            node.set_activation('sigmoid')
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    def addHSIGMOID(self):
        operator = Operator('HSIGMOID', 1)

        def func(node):
            node.set_activation('hard_sigmoid')
            return node

        operator.set_pheno_func(func)
        self.operators.append(operator)

    # generic getter
    def getOperator(self, name):
        for n in self.operators:
            if n.name == name:
                return n
