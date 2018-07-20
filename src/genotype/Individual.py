from src.genotype.operatorLib import OperatorLib
from src.genotype.gnode import Gnode
from src.phenotype.phenoConvertor import PhenoConvertor
from src import parameters


class Individual(object):

    def __init__(self):
        '''
            Constructor for Individual
            This class helps with mapping from genotype to phenotype
        '''
        self.functions = {}
        self.phenFunctions = {}
        self.registerOperators()

        # --- Define Tree ---
        root = Gnode()
        self.root = root

    def registerOperators(self):
        '''
            This registers all the genetic operators so that they can be accessible from Evolution Class
        '''
        opLib = OperatorLib()
        self.operators = opLib.getOperators()
        for op in self.operators:
            self.functions[op.name] = (op.func, op.arity, op.name)
            self.phenFunctions[op.name] = (op.phenoFunc, op.arity)

    def setGenotype(self, genotype):
        self.genotype = genotype

    def getPhenotype(self, individual):
        '''
            This class converts a genotype tree form into phenotype graph structure
        :param individual: genotype tree form
        :return: an array [phenoLib, order]
        '''
        phc = PhenoConvertor(self.phenFunctions)
        result = phc.resolvePheno(self.genotype, individual)  # this operation executes the conversion
        self.phenotype = result[0]
        return result

    def evaluatePhenotype(self, height, score):
        '''
            This function adds additional score logic to phenotype
            penalisation for neuron count
            penalisation for layer count

        :param height: height of the the genotype tree
        :param score: score gained from accuracy testing
        :return: new score
        '''
        neuron_count = self.iteratePheno(self.phenotype[0])
        accuracy_val = score * parameters.SCORE_CONST
        layer_count_val = (parameters.BLOAT_LIMIT - height) * parameters.SCORE_CONST_LAYER
        neuron_count_val = (parameters.MAX_NEURON_THRESHOLD - neuron_count) * parameters.SCORE_CONST_NEURON
        return accuracy_val + layer_count_val + neuron_count_val

    def iteratePheno(self, node):
        '''
            This iterates through phenotype and counts neurons
        :param node:
        :return:
        '''
        if len(node.outputs) == 0:
            return 0
        n_count = 0
        for o in node.outputs:
            n_count += o.neuron_count
            self.iteratePheno(o)
        return n_count + self.iteratePheno(o)
