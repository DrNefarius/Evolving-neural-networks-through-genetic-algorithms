from operators.operatorLib import OperatorLib
from evolution.geno import geno
from src.evolution.treeconverter import PhenoConvertor


class Individual(object):

    def __init__(self):
        self.functions = {}
        self.phenofunctions = {}
        self.registerOperators()

        # --- Define Tree ---
        root = geno()
        self.root = root

    def registerOperators(self):
        opLib = OperatorLib()
        self.operators = opLib.getOperators()
        for op in self.operators:
            self.functions[op.name] = (op.func, op.arity, op.name)
            self.phenofunctions[op.name] = (op.phenofunction, op.arity)

    def setGenotype(self, genotype):
        self.genotype = genotype

    def getPhenotype(self, individual):
        phc = PhenoConvertor(self.phenofunctions)
        result = phc.resolve_pheno(self.genotype, individual)  # Conversion from genotype to phenotype
        self.phenotype = result[0]
        return result

