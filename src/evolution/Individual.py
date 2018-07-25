from operators.operator_lib import OperatorLib
from evolution.geno import geno
from src.evolution.treeconverter import converter


class Individual(object):

    def __init__(self):
        self.functions = {}
        self.phenofunctions = {}
        self.register_operators()

        # --- Define Tree ---
        root = geno()
        self.root = root

    def register_operators(self):
        opLib = OperatorLib()
        self.operators = opLib.get_operators()
        for op in self.operators:
            self.functions[op.name] = (op.func, op.arity, op.name)
            self.phenofunctions[op.name] = (op.phenofunction, op.arity)

    def set_genotype(self, genotype):
        self.genotype = genotype

    def get_phenotype(self, individual):
        conv = converter(self.phenofunctions)
        result = conv.resolve_pheno(self.genotype, individual)  # Conversion from genotype to phenotype
        self.phenotype = result[0]
        return result

