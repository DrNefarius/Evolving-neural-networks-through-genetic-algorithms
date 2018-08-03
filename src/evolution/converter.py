from src.evolution.phenotype import Phenotype
import queue
from src import constants


class Converter(object):

    def __init__(self, phenotype_functions):
        self.phen_functions = phenotype_functions
        self.gb_index = 1
        self.que = queue.Queue()
        self.node_lib = []

    def resolve_pheno(self, genotype, individual):
        self.individual = individual
        self.input = Phenotype(0)
        mother = Phenotype(1)
        self.input.add_output(mother)
        mother.add_input(self.input)
        self.node_lib.append(self.input)
        self.iterate_through(mother, genotype)
        order = self.sort_breadth_first(self.input)
        self.print_phenotype(self.individual, order)
        library = self.convert_node_lib(self.node_lib)
        return library, order

    def iterate_through(self, mother, genotype):
        self.que = queue.Queue()
        self.que.put([mother, genotype])
        while not self.que.empty():
            item = self.que.get()
            if item[0] not in self.node_lib:
                self.node_lib.append(item[0])
            if not item[1].is_terminal():
                self.iterate_pheno(item[0], item[1])

    def iterate_pheno(self, node, genotype_node):
        operator = self.phen_functions[genotype_node.type]
        func = operator[0]
        arity = operator[1]
        if arity == 1:
            result = func(node)
            self.que.put([result, genotype_node.get_left()])
        elif arity == 2:
            self.gb_index = self.gb_index + 1
            result = func(node, self.gb_index)
            self.que.put([result[0], genotype_node.get_left()])
            self.que.put([result[1], genotype_node.get_right()])
        return

    def print_phenotype(self, ind, order):
        outp = ''
        outp += self.get_ways(self.node_lib[0])
        for index in order:
            outp += self.get_ways(self.node_lib[index])
        file = open(constants.PHENOTYPE_PATH, 'a')
        file.write('\n\n' + str(ind) + '\n')
        file.write('Netzstruktur:\n')
        file.write(outp)
        file.write('')
        file.close()

    def convert_node_lib(self, arr):
        lib = [None] * len(arr)
        for n in arr:
            lib[n.index] = n
        return lib

    def get_ways(self, node):
        outp = ''
        for o in node.outputs:
            inName = 'l' + str(node.index) + '_' + str(node.neurons) + 'n_' + node.activation_function
            outName = 'l' + str(o.index) + '_' + str(o.neurons) + 'n_' + o.activation_function
            outp += inName + ' -> ' + outName + '\n'
        return outp

    def sort_breadth_first(self, node):
        que = queue.Queue()
        que.put(node)
        order = []
        while not que.empty():
            node = que.get()
            for out in node.outputs:
                if out.index not in order:
                    order.append(out.index)
                que.put(out)
        return order
