from src.evolution.phenotype import Phenotype
import queue
from src import constants


class Converter(object):

    def __init__(self, phenotype_lib):
        self.phenotype_lib = phenotype_lib
        self.gbIndex = 1
        self.que = queue.Queue()
        self.nodeLib = []

    def resolve_pheno(self, root, individual):
        self.individual = individual
        self.input = Phenotype(0)
        mother = Phenotype(1)
        self.input.add_output(mother)
        mother.add_input(self.input)
        self.nodeLib.append(self.input)
        self.iterate_through(mother, root)
        order = self.get_order_bfs(self.input)
        self.print_phenotype(self.individual, order)
        library = self.convert_node_lib(self.nodeLib)
        return library, order

    def iterate_through(self, mother, root):
        backup = self.que
        self.que = queue.Queue()
        self.que.put([mother, root, 1])
        while not self.que.empty():
            item = self.que.get()
            if item[0] not in self.nodeLib:
                self.nodeLib.append(item[0])
            if not item[1].is_terminal():
                self.iterate_pheno(item[0], item[1])
        self.que = backup

    def iterate_pheno(self, node, genotype_node):
        operator = self.phenotype_lib[genotype_node.type]
        func = operator[0]
        arity = operator[1]
        if arity == 2:
            self.gbIndex = self.gbIndex + 1
        # --- result [left, right]
        if arity == 1:
            result = func(node)
            self.que.put([result, genotype_node.get_left()])
        if arity == 2:
            result = func(node, self.gbIndex)
            self.que.put([result[0], genotype_node.get_left()])
            self.que.put([result[1], genotype_node.get_right()])
        return

    def print_phenotype(self, ind, order):
        outp = ''
        outp += self.get_ways(self.nodeLib[0])
        for index in order:
            outp += self.get_ways(self.nodeLib[index])
        file = open(constants.OUTPUT_PHENOTYPE_TREE, 'a')
        file.write('\n\n' + str(ind) + '\n')
        file.write('digraph{\n')
        file.write(outp)
        file.write('}')
        file.close()

    def convert_node_lib(self, arr):
        lib = [None] * len(arr)
        for n in arr:
            lib[n.index] = n
        return lib

    def get_ways(self, node):
        outp = ''
        for o in node.outputs:
            inName = str(node.index) + '_' + str(node.neuron_count) + '_' + node.activation_function
            outName = str(o.index) + '_' + str(o.neuron_count) + '_' + o.activation_function
            outp += '\"' + inName + '\"->\"' + outName + '\"' + '\n'
        return outp

    def get_order_bfs(self, node):
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
