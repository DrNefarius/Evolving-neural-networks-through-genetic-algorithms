from src.evolution.phenotype import Phenotype
import queue
from src import constants


class Converter(object):
    """
    Convertes a genotype tree to the corresponding phenotypes.
    Phenotypes are getting saved in node_lib.
    """

    def __init__(self, phenotype_functions):
        self.phen_functions = phenotype_functions
        self.gb_index = 1
        self.que = queue.Queue()
        self.node_lib = []

    def resolve_pheno(self, genotype, individual):
        """
        Initializes the following process of conversion by setting up a mother node.
        Returns the phenotypes and the order in which they have to be applied.
        :param genotype: the complete genotype tree
        :param individual: the individual
        :return: the library of phenotype nodes and the order in which they have to be applied
        """
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
        """Iterates through the queue and resolves every connection in the genotype tree"""
        self.que = queue.Queue()
        self.que.put([mother, genotype])
        while not self.que.empty():
            item = self.que.get()
            if item[0] not in self.node_lib:
                self.node_lib.append(item[0])
            if not item[1].is_terminal():
                self.iterate_pheno(item[0], item[1])

    def iterate_pheno(self, node, genotype_node):
        """
        Applies the phenofunction on the phenotype node based on the genotype operator.
        Puts the resulting node and the rest of the genotype tree back into the queue for further resolving.
        :param node: the phenotype node
        :param genotype_node: the genotype node
        :return: nothing, but puts the resulting node and tree back into queue
        """
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
        """Prints the complete phenotype structure, which is basically a textmodel of the neural network"""
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
        """sorting of the node_lib for further work"""
        lib = [None] * len(arr)
        for n in arr:
            lib[n.index] = n
        return lib

    def get_ways(self, node):
        """Used to print the complete phenotype structure by print_phenotype"""
        outp = ''
        for o in node.outputs:
            inName = 'l' + str(node.index) + '_' + str(node.neurons) + 'n_' + node.activation_function
            outName = 'l' + str(o.index) + '_' + str(o.neurons) + 'n_' + o.activation_function
            outp += inName + ' -> ' + outName + '\n'
        return outp

    def sort_breadth_first(self, node):
        """Sorts the order of the indices by breadth first search"""
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
