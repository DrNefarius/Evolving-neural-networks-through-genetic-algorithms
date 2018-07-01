from src.phenotype.pnode import Pnode
from src.genotype.operatorLib import OperatorLib
import queue
from graphviz import Digraph
from src import parameters
import copy

class PhenoConvertor(object):

    def __init__(self, phenoLib):
        self.phenoLib = phenoLib
        self.gbIndex  = 2
        self.que = queue.Queue()
        self.nodeLib = []

    def resolvePheno(self, root, individual):
        self.individual = individual
        inputN = Pnode(0)
        self.input = inputN
        mother = Pnode(1)
        inputN.addOutput(mother)
        mother.addInput(inputN)
        self.nodeLib.append(inputN)
        self.iterateThrough(mother,root, 'basic')
        order = self.getOrderBFS(inputN)
        self.printPhenotype(self.individual, order)
        library = self.convertNodeLib(self.nodeLib)
        return library, order


    def iterateThrough(self, mother, root, level):
        backup = self.que
        self.que = queue.Queue()
        self.que.put([mother, root, 1])
        while not self.que.empty():
            item = self.que.get()
            name = str(item[1].index)
            if item[0] not in self.nodeLib:
                self.nodeLib.append(item[0])
            if level == 'basic':
                item[1].rec_count = item[2]
                if item[1].isRECWeight():
                    item[1].modRecCount()
            if item[1].isREC() and (level == name or level == 'basic'):
                item[1].rec_count -= 1
                if item[1].rec_count <= 0:
                    item[1].type = 'END'
                else:
                    self.iterateThrough(item[0], root, name)
            if not item[1].isFinal():
                self.iteratePheno(item[0], item[1])
        self.que = backup


    def iteratePheno(self, node, genoNode):
        operator = self.phenoLib[genoNode.type]
        func = operator[0]
        arity = operator[1]
        result = func(node, self.gbIndex)
        if arity == 2:
            self.gbIndex = self.gbIndex + 1
        # --- result [left, right]
        if arity == 1:
            self.que.put([result, genoNode.getLeft(), genoNode.rec_count])
        if arity == 2:
            self.que.put([result[0], genoNode.getLeft(), genoNode.rec_count])
            self.que.put([result[1], genoNode.getRight(), genoNode.rec_count])
        return

    def printPhenotype(self, ind, order):
        outp = ''
        outp += self.getWays(self.nodeLib[0])
        for index in order:
            outp += self.getWays(self.nodeLib[index])
        file = open(parameters.OUTPUT_PHENOTYPE_TREE, 'a')
        file.write('\n\n' + str(ind)+'\n')
        file.write('digraph{\n')
        file.write(outp)
        file.write('}')
        file.close()

    def convertNodeLib(self, arr):
        lib = [None] * len(arr)
        for n in arr:
            lib[n.index] = n
        return lib

    def getWays(self, node):
        outp = ''
        # for i in node.inputs:
        #     inName = str(i.index) + '_' + str(i.neuron_count)
        #     outName = str(node.index) + '_' + str(node.neuron_count)
        #     outp += '\"' + outName + '\"->\"' + inName + '\"' +'\n'
        for o in node.outputs:
            inName = str(node.index) + '_' + str(node.neuron_count)
            outName = str(o.index) + '_' + str(o.neuron_count)
            outp += '\"' + inName + '\"->\"' + outName + '\"' +'\n'
        return outp

    def getOrderBFS(self, node):
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