'''
    This object is used to wrap a genetic operator and all its properties
'''
from src.genotype.gnode import Gnode

class Operator(object):
    def __init__(self, name, arity):
        '''
            Constructor for operator
        :param name: The name of the node is equal to the genotype operator
        :param arity: number of descendants
        '''
        self.name = name
        self.arity = arity
        self.func = self.setFunction(arity)
        self.phenoFunc = None

    def setFunction(self, arity):
        '''
            Based on the arity this adds the function used for genotype tree building process
        :param arity: number of descendants (unary or binary)
        :return: reference to a function for genotype build
        '''
        if arity == 1:
            return self.unary
        if arity == 2:
            return self.binary
        if arity == 3:
            return self.ternary

    def setPhenoFunc(self, func):
        self.phenoFunc = func

    def unary(self, st):
        """
            Construct a genotype node
        :param st:
        :return:
        """
        root = Gnode(self.name)
        left = Gnode(st) if isinstance(st, str) else st
        root.setLeft(left)
        left.setParent(root)
        return root

    def binary(self, st, nd):
        """
            Construct a genotype node
        :param st: left child
        :param nd: right child
        :return: parent
        """
        root = Gnode(self.name)
        left = Gnode(st) if isinstance(st, str) else st
        right = Gnode(nd) if isinstance(nd, str) else nd
        root.setLeft(left)
        root.setRight(right)
        left.setParent(root)
        right.setParent(root)
        return root

    def ternary(self, st, nd, rd):
        print(self.name)
        return 'ternary'
