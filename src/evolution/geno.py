'''
This Object is used for implementing the genotype tree structure.
It is used as a link list to create binary trees.
Unary operators only have a left descendant.
'''


class geno(object):

    def __init__(self, type=None):
        if type is not None:
            self.type = type

    def setType(self, type):
        self.type = type

    def setParent(self, node):
        self.parent = node

    def setRight(self, node):
        self.right = node

    def setLeft(self, node):
        self.left = node

    def getRight(self):
        return self.right

    def getLeft(self):
        return self.left

    def getType(self):
        return self.type

    def isTerminal(self):
        return self.type == 'END'
