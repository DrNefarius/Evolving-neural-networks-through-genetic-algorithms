'''
This Object is used for implementing the genotype tree structure.
It is used as a link list to create binary trees.
Unary operators only have a left descendant.
'''

class Gnode(object):

    def __init__(self, type = None):
        '''
            Constructor for Genotype node
        :param type: genetic operator type
        '''
        if type is not None:
            self.type = type

        # rec count defines the number of recurrent calls, this value is inherited from root to leaves
        self.rec_count = 0


    # list of self explanatory setters & getters

    def setType(self, type):
        self.type = type

    def getRight(self):
        return self.right

    def setParent(self, node):
        self.parent = node

    def getLeft(self):
        return self.left

    def setRight(self, node):
        self.right = node

    def setLeft(self, node):
        self.left = node

    def isFinal(self):
        return self.type == 'END' or self.type == 'REC'

    def isREC(self):
        return self.type == 'REC'

    def isRECWeight(self):
        return self.type == 'REC_U' or self.type == 'REC_D'

    def modRecCount(self):
        if self.type == 'REC_U':
            self.rec_count += 1
        if self.type == 'REC_D' and self.rec_count > 0:
            self.rec_count -= 1

    def getType(self):
        return self.type