'''
This Object is used for implementing the genotype tree structure.
It is used as a link list to create binary trees.
Unary operators only have a left descendant.
'''


class geno(object):

    def __init__(self, type=None):
        if type is not None:
            self.type = type

    def set_type(self, type):
        self.type = type

    def set_parent(self, node):
        self.parent = node

    def set_right(self, node):
        self.right = node

    def set_left(self, node):
        self.left = node

    def get_right(self):
        return self.right

    def get_left(self):
        return self.left

    def get_type(self):
        return self.type

    def is_terminal(self):
        return self.type == 'END'
