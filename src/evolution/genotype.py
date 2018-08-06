class Genotype(object):
    """
    Represents a node in the genotype tree.
    Based on simple tree logic, could be the root of a complete tree, a leaf or a subtree.
    """

    def __init__(self, type=None):
        """
        :param type: equals the string name of the operator assigned to this genotype node
        """
        if type is not None:
            self.type = type
        self.parent = None
        self.right = None
        self.left = None

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
