from evolution.geno import geno


class Operator(object):
    def __init__(self, name, arity):
        self.name = name
        self.arity = arity
        self.func = self.setFunction(arity)
        self.phenofunction = None

    def setFunction(self, arity):
        if arity == 1:
            return self.unary
        elif arity == 2:
            return self.binary
        else:
            raise ValueError(
                "Stelligkeitswert für Operatorfunktion nicht unterstützt (ist höher als 2 oder niedriger als 1)")

    def setPhenoFunc(self, function):
        self.phenofunction = function

    def unary(self, first):
        root = geno(self.name)
        left = geno(first) if isinstance(first, str) else first
        root.setLeft(left)
        left.setParent(root)
        return root

    def binary(self, first, second):
        root = geno(self.name)
        left = geno(first) if isinstance(first, str) else first
        right = geno(second) if isinstance(second, str) else second
        root.setLeft(left)
        root.setRight(right)
        left.setParent(root)
        right.setParent(root)
        return root
