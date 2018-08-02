import operator
import time
import random

from evolution.converter import Converter
from operators.operator_lib import OperatorLib
from src import constants
from deap import base
from deap import creator
from deap import tools
from deap import gp
from model_nn import ModelNN
from operator import attrgetter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import queue


def evolve():
    print(str(constants.NGEN) + ' Generationen mit ' + str(
        constants.POPS) + ' Netzen pro Generation.')

    # global counter for network printing
    global noOfNet
    noOfNet = 0

    pset = gp.PrimitiveSet("main", 0)

    primitives = OperatorLib().get_operators()
    for primitive in primitives:
        pset.addPrimitive(primitive.func, primitive.arity, primitive.name)

    pset.addTerminal('END')

    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluation of one individual
    def evaluate(individual):
        print(individual)
        genotype = gp.compile(individual, pset)

        functions = {}
        operators = OperatorLib().get_operators()
        for op in operators:
            functions[op.name] = (op.phenofunction, op.arity)
        converter = Converter(functions)
        phenotype = converter.resolve_pheno(genotype, individual)

        global noOfNet
        network = ModelNN(phenotype, noOfNet)
        noOfNet += 1
        print(network.test_acc)
        individual.score = network.test_acc
        individual.trainAcc = network.train_acc
        score_gen = network.test_acc
        return score_gen,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # decorate mate and mutate to prevent bloating
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=constants.BLOAT_LIMIT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=constants.BLOAT_LIMIT))

    random.seed(42)  # the meaning of life
    population = toolbox.population(n=constants.POPS)

    def log_pop(pop, c_index):
        pop.sort(key=attrgetter('fitness'), reverse=True)
        last_index = len(pop) - 1
        factor = c_index / (constants.NGEN * 1.0)
        completion_string = str(round(factor * 100, 2)) + "% abgeschlossen.\n"
        best_string = 'Bestes Individuum erreicht ' + str(pop[0].score) + '% Genauigkeit.'
        print(completion_string + best_string)
        worst.append(pop[last_index].score)
        best.append(pop[0].score)
        train_acc.append(pop[0].trainAcc)
        test_acc.append(pop[0].score)
        total_score = 0
        for individual in pop:
            total_score += individual.score
        average.append(total_score / constants.POPS)

    # Data for graph
    worst = []
    average = []
    best = []
    train_acc = []
    test_acc = []

    comp_index = 1
    for gen in range(constants.NGEN):
        if gen != constants.NGEN:
            selected = toolbox.select(population, len(population))
            offspring = [toolbox.clone(ind) for ind in selected]

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < constants.CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < constants.MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # mutate duplicates
            dup_counter = 0
            for outer in range(len(offspring)):
                for inner in range(outer + 1, len(offspring)):
                    if str(offspring[outer]) == str(offspring[inner]):
                        toolbox.mutate(offspring[inner])
                        del offspring[inner].fitness.values
                        dup_counter += 1
            print(str(dup_counter) + " Duplikate mutiert.")

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            print(str(len(invalid_ind)) + " ungültige Individuen werden neu ausgewertet.")
            fitness_list = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitness_list):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            population[:] = offspring

            log_pop(population, comp_index)
            comp_index += 1

    # create graph
    gene_arr = range(1, constants.NGEN + 1)
    plt.plot(gene_arr, train_acc, 'bo-')
    plt.plot(gene_arr, test_acc, 'go-')
    b_patch = mpatches.Patch(color='blue', label='TRAIN')
    g_patch = mpatches.Patch(color='green', label='TEST')
    plt.legend(handles=[b_patch, g_patch])
    plt.ylabel('ACCURACY')
    plt.xlabel('GENERATION NUMBER')
    plt.savefig(constants.KERASGRAPH_PATH)
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(gene_arr, worst, 'ro-')
    plt.plot(gene_arr, average, 'bo-')
    plt.plot(gene_arr, best, 'go-')
    red_patch = mpatches.Patch(color='red', label='Worst')
    blue_patch = mpatches.Patch(color='blue', label='Average')
    green_patch = mpatches.Patch(color='green', label='Best')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.ylabel('ACCURACY')
    plt.xlabel('GENERATION NUMBER')
    plt.savefig(constants.GENGRAPH_PATH)
    return population


# -------------- PRINT INDIVIDUAL GENOTYPE -----------------
def print_genotype(indi, root):
    outp = ''
    nodes = index_tree(root)
    for node in nodes:
        outp += get_ways(node)
    file = open(constants.GENOTYPE_PATH, 'a')
    file.write('\n\n' + str(indi) + '\n')
    if hasattr(indi, 'score'):
        file.write(str(indi.score) + '   ' + str(indi.height) + '\n')
    file.write('digraph{\n')
    file.write(outp)
    file.write('}')
    file.close()


def index_tree(node):
    que = queue.Queue()
    que.put(node)
    arr = []
    index = 0
    while not que.empty():
        node = que.get()
        arr.append(node)
        node.index = str(index) + '_' + node.get_type()
        index = index + 1
        if hasattr(node, 'left'):
            que.put(node.left)
        if hasattr(node, 'right'):
            que.put(node.right)
    return arr


def get_ways(node):
    outp = ''
    if hasattr(node, 'left'):
        outp += '\"' + str(node.index) + '\"' + '->' + '\"' + str(node.left.index) + '\"' + '\n'
    if hasattr(node, 'right'):
        outp += '\"' + str(node.index) + '\"' + '->' + '\"' + str(node.right.index) + '\"' + '\n'
    return outp


def print_tree(node, space):
    offset = get_offset(space)
    if type(node) is not 'str':
        print(offset + node.get_type())
    if not node.is_terminal():
        if hasattr(node, 'left'):
            print_tree(node.left, space + 1)
        if hasattr(node, 'right'):
            print_tree(node.right, space + 1)
    return


def get_offset(count):
    offset = ''
    for i in range(0, count):
        offset = offset + '\t'
    return offset


def debug_individual(individual):
    pset = gp.PrimitiveSet('main', 0)
    primitives = OperatorLib().get_operators()
    for primitive in primitives:
        pset.addPrimitive(primitive.func, primitive.arity, primitive.name)
    pset.addTerminal('END')
    tree = gp.PrimitiveTree.from_string(individual, pset)
    genotype = gp.compile(tree, pset)
    functions = {}
    operators = OperatorLib().get_operators()
    for op in operators:
        functions[op.name] = (op.phenofunction, op.arity)
    converter = Converter(functions)
    phenotype = converter.resolve_pheno(genotype, individual)
    network = ModelNN(phenotype, 9999)
    return network.test_acc, network.train_acc


def main():
    start = time.time()
    last_generation = evolve()
    end = time.time()
    print('Benötigte Zeit: '),
    print(str(end - start) + ' Sekunden.'),
    print('Siehe ' + constants.GENGRAPH_PATH + ' für eine detaillierte Auswertung aller Generationen.')
    print('Letzte Generation:')
    for ind in last_generation:
        print(ind)


def debug():
    print("DEBUG ACTIVE")
    individual = "PAR(DOUB(SEQ(DOUB(RELU(PAR(PAR(PAR(HALF(SOFTSIGN(ELU(DOUB(RELU(RELU('END')))))), 'END'), 'END'), RELU('END')))), 'END')), RELU(RELU('END')))"
    # SEQ(PAR(PAR(PAR(PAR(PAR(TANH('END'), 'END'), DOUB(DOUB('END'))), 'END'), 'END'), DOUB('END')), TANH('END'))
    # PAR(DOUB(SEQ(DOUB(RELU(PAR(PAR(PAR(HALF(SOFTSIGN(ELU(DOUB(RELU(RELU('END')))))), 'END'), 'END'), RELU('END')))), 'END')), RELU(RELU('END')))
    accs = debug_individual(individual)
    print(accs)


if __name__ == "__main__":
    # TODO: integrate preprocessor
    if constants.DEBUG:
        debug()
    else:
        main()
