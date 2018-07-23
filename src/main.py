import operator
import time
import random
from src import parameters
from deap import base
from deap import creator
from deap import tools
from deap import gp
from Individual import Individual
from modelNN import ModelNN
from operator import attrgetter
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import queue
import sys


# matplotlib.use('Agg')

def evolve():
    '''
        This function defines all the important things for the EA to work properly
    :return:
    '''
    print(str(parameters.NUMBER_OF_GENERATIONS) + ' Generationen mit ' + str(
        parameters.POPULATION_SIZE) + ' Netzen pro Generation.')

    # global counter for network printing
    global noOfNet
    noOfNet = 0

    pset = gp.PrimitiveSet("main", 0)
    primitives = Individual()
    for key, value in primitives.functions.items():  # getting all primitives from
        pset.addPrimitive(value[0], value[1], value[2])
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
        converter = Individual()
        converter.setGenotype(genotype)
        printGenotype(individual, genotype)
        result = converter.getPhenotype(individual)
        global noOfNet
        krs = ModelNN(result, noOfNet)
        noOfNet += 1
        print(krs.testAcc)
        individual.score = krs.testAcc
        individual.trainAcc = krs.trainAcc
        scoreGen = krs.testAcc
        return scoreGen,

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # decorate mate and mutate to prevent bloating
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=parameters.BLOAT_LIMIT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=parameters.BLOAT_LIMIT))

    random.seed(42)  # the meaning of life
    population = toolbox.population(n=parameters.POPULATION_SIZE)

    def logPop(population, compIndex):
        population.sort(key=attrgetter('fitness'), reverse=True)
        lastIndex = len(population) - 1
        comp = compIndex / (parameters.NUMBER_OF_GENERATIONS * 1.0)
        compl = 'COMPLETE [ ' + str(round(comp * 100, 2)) + ' ] \n'
        gene = 'BEST SOLUTION: [ ' + str(population[0].score) + ' %] '
        print(sys.stderr, compl + gene)
        wrstArr.append(population[lastIndex].score)
        bestArr.append(population[0].score)
        trainAcc.append(population[0].trainAcc)
        testAcc.append(population[0].score)
        totalScore = 0
        for ind in population:
            totalScore += ind.score
        avgArr.append(totalScore / parameters.POPULATION_SIZE)

    # Data for graph
    wrstArr = []  # WRST RED
    avgArr = []  # AVG BLUE
    bestArr = []  # BEST GREEN
    trainAcc = []
    testAcc = []

    # Evaluate the entire generation
    fitnesses = map(toolbox.evaluate, population)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    compIndex = 1
    for gen in range(parameters.NUMBER_OF_GENERATIONS):
        logPop(population, compIndex)
        compIndex += 1
        if gen != parameters.NUMBER_OF_GENERATIONS:
            selected = toolbox.select(population, len(population))
            offspring = [toolbox.clone(ind) for ind in selected]

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < parameters.CROSSOVER_PROBABILITY:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < parameters.MUTATION_PROBABILTY:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            # Remove duplicates in the offspring
            for outer in range(len(offspring)):
                for inner in range(outer + 1, len(offspring)):
                    if str(offspring[outer]) == str(offspring[inner]):
                        print(' *')
                        # list(offspring)[inner] = mutateWithLimit(list(offspring)[inner])
                        toolbox.mutate(offspring[inner])
                        del offspring[inner].fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            population[:] = offspring

    # ------------------------------  PLOT THE TRAIN AND TEST ACCURACY ---------------------------------------------
    # Data for graph
    geneArr = range(1, parameters.NUMBER_OF_GENERATIONS + 1)
    plt.plot(geneArr, trainAcc, 'bo-')  # TRAIN BLUE
    plt.plot(geneArr, testAcc, 'go-')  # TEST GREEN
    b_patch = mpatches.Patch(color='blue', label='TRAIN')
    g_patch = mpatches.Patch(color='green', label='TEST')
    plt.legend(handles=[b_patch, g_patch])
    plt.ylabel('ACCURACY')
    plt.xlabel('GENERATION NUMBER')
    plt.savefig(parameters.OUTPUT_TRAINTEST)
    # ----------------------------------------------------------------------------------------------------------
    plt.clf()
    plt.cla()
    plt.close()
    # ------------------------------  PLOT THE EVOLUTION ------------------------------------------------------
    plt.plot(geneArr, wrstArr, 'ro-')  # WRST RED
    plt.plot(geneArr, avgArr, 'bo-')  # AVG BLUE
    plt.plot(geneArr, bestArr, 'go-')  # BEST GREErN
    red_patch = mpatches.Patch(color='red', label='Worst')
    blue_patch = mpatches.Patch(color='blue', label='Average')
    green_patch = mpatches.Patch(color='green', label='Best')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.ylabel('ACCURACY')
    plt.xlabel('GENERATION NUMBER')
    plt.savefig(parameters.OUTPUT_GRAPH)
    # ----------------------------------------------------------------------------------------------------------
    return population


# -------------- PRINT INDIVIDUAL GENOTYPE -----------------
def printGenotype(indi, root):
    outp = ''
    genoArr = indexTree(root)
    for node in genoArr:
        outp += getWays(node)
    file = open(parameters.OUTPUT_GENOTYPE_TREE, 'a')
    file.write('\n\n' + str(indi) + '\n')
    if hasattr(indi, 'score'):
        file.write(str(indi.score) + '   ' + str(indi.height) + '\n')
    file.write('digraph{\n')
    file.write(outp)
    file.write('}')
    file.close()


def indexTree(node):
    que = queue.Queue()
    que.put(node)
    arr = []
    index = 0
    while not que.empty():
        node = que.get()
        arr.append(node)
        node.index = str(index) + '_' + node.getType()
        index = index + 1
        if hasattr(node, 'left'):
            que.put(node.left)
        if hasattr(node, 'right'):
            que.put(node.right)
    return arr


def getWays(node):
    outp = ''
    if hasattr(node, 'left'):
        outp += '\"' + str(node.index) + '\"' + '->' + '\"' + str(node.left.index) + '\"' + '\n'
    if hasattr(node, 'right'):
        outp += '\"' + str(node.index) + '\"' + '->' + '\"' + str(node.right.index) + '\"' + '\n'
    return outp


def printTree(node, space):
    offset = getOffset(space)
    if type(node) is not 'str':
        print(offset + node.getType())
    if not node.isTerminal():
        if hasattr(node, 'left'):
            printTree(node.left, space + 1)
        if hasattr(node, 'right'):
            printTree(node.right, space + 1)
    return


def getOffset(count):
    str = ''
    for i in range(0, count):
        str = str + '\t'
    return str


def debugIndividual(individual):
    pset = gp.PrimitiveSet('main', 0)
    indi = Individual()
    for key, value in indi.functions.items():
        pset.addPrimitive(value[0], value[1], value[2])
    pset.addTerminal('END')
    tree = gp.PrimitiveTree.from_string(individual, pset)
    genotype = gp.compile(tree, pset)
    indi.setGenotype(genotype)
    phenotype = indi.getPhenotype(individual)
    network = ModelNN(phenotype, 9999)
    return network.testAcc, network.trainAcc


def main():
    start = time.time()
    results = evolve()
    end = time.time()
    print('Benötigte Zeit: '),
    print(end - start),
    print(' Sekunden.')
    print('Letzte Generation: ')
    if len(results) > 0:
        for i in results:
            print(i.score / 100),
            print('   ----    '),
            print(i)


def debug():
    print("DEBUG ACTIVE")
    individual = "PAR(DOUB(SEQ(DOUB(RELU(PAR(PAR(PAR(HALF(SOFTSIGN(ELU(DOUB(RELU(RELU('END')))))), 'END'), 'END'), RELU('END')))), 'END')), RELU(RELU('END')))"
    # SEQ(PAR(PAR(PAR(PAR(PAR(TANH('END'), 'END'), DOUB(DOUB('END'))), 'END'), 'END'), DOUB('END')), TANH('END'))
    # PAR(DOUB(SEQ(DOUB(RELU(PAR(PAR(PAR(HALF(SOFTSIGN(ELU(DOUB(RELU(RELU('END')))))), 'END'), 'END'), RELU('END')))), 'END')), RELU(RELU('END')))
    accs = debugIndividual(individual)
    print(accs)


if __name__ == "__main__":
    # TODO: integrate preprocessor
    if parameters.DEBUG:
        debug()
    else:
        main()
