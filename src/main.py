import time
import random
from src import parameters
from deap import base
from deap import creator
from deap import tools
from deap import gp
from src.genotype.Individual import Individual
from src.phenotype.modelNN import ModelNN
from operator import attrgetter
import matplotlib
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
    pset = gp.PrimitiveSet("main", 0)
    # --- Define functions ---
    Ind = Individual()
    for key, value in Ind.functions.items():  # this for cycle defines all the genetic operators
        pset.addPrimitive(value[0], value[1], value[2])

    # --- Define terminals ---
    pset.addTerminal('END')

    # --- Define creator ---
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)
    global noOfNet
    noOfNet = 0

    # --- Define Fitness function ---
    def evalGenotype(individual):
        print(individual)
        genotype = gp.compile(individual, pset)
        Ind.setGenotype(genotype)
        printGenotype(individual, genotype)
        result = Ind.getPhenotype(individual)
        global noOfNet
        krs = ModelNN(result, noOfNet)
        noOfNet += 1
        print(krs.testAcc)
        individual.score = krs.testAcc
        individual.trainAcc = krs.trainAcc
        scoreGen = krs.testAcc
        # scoreGen = Ind.evaluatePhenotype(individual.height, krs.testAcc)
        return scoreGen,

    # --- Define the mutation function --
    # def mutateWithLimit(ancestor):
    #     mutant = toolbox.clone(ancestor)
    #     toolbox.mutate(mutant)
    #     while mutant.height > BLOAT_LIMIT:
    #         mutant = toolbox.clone(ancestor)
    #         toolbox.mutate(mutant)
    #     del mutant.fitness.values
    #     return mutant

    # --- Define toolbox ---
    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evalGenotype)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # toolbox.decorate("mate", gp.staticLimit('height', 17))
    # toolbox.decorate("mutate", gp.staticLimit('height', 17))

    # --- Init params and call ---
    POPS = parameters.POPULATION_SIZE
    ELITISM = parameters.ELITISM
    NGEN = parameters.NUMBER_OF_GENERATIONS
    MUTPB = parameters.MUTATION_PROBABILTY
    BLOAT_LIMIT = parameters.BLOAT_LIMIT
    CXPB = parameters.CROSSOVER_PROBABILITY

    random.seed(69)
    population = toolbox.population(n=POPS)

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
    population.sort(key=attrgetter('fitness'), reverse=True)

    compIndex = 1
    for g in range(NGEN):
        population.sort(key=attrgetter('fitness'), reverse=True)
        # ------------------------------  LOG THE POPULATION ------------------------------------------------------
        lastIndex = len(population) - 1
        comp = compIndex / (NGEN * 1.0)
        compIndex += 1
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
        avgArr.append(totalScore / POPS)
        if g != NGEN:
            # ----------------------------------------------------------------------------------------------------------

            selected = toolbox.select(population, len(population))
            offspring = [toolbox.clone(ind) for ind in selected]

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            # Apply mutation on the offspring
            for mutant in offspring:
                if random.random() < MUTPB:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values
            # for ofIndex in range(len(list(offspring))):
            #     if len(list(offspring)) > 0:
            #         mutant = list(offspring)[ofIndex]
            #         if random.random() < MUTPB:
            #             ancestor = toolbox.clone(mutant)
            #             offspring[ofIndex] = mutateWithLimit(mutant)
            #             printMutationChange(ancestor, offspring[ofIndex])

            # Use elitism for
            list(offspring).sort(key=attrgetter('fitness'), reverse=True)  # <------------  SORT !
            for ind in range(ELITISM):
                potential = population[ind]
                if list(offspring)[POPS - 1 - ind].score < potential.score:
                    list(offspring)[POPS - 1 - ind] = potential

            # Remove duplicates in the offspring
            list(offspring).sort(key=attrgetter('fitness'), reverse=True)  # <------------  SORT !
            for outer in range(len(list(offspring))):
                for inner in range(outer + 1, len(list(offspring))):
                    if str(list(offspring)[outer]) == str(list(offspring)[inner]):
                        print(' *')
                        # list(offspring)[inner] = mutateWithLimit(list(offspring)[inner])
                        toolbox.mutate(list(offspring)[inner])
                        del list(offspring)[inner].fitness.values

            # # Try selective mutations on BEST individuals
            # list(offspring).sort(key=attrgetter('fitness'), reverse=True)  # <------------  SORT !
            # needsSorting = False
            # print('TRY UPGRADE')
            # for index in range(parameters.ELITISM_TUNE_UP):
            #     best = toolbox.clone(list(offspring)[0])
            #     toolbox.mutate(best)
            #     newBest = mutateWithLimit(best)
            #     print('[  ' + str(best.score) + ' -- > ' + str(newBest.score) + '  ]')
            #     if newBest.score > best.score:
            #         print('******')
            #         list(offspring)[POPS - 1 - ind] = newBest
            #         needsSorting = True
            #         break
            #
            # if needsSorting:
            #     list(offspring).sort(key=attrgetter('fitness'), reverse=True)  # <------------  SORT !

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # ----------- PRINT THE WHOLE POPULATION
            printPopulationGenotypes = True
            if printPopulationGenotypes:
                file = open(parameters.OUTPUT_GENOTYPE_TREE, 'a')
                file.write('\n\n --------- NEXT GENERATION : [ ' + str(g) + ' ] ----------- \n\n')
                file.close()
                for indi in list(offspring):
                    root = gp.compile(indi, pset)
                    printGenotype(indi, root)

            # The population is entirely replaced by the offspring
            population[:] = offspring

    # ------------------------------  PLOT THE TRAIN AND TEST ACCURACY ---------------------------------------------
    # Data for graph
    geneArr = range(1, NGEN + 1)
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
    if not node.isFinal():
        if hasattr(node, 'left'):
            printTree(node.left, space + 1)
        if hasattr(node, 'right'):
            printTree(node.right, space + 1)
    return


def printMutationChange(origin, change):
    file = open(parameters.OUTPUT_MUTATION_CHANGE, 'a')
    file.write('\n\n' + str(origin) + '\n' + str(change))
    file.close()


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


# TODO: integrate preprocessor for #ifdef DEBUG python alternative
if parameters.DEBUG:
    print("DEBUG ACTIVE")
    individual = "PAR(DOUB(SEQ(DOUB(RELU(PAR(PAR(PAR(HALF(SOFTSIGN(ELU(DOUB(RELU(RELU('END')))))), 'END'), 'END'), RELU('END')))), 'END')), RELU(RELU('END')))"
    # SEQ(PAR(PAR(PAR(PAR(PAR(TANH('END'), 'END'), DOUB(DOUB('END'))), 'END'), 'END'), DOUB('END')), TANH('END'))
    # PAR(DOUB(SEQ(DOUB(RELU(PAR(PAR(PAR(HALF(SOFTSIGN(ELU(DOUB(RELU(RELU('END')))))), 'END'), 'END'), RELU('END')))), 'END')), RELU(RELU('END')))
    accs = debugIndividual(individual)
    print(accs)
else:
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
