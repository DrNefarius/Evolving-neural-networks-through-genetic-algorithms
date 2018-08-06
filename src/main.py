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


def evolve():
    """Loops over all generations, evolves and evaluates them."""

    print(str(constants.NGEN) + ' Generationen mit ' + str(
        constants.POPS) + ' Netzen pro Generation.')

    # global counter for network printing
    global noOfNet
    noOfNet = 0

    pset = gp.PrimitiveSet("main", 0)

    # getting all operators and adding them as primitves to the set
    primitives = OperatorLib().get_operators()
    for primitive in primitives:
        pset.addPrimitive(primitive.func, primitive.arity, primitive.name)

    pset.addTerminal('END')

    # defining the problem and Individual
    creator.create("FitnessMin", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin, pset=pset)

    # Initialization of the DEAP toolbox
    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=1, max_=1)
    toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr_init)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # evaluation of one individual
    def evaluate(individual):
        """Turns genotype to phenotype, creates and evaluates the Keras model, returns the models test accuracy"""
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
        individual.test_acc = network.test_acc
        individual.train_acc = network.train_acc
        return network.test_acc,

    # further initialization of the toolbox
    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("expr_mut", gp.genFull, pset=pset, min_=1, max_=3)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    # decorate mate and mutate to prevent bloating
    toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=constants.BLOAT_LIMIT))
    toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=constants.BLOAT_LIMIT))

    random.seed(42)  # the meaning of life
    population = toolbox.population(n=constants.POPS)  # create the initial population

    def log_pop(pop, c_index):
        """Logs one population."""
        pop.sort(key=attrgetter('fitness'), reverse=True)
        last_index = len(pop) - 1
        factor = c_index / (constants.NGEN * 1.0)
        completion_string = str(round(factor * 100, 2)) + "% abgeschlossen.\n"
        best_string = 'Bestes Individuum erreicht ' + str(pop[0].test_acc) + '% Genauigkeit.'
        print(completion_string + best_string)
        worst.append(pop[last_index].test_acc)
        best.append(pop[0].test_acc)
        train_acc.append(pop[0].train_acc)
        test_acc.append(pop[0].test_acc)
        total_score = 0
        for individual in pop:
            total_score += individual.test_acc
        average.append(total_score / constants.POPS)

    # Data for graph
    worst = []
    average = []
    best = []
    train_acc = []
    test_acc = []

    comp_index = 1  # completion index
    for gen in range(constants.NGEN):
        if gen != constants.NGEN:  # prevents additional loop
            start = time.time()

            # select the offspring and clone them
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
            for i in range(len(offspring)):
                for j in range(i + 1, len(offspring)):
                    if str(offspring[i]) == str(offspring[j]):
                        toolbox.mutate(offspring[j])
                        del offspring[j].fitness.values
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
            end = time.time()
            print(str(end - start) + ' Sekunden für Generation ' + str(comp_index) + ' benötigt.')
            comp_index += 1

    # create graphs
    ngen_range = range(1, constants.NGEN + 1)
    plt.plot(ngen_range, train_acc, 'bo-')
    plt.plot(ngen_range, test_acc, 'go-')
    b_patch = mpatches.Patch(color='blue', label='TRAIN')
    g_patch = mpatches.Patch(color='green', label='TEST')
    plt.legend(handles=[b_patch, g_patch])
    plt.ylabel('ACCURACY')
    plt.xlabel('GENERATION NUMBER')
    plt.savefig(constants.KERASGRAPH_PATH)
    plt.clf()
    plt.cla()
    plt.close()
    plt.plot(ngen_range, worst, 'ro-')
    plt.plot(ngen_range, average, 'bo-')
    plt.plot(ngen_range, best, 'go-')
    red_patch = mpatches.Patch(color='red', label='Worst')
    blue_patch = mpatches.Patch(color='blue', label='Average')
    green_patch = mpatches.Patch(color='green', label='Best')
    plt.legend(handles=[red_patch, blue_patch, green_patch])
    plt.ylabel('ACCURACY')
    plt.xlabel('GENERATION NUMBER')
    plt.savefig(constants.GENGRAPH_PATH)

    return population


def debug_individual(individual):
    """Evaluates one specific individual and returns the accuracies. individual must be a string."""
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
    """Starts the evolution and logs the time."""
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
    """Defines the to be tested individual and presents the calculated accuracies."""
    print("DEBUG ACTIVE")
    individual = "PAR(DOUB(SEQ(DOUB(RELU(PAR(PAR(PAR(HALF(SOFTSIGN(ELU(DOUB(RELU(RELU('END')))))), 'END'), 'END'), RELU('END')))), 'END')), RELU(RELU('END')))"
    # SEQ(PAR(PAR(PAR(PAR(PAR(TANH('END'), 'END'), DOUB(DOUB('END'))), 'END'), 'END'), DOUB('END')), TANH('END'))
    # PAR(DOUB(SEQ(DOUB(RELU(PAR(PAR(PAR(HALF(SOFTSIGN(ELU(DOUB(RELU(RELU('END')))))), 'END'), 'END'), RELU('END')))), 'END')), RELU(RELU('END')))
    accs = debug_individual(individual)
    print(accs)


if __name__ == "__main__":
    """The program starts here."""
    # TODO: integrate preprocessor
    if constants.DEBUG:
        debug()
    else:
        main()
