

from IOHexperimenter import IOH_function, IOH_logger, IOHexperimenter
import numpy as np
import math
import sys

from numpy.core.numeric import ones_like


class EvolutionStrategy():
    budget_base = 10000

    def __init__(self, hyperparameters):
        # Population size
        # Selection type: 1: .
        # Mutation type:
        # Recombination type:

        self.population_size = int(hyperparameters[0])

        self.selection_type = int(hyperparameters[1])
        self.mutation_type = int(hyperparameters[2])
        self.recombination_type = int(hyperparameters[3])

        self.one_sig = 1

    def optimize(self, problem):
        self.t0 = 1 / math.sqrt(problem.number_of_variables)

        # Initiliaze required variables
        n = problem.number_of_variables

        fopt = -sys.maxsize-1
        x_prime = None

        # Generate initial population
        population = np.random.uniform(
            low=-5, high=5, size=(self.population_size, n))
        # Add the one sigmas
        sigmas = np.ones(len(population))
        population = np.append(population, np.array([sigmas]).T, axis=1)

        performance = self.eval_population(population, problem)

        # !! final_target_hit returns True if the optimum has been found.
        # !! evaluations returns the number of function evaluations has been done on the problem.

        while not problem.final_target_hit and problem.evaluations < self.budget_base * n:
            # Recombination
            recombined = self.recombine(population, 5)

            # Mutation do some crazy shit on the offspring
            mutated = self.mutate(recombined)

            # Evaluation
            evaluated = self.eval_population(mutated, problem)

            # Selection
            population, performance = self.selection(
                population, mutated, performance, evaluated)

            x, f = self.find_best(population, performance)
            if f >= fopt:
                x_prime = x
                fopt = f

        return x_prime, fopt

    def recombine(self, population, n_offspring):
        if self.recombination_type == 0:
            return self.global_intermediate_recombination(population, n_offspring)
        if self.recombination_type == 1:
            return self.intermediate_recombination(population, n_offspring)
        if self.recombination_type == 2:
            return self.discrete_recombination(population, n_offspring)
        if self.recombination_type == 3:
            return self.global_discrete_recombination(population, n_offspring)

    def mutate(self, population):
        # T0 = 1 / sqrt(n) with n = dimension?
        if self.mutation_type == 0:
            return self.one_sigma(population)
        if self.mutation_type == 1:
            return self.individual_sigma(population)

    def selection(self, parents, offspring, p_eval, o_eval):
        if self.selection_type == 0:  # (u, l)-selection
            return self.select_tournament(offspring, o_eval)
        if self.selection_type == 1:  # (u + l)-selection
            return self.select_tournament(np.append(parents, offspring, axis=0), np.append(p_eval, o_eval, axis=0))

    def one_sigma(self, population):
        # Update one_sigma for entire population
        self.one_sig = self.one_sig * math.exp(np.random.normal(0, self.t0))

        population += np.random.normal(0, self.one_sig, population.shape)
        return population

    def individual_sigma(self, population):
        population[:, -1] = population[:, -1] * math.exp(np.random.normal(0, self.t0))

        population[:, :-1] += np.random.normal(0, population[:, -1], (population.shape[0], population.shape[1] - 1))
        return population

    def intermediate_recombination(self, population, n_offspring):
        offsprings = []

        # Repeat n_offspring times.
        for _ in range(n_offspring):
            # Sample random individuals
            idxs = np.random.randint(0, len(population), 2)
            individuals = population[idxs]
            offsprings.append(np.mean(individuals, axis=0))
        return np.array(offsprings)

    def discrete_recombination(self, population, n_offspring):
        offsprings = []

        # Repeat n_offspring times.
        for _ in range(n_offspring):
            # Sample random individuals
            idxs = np.random.randint(0, len(population), 2)
            individuals = population[idxs]

            # recombine individuals
            parent_index = np.random.randint(low=0, high=len(idxs), size=len(individuals[0]))

            offspring = [individuals[p_i, i] for i, p_i in enumerate(parent_index)]
            offsprings.append(offspring)

        return np.array(offsprings)

    def global_discrete_recombination(self, population, n_offspring):
        offsprings = []

        # Repeat n_offspring times.
        for _ in range(n_offspring):
            # recombine individuals
            parent_index = np.random.randint(low=0, high=population.shape[0], size=population.shape[1])

            offspring = [population[p_i, i] for i, p_i in enumerate(parent_index)]
            offsprings.append(offspring)

        return np.array(offsprings)

    def global_intermediate_recombination(self, population, n_offspring):
        # Average of all of the parents
        # Collapses the whole population to single individual
        # Mutaiton operation does drastic modifaction.
        average_guy = np.mean(population, axis=0)
        return np.array([average_guy] * n_offspring)

    def select_tournament(self, population, performance):
        tournament = np.array([(x, y) for x, y in zip(population, performance)])

        np.random.shuffle(tournament)

        winners = []
        performances = []
        for _ in range(self.population_size):
            a = tournament[np.random.randint(0, high=len(population))]
            b = tournament[np.random.randint(0, high=len(population))]

            winning = [a, b][np.argmax([a[1], b[1]])]

            winners.append(winning[0])
            performances.append(winning[1])

        return np.array(winners), np.array(performances)

    def eval_population(self, population, problem):
        return np.array([problem(x[:-1]) for x in population])

    def find_best(self, population, performance):
        idx = np.argmax(performance)
        score = performance[idx]
        x_prime = population[idx]
        return x_prime, score


##################################################################################################
# Population size
# Selection type
#   0: (u, l)
#   1: (u + l)
# Mutation types
#   0: one_sigma
#   1: individual_sigma
# Recombination types
#   0: global_intermediate_recombination
#   1: intermediate_recombination
#   2: discrete_recombination
#   3: global_discrete_recombination

def aarnoutse_witte_ES(problem):
    es = EvolutionStrategy([10, 0, 0, 3])
    return es.optimize(problem)


# if __name__ == '__main__':

#     # Declarations of Ids, instances, and dimensions that the problems to be tested.
#     problem_id = range(1, 2)
#     instance_id = range(1, 26)
#     dimension = [2, 5, 20]

#     # Declariation of IOHprofiler_csv_logger.
#     # 'result' is the name of output folder.
#     # 'studentname1_studentname2' represents algorithm name and algorithm info, which will be caption of the algorithm in IOHanalyzer.
#     logger = IOH_logger("./", "result", "aarnoutse_witte", "aarnoutse_witte")

#     for p_id in problem_id:
#         for d in dimension:
#             print(f' problem: {p_id}, dim: {d}')
#             for i_id in instance_id:
#                 # Getting the problem with corresponding id,dimension, and instance.
#                 f = IOH_function(p_id, d, i_id, suite="BBOB")
#                 f.add_logger(logger)
#                 xopt, fopt = aarnoutse_witte_ES(f)

#     logger.clear_logger()


# Population size
# Selection type
#   0: (u, l)
#   1: (u + l)
# Mutation types
#   0: one_sigma
#   1: individual_sigma
# Recombination types
#   0: global_intermediate_recombination
#   1: intermediate_recombination
#   2: discrete_recombination
#   3: global_discrete_recombination


def experiment():
    problem_id = range(1, 2)
    instance_id = range(1, 5)
    dimension = [2, 5, 20]

    configurations_space = [[10, 0, 0, 0], [10, 1, 0, 0]]
    logger = IOH_logger("./", "result", f"aarnoutse_witte", f"aarnoutse_witte")
    for index, configuration in enumerate(configurations_space):
        for p_id in problem_id:
            for d in dimension:
                print(f' problem: {p_id}, dim: {d}')
                for i_id in instance_id:
                    # Getting the problem with corresponding id,dimension, and instance.
                    f = IOH_function(p_id, d, i_id, suite="BBOB")
                    f.add_logger(logger)
                    es = EvolutionStrategy(configuration)
                    xopt, fopt = es.optimize(f)
    logger.clear_logger()


experiment()
