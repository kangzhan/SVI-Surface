
import numpy as np
from numpy.linalg import norm
from gaft import GAEngine
from gaft.components import BinaryIndividual
from gaft.components import Population
from gaft.operators import TournamentSelection
from gaft.operators import UniformCrossover
from gaft.operators import FlipBitMutation
from svi_jumpwing import svi_jumpwing
# # Analysis plugin base class.
from gaft.plugin_interfaces.analysis import OnTheFlyAnalysis

# Built-in best fitness analysis.
from gaft.analysis.fitness_store import FitnessStore
from svi_raw import svi_raw

class optimize_ga:
    def __init__(self ,k ,total_implied_variance ,slice_before ,slice_after ,tau):
        self.k =k
        self.total_implied_variance =total_implied_variance
        self.slice_before =slice_before
        self.slice_after = slice_after
        self.tau = tau

        # Define population.
        indv_template = BinaryIndividual(ranges=[(1e-5, 20),(1e-5, 20),(1e-5, 20)], eps=0.001)
        self.population = Population(indv_template=indv_template, size=30).init()

        # Create genetic operators.
        selection = TournamentSelection()
        crossover = UniformCrossover(pc=0.8, pe=0.5)
        mutation = FlipBitMutation(pm=0.1)

        # Create genetic algorithm engine.
        self.engine = GAEngine(population=self.population, selection=selection,
                               crossover=crossover, mutation=mutation,
                               analysis=[FitnessStore])

        # Define fitness function.
        @self.engine.fitness_register
        @self.engine.minimize
        def fitness(indv):
            a, b, m, rho, sigma = indv.solution

            model_total_implied_variance=svi_raw(self.k,np.array([a, b, m, rho, sigma]),self.tau)
            value = norm(self.total_implied_variance - model_total_implied_variance,ord=2)

            # if bool(len(self.slice_before)) and np.array(model_total_implied_variance < self.slice_before).any():
            #     value +=(np.count_nonzero(~np.array(model_total_implied_variance < self.slice_before))*100)
            #     # value = 1e6
            #
            # if bool(len(self.slice_after)) and np.array(model_total_implied_variance > self.slice_after).any():
            #     value += float(np.count_nonzero(~np.array(model_total_implied_variance > self.slice_after)) * 100)
            #     # value = 1e6
            # if np.isnan(value):
            #     value = 1e6

            value = float(value)
            return value


    # Define on-the-fly analysis.
    #     @self.engine.analysis_register
    #     class ConsoleOutputAnalysis(OnTheFlyAnalysis):
    #         interval = 1
    #         master_only = True
    #
    #         def register_step(self, g, population, engine):
    #             best_indv = population.best_indv(engine.fitness)
    #             msg = 'Generation: {}, best fitness: {:.3f}'.format(g, engine.ori_fmax)
    #             self.logger.info(msg)
    #
    #         def finalize(self, population, engine):
    #             best_indv = population.best_indv(engine.fitness)
    #             x = best_indv.solution
    #             y = engine.ori_fmax
    #             msg = 'Optimal solution: ({}, {})'.format(x, y)
    #             self.logger.info(msg)
    def optimize(self):
        self.engine.run(ng=500)
        return self.population.best_indv(self.engine.fitness).solution

# if '__main__' == __name__:
#     wzh=optimize_ga()
