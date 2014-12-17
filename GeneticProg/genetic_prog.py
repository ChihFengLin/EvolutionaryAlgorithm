#
# Symbolic Regression Problem
# Name: Chih-Feng Lin   
# AndrewID: chihfenl
#
# To implement this code, you should have to install genetic libary "deap"
# Installation method can be referenced from http://deap.gel.ulaval.ca/doc/default/installation.html
# In addition, you also have to install module of "numpy" and "matplotlib" to present picture
# numpy: http://www.numpy.org
# matplotlib: http://matplotlib.org
# 
# Use command line to compile this file with "python genetic_prog.py" 
#


import operator
import math
import random
import numpy
import numpy as np
import matplotlib.pyplot as plt

from deap import algorithms
from deap import base
from deap import creator
from deap import tools
from deap import gp

# Define new functions
def safeDiv(left, right):
    try:
        return left / right
    except ZeroDivisionError:
        return 0

pset = gp.PrimitiveSet("MAIN", 1)
pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)
pset.addPrimitive(safeDiv, 2)
pset.addPrimitive(operator.neg, 1)
pset.addPrimitive(math.cos, 1)
pset.addPrimitive(math.sin, 1)
pset.addEphemeralConstant("rand101", lambda: random.randint(-1,1))
pset.renameArguments(ARG0='x')

creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMin)

toolbox = base.Toolbox()
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=2)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("compile", gp.compile, pset=pset)

def evalSymbReg(individual, points):
    # Transform the tree expression in a callable function
    func = toolbox.compile(expr=individual)
    # Evaluate the mean squared error between the expression
    # and the real function : x**4 + x**3 + x**2 + x
    sqerrors = ((func(x) - x**4 - x**3 - x**2 - x)**2 for x in points)
    return math.fsum(sqerrors) / len(points),

toolbox.register("evaluate", evalSymbReg, points=[x/10. for x in range(-10,10)])
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genFull, min_=0, max_=2)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

def main():
	random.seed(1234)
	gen_num = 180
	pop = toolbox.population(n=800)
	hof = tools.HallOfFame(1)
    
	stats_fit = tools.Statistics(lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(len)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", numpy.mean)
	mstats.register("std", numpy.std)
	mstats.register("min", numpy.min)
	mstats.register("max", numpy.max)
	
	pop, log = algorithms.eaSimple(pop, toolbox, 0.5, 0.1, gen_num, stats=mstats,
			halloffame=hof, verbose=False)

	return pop, log, hof, gen_num

if __name__ == "__main__":
	pop, log, hof, gen_num = main()
	time = np.linspace(0, gen_num, gen_num+1)
	performance = log.chapters['fitness'].select("min")
	size = log.chapters['size'].select('avg')


	fig, ax1 = plt.subplots()
	ax1.set_title('Symbolic Regression Example')
	ax1.plot(time, performance, "rD--")
	ax1.set_xlabel('Generations')
	ax1.set_ylabel('Performance(Mean Square Error)', color='r')
	for tl in ax1.get_yticklabels():
		tl.set_color('r')
	
	ax2 = ax1.twinx()
	ax2.plot(time, size, "bo")
	ax2.set_ylabel('Average Size', color='b')
	for tl in ax2.get_yticklabels():
		tl.set_color('b')
	plt.show()
	
