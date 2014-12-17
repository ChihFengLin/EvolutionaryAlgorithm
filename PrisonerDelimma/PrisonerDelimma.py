# 
# Name: Chih-Feng Lin
# Andrew ID: chihfenl
#
# Prisoner's Dilemma
# - Use binary representation: cooperate = 1, defect = 0
# - Player's strategy squence is GA's individual
#
# In order to run this code, please install the following required libaries via "sudo pip install"
# statistics, numpy, matplotlib
#
# In the command line, type "python PrisonerDelimma.py" to implement this code
# 
#

import sys
import numpy as np
import random
import statistics as sat
import matplotlib.pyplot as plt

class PrisonerDilemma:

	def __init__(self, population_size, binary_string_size, mutate_rate, crossover_rate):
		self.population_size = population_size
		self.binary_string_size = binary_string_size
		self.mutate_rate = mutate_rate
		self.crossover_rate = crossover_rate
		self.next_generation = []

	def start_population(self):
		self.population = [[random.choice([1, 0]) for j in range(self.binary_string_size)] for i in range(self.population_size)]
		self.fitness = self.fitness_calculation()

	def fitness_calculation(self):
		self.score_list = np.zeros(self.population_size)
		for i in range(self.population_size):
			strategy_p1 = self.population[i]
			for j in range(self.population_size):
				strategy_p2 = self.population[j]
				if (i == j):
					continue
				elif (i != j):
                    #Compete with each other players
					for k in range(self.binary_string_size):
						
						action_p1 = bool(strategy_p1[k])
						action_p2 = bool(strategy_p2[k])
						
						# CC = 3 point, CD = 0 point, DC = 5 point, DD = 1 point
						# Case1: cooperate and cooperate
						if (action_p1) and (action_p2):
							self.score_list[i] += 3
						# Case2: cooperate and defect
						if (action_p1) and (not action_p2):
							self.score_list[i] += 0
						# Case3: defect and cooperate
						if (not action_p1) and (action_p2):
							self.score_list[i] += 5
						# Case4: defect and defect
						if (not action_p1) and (not action_p2):
							self.score_list[i] += 1
		
		return self.score_list


	def proportionate_selection(self):
		self.tuple_pair = []	
		for i in range(self.population_size):
			individual_prob = self.fitness[i]/float(sum(self.fitness))
			self.tuple_pair.append((self.population[i], individual_prob))

	def roulette_wheel(self):
		temp = 0
		current_member = 0
		self.cul_prob_list = list()
		self.mating_pool = list()
		for tuple_item in self.tuple_pair:
			temp = temp + float(tuple_item[1])
			self.cul_prob_list.append(temp)

		while(current_member < self.population_size):
			r = random.random()
			i = 0
			while(self.cul_prob_list[i] < r):
				i = i + 1
			self.mating_pool.append(self.tuple_pair[i][0])
			current_member = current_member + 1     

	def mutate(self, geno):
		for i in range(len(geno)):
			if random.random() >= self.mutate_rate:
				continue
			else:
				geno[i] = not(geno[i])
		return geno

	def crossover(self, father, mother):
		if random.random() >= self.crossover_rate:
			return (father, mother)
		else:
			cross_point = random.randint(0, self.binary_string_size - 1)
			first_child = mother[:cross_point] + father[cross_point:]
			second_child = mother[cross_point:] + father[:cross_point]
			return (first_child, second_child)

	def produce_offspring(self):
		for i in range(0, self.population_size, 2):	
			father = self.mating_pool[i]
			mother = self.mating_pool[i+1]
			children = self.crossover(father, mother)
			self.next_generation = self.next_generation + [self.mutate(children[0])]
			self.next_generation = self.next_generation + [self.mutate(children[1])]

	def replacement(self):
		self.population = self.next_generation
		self.next_generation = []
		self.fitness = self.fitness_calculation()

	def genetic_algo_cycle(self):
		while len(self.next_generation) < self.population_size:
			self.produce_offspring()
		self.replacement()


#Main Function
num_of_runs = 1    # Here you can modify independent run time 
for i in range(num_of_runs):
	mean_list = []
	std_list = []
	best_fit_list = []

	# GA Related Parameters
	binary_string_length = 6
	pc = 0.75
	pm = 0.005
	whole_size = 50
	num_of_generation = 300
	
	# Start GA  
	genetic_algo = PrisonerDilemma(whole_size, binary_string_length, pm, pc)
	genetic_algo.start_population()
	
	for i in range(num_of_generation):
		mean_list.append(sum(genetic_algo.fitness) / float(len(genetic_algo.fitness)))
		std_list.append(sat.pstdev(genetic_algo.fitness))
		best_fit_list.append(max(genetic_algo.fitness))
		genetic_algo.proportionate_selection()
		genetic_algo.roulette_wheel()
		genetic_algo.genetic_algo_cycle()

plt.figure(0)
plt.plot(range(num_of_generation), mean_list)
plt.plot(range(num_of_generation), best_fit_list)
plt.plot(range(num_of_generation), std_list)
plt.title('Mean Fitness Value versus Generations')
plt.xlabel('Number_of_Generations')
plt.ylabel('Fitness Value')
plt.show()

