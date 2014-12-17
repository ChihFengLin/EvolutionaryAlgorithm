#
# Name: Chih-Feng Lin
# Andrew ID: chihfenl
#
# --MaxOne Problem for Eiben & Smith Textbook 3-7--
#

import random
import time

random.seed(1234)

class MaxOneProb:

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
		total = 0
		geno_fitness = []
		for i in range(self.population_size):
			for j in range(self.binary_string_size):
				total = total + self.population[i][j]
			geno_fitness.append(total)
			total = 0
		return geno_fitness

	def proportionate_selection(self):
		self.tuple_pair = list()	
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


#Main function
result1 = list()
result2 = list()
run_time = 1          #Q1: runtime = 1 Q2: runtime = 10
for i in range(run_time):
	tStart = time.time()
	binary_string_length = 25
	pc = 0.7
	pm = 1/float(binary_string_length)
	whole_size = 100
	
	genetic_algo = MaxOneProb(whole_size, binary_string_length, pm, pc)
	genetic_algo.start_population()
	count = 0
	
	if run_time == 1:
		for i in range(100):
			mean = sum(genetic_algo.fitness) / float(len(genetic_algo.fitness))
			print str(max(genetic_algo.fitness)) + '\t' + str(min(genetic_algo.fitness)) + '\t' + str(mean)
			result1.append(str(max(genetic_algo.fitness)) + '\t' + str(min(genetic_algo.fitness)) + '\t' + str(mean))
			count = count + 1
			if max(genetic_algo.fitness) == 25:
				print "Finish Time: " + str(count)
				break
	
			genetic_algo.proportionate_selection()
			genetic_algo.roulette_wheel()
			genetic_algo.genetic_algo_cycle()

	if run_time == 10:
		while(max(genetic_algo.fitness) != 25):
			mean = sum(genetic_algo.fitness) / float(len(genetic_algo.fitness))
			count = count + 1
			if max(genetic_algo.fitness) == 25:
				print "Finish Time: " + str(count)
				break
			
			genetic_algo.proportionate_selection()
			genetic_algo.roulette_wheel()
			genetic_algo.genetic_algo_cycle()
			 
		tStop = time.time()
		result2.append(str(tStop - tStart))


if run_time == 1:
	open('worst_best_mean.txt', 'w').write('%s' % '\n'.join(result1))

if run_time == 10:
	open('execution_time.txt', 'w').write('%s' % '\n'.join(result2))
	
