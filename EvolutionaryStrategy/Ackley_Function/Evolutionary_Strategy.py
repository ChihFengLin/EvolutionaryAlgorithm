#
# Name: Chih-Feng Lin
# Andrew ID: chihfenl
#
# --Evolutionary Strategy for Ackley Function--
#
# <Modification Version - change the procedure of mutation and recombination>
# In this program, I found the trend of best fitness value for each evalutions is gradually dcreasing. 
# That meets the characteristic of evolutionary strategy but not decreasing quickly.
# If it runs 200000 evalutions, it will decrease to be close to zero.
#

import random
import math

random.seed(1234)

class AckleyFunction:

	def __init__(self, population_size, dimension):
		self.population_size = population_size
		self.dimension = dimension
		self.step_size = [0.1 for i in range(dimension)]
		self.tau = 0.05*(1/float(math.sqrt(2*math.sqrt(dimension))))
		self.tau1 = 0.15*(1/float(math.sqrt(2*dimension)))
		self.next_generation = []
		self.top_fitness = 0

	def start_population(self):
		self.population = []
		self.population_temp = [[random.uniform(-30, 30) for j in range(self.dimension)]
				              for i in range(self.population_size)]
		for item in self.population_temp:
			item.extend(self.step_size)
			self.population.append(item)

	def fitness_calculation(self, length, pool):
		total = 0
		firstSum = 0
		secondSum = 0
		individuals_fitness = []
		for i in range(length):
			for j in range(self.dimension):
				firstSum += pool[i][j]**2
				secondSum += math.cos(2*math.pi*pool[i][j])
			n = float(len(pool[i]))
			total = -20*math.exp(-0.2*math.sqrt(firstSum/n)) - math.exp(secondSum/n) + 20 + math.e
			individuals_fitness.append(total)
			total = 0
			firstSum = 0
			secondSum = 0
		return individuals_fitness

	def mutate_nstep(self):
		for i in range(len(self.next_generation)):
			N = (random.gauss(0, 1))
			for j in range(30):
				N1 = (random.gauss(0, 1))
				sigma1 = (self.next_generation[i][j+30])*(math.exp(self.tau1*N + self.tau*N1))
				if sigma1 <= 0.1:
					sigma1 = 0.1
				self.next_generation[i][j+30] = sigma1
				self.next_generation[i][j] = self.next_generation[i][j] + sigma1*N1

	def discrete_recombination(self):
		child_object = []
		index1 = random.randint(0, 29)
		index2 = random.randint(0, 29)
		while(index1 == index2):
			index2 = random.randint(0, 29)
		for i in range(30):
			father = self.population[index1] 
			mother = self.population[index2]
			child_object.append(random.choice([father[i], mother[i]]))
		return child_object

	def global_recombination(self):
		child_stepsize = []
		for j in range(30, 60):
			index1 = random.randint(0, 29)
			index2 = random.randint(0, 29)
			while(index1 == index2):
				index2 = random.randint(0, 29)
			child_stepsize.append((self.population[index1][j] + self.population[index2][j])/float(2))
		return child_stepsize


	def produce_offspring(self):
		children = []
		children.extend(self.discrete_recombination())
		children.extend(self.global_recombination())
		self.next_generation.append(children)

	def replacement(self, tuple_pair_list):
		temp_list = []
		for tuple_item in tuple_pair_list:
			temp_list.append(tuple_item[0])
			if len(temp_list) == 30:
				break
		self.population = temp_list
		self.next_generation = []

	def es_algo_cycle(self):
		target_offspring_num = 200
		while len(self.next_generation) < target_offspring_num:  #200 times
			self.produce_offspring()
		self.mutate_nstep()
		#survivor selection
		tuple_pair = []
		self.fitness_unselect = self.fitness_calculation(len(self.next_generation), self.next_generation)
		self.top_fitness = min(self.fitness_unselect)
		for i in range(len(self.fitness_unselect)):
			tuple_pair.append((self.next_generation[i], self.fitness_unselect[i]))
		sorted_pair = sorted(tuple_pair, key=lambda x:int(x[1]))
		self.replacement(sorted_pair)


#Main function
run_time = 1        # run_time = 1 for test, run_time = 100 for question required
for i in range(run_time):
	result1 = list()
	dimension = 30
	whole_size = 30
	
	ES = AckleyFunction(whole_size, dimension)
	ES.start_population()
	count = 0
	
	for i in range(1000):
		count = count + 1
		ES.es_algo_cycle()
		result1.append(ES.top_fitness)
		#print ES.top_fitness
		if ES.top_fitness == 0:
			print "Finish Time: " + str(count)
			break

	sorted_result1 = sorted(result1)
	print "Min Value:" + '\t' + str(sorted_result1[0])
	
