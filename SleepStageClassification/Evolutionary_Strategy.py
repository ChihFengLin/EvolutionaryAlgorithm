
# Name: Chih-Feng Lin
# Andrew ID: chihfenl
#
# --Evolutionary Strategy for Sleep Classification--
#

import sys
import random
import math
import statistics as stat
from svmutil import *

#random.seed(1234)

fd1 = open('/Users/chih-fenglin/Dropbox/CMU Course/Evolutionary Algorithm/Final Project/Data/feature_training_data.txt','r')
fd3 = open('/Users/chih-fenglin/Dropbox/CMU Course/Evolutionary Algorithm/Final Project/Data/feature_testing_data.txt','r')
fd2 = open('/Users/chih-fenglin/Dropbox/CMU Course/Evolutionary Algorithm/Final Project/Data/label_training_data.txt','r')
fd4 = open('/Users/chih-fenglin/Dropbox/CMU Course/Evolutionary Algorithm/Final Project/Data/label_testing_data.txt','r')

temp = []
feature_training_data = []
label_training_data = []
feature_testing_data = []
label_testing_data = []

line1 = fd1.readline()
line2 = fd2.readline()
line3 = fd3.readline()
line4 = fd4.readline()

while line1:
	line1 = line1.strip()
	line1 = line1.split(',')
	for item in line1:
		temp.append(float(item))
	feature_training_data.append(temp)
	temp = []
	line1 = fd1.readline()


while line2:
	line2 = line2.strip()
	label_training_data.append(int(line2))
	line2 = fd2.readline()


while line3:
	line3 = line3.strip()
	line3 = line3.split(',')
	for item in line3:
		temp.append(float(item))
	feature_testing_data.append(temp)
	temp = []
	line3 = fd3.readline()


while line4:
	line4 = line4.strip()
	label_testing_data.append(int(line4))
	line4 = fd4.readline()

#print label_testing_data
#sys.exit("STOP")

fd1.close()
fd2.close()
fd3.close()
fd4.close()

class EvolutionaryStrategy:

	def __init__(self, population_size, dimension, feature_training_data, label_training_data, feature_testing_data, label_testing_data):
		self.population_size = population_size
		self.dimension = dimension
		self.step_size = [0.1 for i in range(dimension)]
		self.tau = 0.05*(1/float(math.sqrt(2*math.sqrt(dimension))))
		self.tau1 = 0.15*(1/float(math.sqrt(2*dimension)))
		self.next_generation = []
		self.top_fitness = []
		self.avg_fitness = []
		self.std_fitness = []
		self.feature_training_data = feature_training_data
		self.label_training_data = label_training_data
		self.feature_testing_data = feature_testing_data
		self.label_testing_data = label_testing_data


	def start_population(self):
		self.population = []
		self.population_temp = [[random.uniform(0,36) for j in range(self.dimension)]
				              for i in range(self.population_size)]
		for item in self.population_temp:
			item.extend(self.step_size)
			self.population.append(item)

	def fitness_calculation(self, length, pool):
		
		individuals_fitness = []
		
		for i in range(length):
			x_train = []
			y_train = []
			x_test = []
			y_test = []
			for count1 in range(len(self.feature_training_data)):
				temp1 = []
				for j in range(self.dimension):
					temp1.append(self.feature_training_data[count1][int(round(pool[i][j]))])

				y_train.append(self.label_training_data[count1])
				x_train.append(temp1)

			for count2 in range(len(self.feature_testing_data)):
				temp2 = []
				for j in range(self.dimension):
					temp2.append(self.feature_testing_data[count2][int(round(pool[i][j]))])
				
				y_test.append(self.label_testing_data[count2])
				x_test.append(temp2)

			m = svm_train(y_train, x_train, '-c 4')
			p_label, p_acc, p_val = svm_predict(y_test, x_test, m)
			ACC, MSE, SCC = evaluations(y_test, p_label)

			individuals_fitness.append(ACC)
		
		return individuals_fitness

	def mutate_nstep(self):
		for i in range(len(self.next_generation)):
			N = (random.gauss(0, 1))
			for j in range(10):
				N1 = (random.gauss(0, 1))
				sigma1 = (self.next_generation[i][j+10])*(math.exp(self.tau1*N + self.tau*N1))
				if sigma1 <= 0.1:
					sigma1 = 0.1
				self.next_generation[i][j+10] = sigma1
				self.next_generation[i][j] = self.next_generation[i][j] + sigma1*N1

	def discrete_recombination(self):
		child_object = []
		index1 = random.randint(0, 29)
		index2 = random.randint(0, 29)
		while(index1 == index2):
			index2 = random.randint(0, 29)
		for i in range(10):
			father = self.population[index1] 
			mother = self.population[index2]
			child_object.append(random.choice([father[i], mother[i]]))
		return child_object

	def global_recombination(self):
		child_stepsize = []
		for j in range(10, 20):
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
		temp_fit_list = []
		for tuple_item in tuple_pair_list:
			temp_list.append(tuple_item[0])
			temp_fit_list.append(tuple_item[1])
			if len(temp_list) == 30:
				break
		
		self.top_fitness.append(max(temp_fit_list))
		self.avg_fitness.append(stat.mean(temp_fit_list))
		self.std_fitness.append(stat.stdev(temp_fit_list))
		self.population = temp_list
		self.next_generation = []

	def es_algo_cycle(self):
		target_offspring_num = 100
		while len(self.next_generation) < target_offspring_num:  #200 times
			self.produce_offspring()
		self.mutate_nstep()
		#survivor selection
		tuple_pair = []
		self.fitness_unselect = self.fitness_calculation(len(self.next_generation), self.next_generation)
		#self.top_fitness = max(self.fitness_unselect)
		for i in range(len(self.fitness_unselect)):
			tuple_pair.append((self.next_generation[i], self.fitness_unselect[i]))
		sorted_pair = sorted(tuple_pair, key=lambda x:int(x[1]), reverse=True)
		self.replacement(sorted_pair)


#Main function
run_time = 1        # run_time = 1 for test, run_time = 100 for question required
k = 0
best_each_run = []
for i in range(run_time):
	result1 = list()
	dimension = 10
	whole_size = 30
	generations = 500

	ES = EvolutionaryStrategy(whole_size, dimension, feature_training_data, label_training_data, feature_testing_data, label_testing_data)
	ES.start_population()
	#count = 0
	
	for j in range(generations):
		#count = count + 1
		ES.es_algo_cycle()
		#if ES.top_fitness == 100:
		#	print "Finish Time: " + str(count)
		#	break

	#print ES.population

	counter_dict = { k:0 for k in range(37) }

	for i in range(len(ES.population)):
		for j in range(10):
			counter_dict[int(round(ES.population[i][j]))] += 1

	print counter_dict

	best_each_run.append(max(ES.top_fitness))

	filename1 = "test_result_%d.txt" % (k)
	filename2 = "best_result_%d.txt" % (k)
	filename3 = "mean_result_%d.txt" % (k)
	filename4 = "std_result_%d.txt" % (k)
	
	open(filename1, 'w').write('%s' % counter_dict)
	open(filename2, 'w').write('%s' % ES.top_fitness)
	open(filename3, 'w').write('%s' % ES.avg_fitness)
	open(filename4, 'w').write('%s' % ES.std_fitness)

	k = k + 1

open('best_each_run.txt', 'w').write('%s' % best_each_run)
