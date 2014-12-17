import random
import math
import matplotlib.pyplot as plt

import numpy as np
import pylab as pl
#from svm import *
from svmutil import *

#set random seed as instructed
random.seed(2)
#random.seed(3)
def avg(x):
    return sum(x)/(float)(len(x))

def stdDeviation(a):
    m=avg(a)
    d=0
    for i in a:
        d+=(i-m)**2
    return math.sqrt(1.0*d/len(a))
    
#using fitness proportionate while selecting parents, implemented via roulette wheel
def roulette(population, fitness):
    totalfitness= float(sum(fitness))
    relfitness=[f/totalfitness for f in fitness]
    p= [sum(relfitness[:i+1]) for i in range(len(relfitness))]
    while 1:
        r=random.random() 
        for i in range(len(population)):
            if r <= p[i]:
                return population[i]
def getlistx(fd):
    line = fd.readline()
    temp=[]
    a=[]
    while line:
        line = line.strip()
        line = line.split(',')
        for item in line :
            temp.append(float(item))
        a.append(temp)
        temp=[]
        line = fd.readline()
    return a
    
def getlisty(fd):
    line = fd.readline()
    a=[]
    while line:
        line = line.strip()
        a.append(int(line))
        line = fd.readline()
    return a


#get revised x_train with selected features
def getx_train(pop):
    a=[]
    b=[]
    for t in range(len(x_train)):
        for j in range(37):
            if(pop[j]==1):
                b.append(x_train[t][j])
        a.append(b)
        b=[]
    return a

#get revised x_test with selected features
def getx_test(pop):
    a=[]
    b=[]
    for t in range(len(x_test)):
        for j in range(37):
            if(pop[j]==1):
                b.append(x_test[t][j])
        a.append(b)
        b=[]
    return a


class OneMax:
    #initialize the input values
    def __init__(self, length, popsize, mutprob, crossprob):
        self.length = length
        self.popsize = popsize
        self.mutprob = mutprob
        self.crossprob= crossprob
        self.nextpop = []

    #random the starting (initial) population and measure the fitness
    def startings(self):
        self.pop = [[random.choice([1, 0])
        for y in range(self.length)]
        for x in range(self.popsize)]
        self.fitness = self.measurefit()
    
    

    #measure the fitness 
    def measurefit(self):
        individuals_fitness = []
        for i in range(self.popsize):
            
            m = svm_train(y_train, getx_train(self.pop[i]),'-c 4')
            p_label, p_acc, p_val = svm_predict(y_test, getx_test(self.pop[i]), m)
            ACC, MSE, SCC = evaluations(y_test, p_label)
            individuals_fitness.append(ACC)
        return individuals_fitness
        #for x in range(self.popsize)]

    #do bit-flip mutation with given probability
    def mutate(self, gene):
        for i in range(len(gene)):
            if random.random() < self.mutprob:
                gene[i] = not(gene[i])
        return gene

    #do one-point recombination at random point with given probability
    def crossover(self, par1, par2):
        if random.random()<self.crossprob:
            splitp = random.randint(0, self.length - 1)
            #print "splitp is "+str(splitp)+", par1 length is "+str(len(par1))+", par2 length is "+str(len(par2))
            child1 = par2[splitp:] + par1[:splitp] 
            child2 = par1[splitp:] + par2[:splitp]
        else:
            child1 = par1
            child2 = par2
        return (child1, child2)

    #create the next generation
    def create(self):
        par1 = roulette(self.pop, self.fitness)
        par2 = roulette(self.pop, self.fitness)
        children = self.crossover(par1, par2)
        self.nextpop += [self.mutate(children[0])]
        self.nextpop += [self.mutate(children[1])]
        

    #using strict generational
    def replace(self):
        self.pop = self.nextpop
        #print self.pop
        self.nextpop = []
        self.fitness = self.measurefit()

    #generate
    def generate(self):
        while len(self.nextpop) < self.popsize:
            self.create()
        self.replace()


#Please shut down the popped-out plot windows first before you could get the print results
fd1=open('/home/ubuntu/feature_training_data.txt','r')
fd2=open('/home/ubuntu/label_training_data.txt','r')
fd3=open('/home/ubuntu/feature_testing_data.txt','r')
fd4=open('/home/ubuntu/label_testing_data.txt','r')

x_train=getlistx(fd1)
y_train=getlisty(fd2)
x_test=getlistx(fd3)
y_test=getlisty(fd4)
onemax1= OneMax(37, 30, 0.04, 0.7)
onemax1.startings()
t1=[]
y1m=[]
y1b=[]
y1w=[]
best_fitness=0
best_result=[0]*37
resulta=[]
resultb=[]
#50 generations
for i in range(500):
    v=stdDeviation(onemax1.fitness)
    #print "At generation "+ str(i)+ " the average fitness is "+ str(avg(onemax1.fitness))+ ", the best fitness is "+ str(max(onemax1.fitness))+ "the standard deviation is "+ str(v)
    #print onemax1.fitness
    if(max(onemax1.fitness)>best_fitness):
        best_fitness=max(onemax1.fitness)
        average_fitness=avg(onemax1.fitness)
        best_result=onemax1.pop[onemax1.fitness.index(best_fitness)]
    #t1.append(i)
    resultb.append(str(i)+"-"+str(best_fitness))
    resulta.append(str(i)+"-"+str(average_fitness))
    #y1m.append(average_fitness)
    #y1w.append(v) 
    onemax1.generate()
    
    

print best_result
print resultb
print resulta
print "best_fitness is "+str(best_fitness)


#plot the results
#plt.figure(1)
#plt.hold(True)
#plt.plot(t1,y1m,'r+-',t1,result,'k*:',t1,y1w,'b+--')
#plt.ylim(70,100)
#plt.xlabel('generation')
#plt.ylabel('fitness')


#plt.show()       
