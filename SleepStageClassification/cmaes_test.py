import cma
import math
import random
import sys
import operator
import os
import csv
from svmutil import *

Train_data=[]
Train_label=[]
Test_data=[]
Test_label=[]

def object_function(x):
    global Train_data
    global Train_label
    global Test_data
    global Test_label

    sel_index = sel_var_length(x)
    sel_train_data = pre_data(sel_index,Train_data)
    sel_test_data = pre_data(sel_index,Test_data)

    m = svm_train(Train_label, sel_train_data, '-h 0')
    
    p_label, p_acc, p_val = svm_predict(Test_label, sel_test_data, m)
    ACC, MSE, SCC = evaluations(Test_label, p_label)
    
    return (100-ACC)

def pre_data(sel_index,data):
  
    sel_data =[]
    tmp=[]

    for d in data:
        for i in sel_index:
            tmp.append(d[i])
        sel_data.append(tmp)
        tmp=[]

    return sel_data

def sel_var_length(x):
    threshold = 0
    sel_index = []
    index, value =  zip(*sorted(enumerate(x), key=operator.itemgetter(1), reverse=True))
    for k in range(len(value)):
        if value[k] < 1 :
            break
        sel_index.append(index[k])

    return sel_index

def sel_max_weighted(x):
    index, value =  zip(*sorted(enumerate(x), key=operator.itemgetter(1)))
    sel_index = index[27:]

    return sel_index

def weighted(x):

    #weighted_x = [round(10*round(k/sum(x),2)) for k in x]
    weighted_x = [ 0 if k < 0  else round(k) for k in x ]
    diff = 10 - sum(weighted_x)
    
    while sum(weighted_x) != 10:
        k = random.randint(0,36)
        if diff > 0:
                weighted_x[k] = weighted_x[k]+1
                diff = diff-1
        elif weighted_x[k] > 0:
                weighted_x[k] = weighted_x[k]-1
                diff = diff+1
    
    sel_index = []
    for i, d in enumerate(weighted_x):
        while d != 0:
            sel_index.append(i)
            d = d -1

    return sel_index

def read_file(filename,flag):
    tmp = []
    data = []
    f = open(filename,'r')
    for line in f:
        if not line.strip():
            continue
        if flag == 0:
            newline = line.split(',')
            tmp = [float(item) for item in newline]
            data.append(tmp)
            tmp=[]
        else:
            data.append(int(line))
    f.close()     

    return data


if __name__ == '__main__':
    

    Train_data = read_file('feature_training_data.txt',0)
    Test_data = read_file('feature_testing_data.txt',0)
    Train_label = read_file('label_training_data.txt',1)
    Test_label = read_file('label_testing_data.txt',1)
    seed = random.randint(1, 10000)
    es = cma.CMAEvolutionStrategy([0.5] * 37, 1 , {'CMA_mu': int(sys.argv[1]),'popsize': 100,'seed':seed })#
    es.optimize(object_function, iterations = int(sys.argv[2]))
    res = es.result()
    #cma.pprint(res)
    cma.plot()
    cma.show()
    cma.savefig('fig_p'+sys.argv[1]+'_s1_i'+sys.argv[2]+'.png')
    #cma.closefig()
    os.system('cls')
    sel = sel_var_length(res[0])
    print res[0] 
    print sel
    print ('Seed number = %f' % (seed)) 
    print('best solutions fitness = %f' % (100-res[1])) 

    line = ','.join(str(e+1) for e in sel) #+'\n'
    with open('output.csv','a') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([line,sys.argv[2],sys.argv[1],str(seed),str(len(sel)),str(round(100-res[1],4))])

    #f = open('output.txt','a')
    #line = ' '.join(str(round(e,2)) for e in res[0]) + '\n'
    #f.write('ind_wei: '+line)
    
    #f.write('fea_sel: '+line)
    #f.write('iter: '+sys.argv[2]+', ' +'pop: '+ sys.argv[1]+', ')
    #f.write('seed: '+str(seed)+', ' +'num_fea_sel: '+ str(len(sel))+', ')
    #f.write('ACC: '+str(round(100-res[1],4))+'\n')
    #f.close()
    
