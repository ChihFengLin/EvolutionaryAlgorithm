
Instructions:

1.) put the training data, testing data, pyeeg module and the codes under the same folder.
	The files below are our training data and testing data. 
	feature_training_data.txt
	feature_testing_data.txt
	label_training_data.txt
	label_testing_data.txt

2.) Install libsvm, cmaes
	>>sudo apt-get install python-libsvm
	>>pip install cma
	(if no pip command)
	>>sudo apt-get install python-pip

3.) Runnning CMAES:
	python cmaes_test.py offspring_size number_of_iterations
	E.g.
	>> python cmaes_test.py 30 500

	Output: 
	a.) 
	It generates a output.csv file which contains six columns showing
	[selected features of the best result, 
	number of iterations, 
	offspring size, 
	seed number, 
	number of features selected of the best result, 
	the best result]

	b.)
	On the screen, it shows up:
	>> The individual of the best result
	>> The selected features
	>> The Seed number
	>> The Best fitness value
	E.g.
	[ -1.08750216 -10.50481158  -0.56485319  -1.89558503 -45.59674307
	 -31.66687185 -46.05201992   0.88601661 -54.34816065 -53.65696618
	 -29.14804291 -35.06699041  -1.04702176  21.34500335  56.92980645
	 -10.2490337   26.64695052 -53.65050773 -28.82455683 -33.4205589
	 -56.01865848  -6.87574498 -54.69961774 -12.84340603 -41.31918365
	  -2.86240853 -16.53034594 -51.99002967  32.9805906  -62.46560763
	 -70.37702029 -29.39202297  -3.26724788  43.23193756 -14.48384613
	 -18.61797189 -27.70714941]
	[14, 33, 28, 16, 13]
	Seed number = 5016.000000
	best solutions fitness = 90.520134

4) Running Evolutionary Strategy:
	a) Check parameters
		fd1: feature training data
		fd2: label training data
		fd3: feature testing data
		fd4: label testing data

		run_time: it can make algorithm implement multiple runs
		generations: set up generation number for each run

	b) Type command line
	   >> python Evolutionary_Strategy.py

	c) Output
		test_result_0.txt: statistics of feature selected times
		best_result_0.txt: best fitness value for each generation
		mean_result_0.txt: mean fitness value for each generation
		std_result_0.txt: std of fitness value for each generation

5) Runnning GA:
	>> python GA.py

	Output: 
	a.) 
	On the screen, it shows as:
	>> The result of the best individual (feature bit-string)
	>> The best fitness for every generation
	>> The average fitness for every generation
        >> The best fitness of all the individuals

