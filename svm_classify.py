#!/usr/bin/env python3

import sys 
import numpy as np
import scipy
from scipy.spatial import distance

#svm_classify.sh test_data model_file sys_output
test_data = open(sys.argv[1], 'r')
model_file = open(sys.argv[2], 'r')
sys_output = open(sys.argv[3], 'w')

#kernel functions
def linear(x, z):
	return np.inner(x, z)

def polynomial(x, z, gamma, coef0, degree):
	return ((gamma * np.inner(x, z)) + coef0)**degree

def radial_basis_function(x, z, gamma):
	return np.exp(-gamma * (scipy.spatial.distance.cdist(x, z)**2))

def sigmoid(x, z, gamma, coef0):
	return np.tanh((gamma * np.inner(x, z)) + coef0)

#classify intasnces in test data
def classify(vec_dict, test_dict, weights, unique_feats, mapping, true_labels, kernel_type, degree, gamma, coef0, rho):
	#calculate f(x) for all instances
	#create matrices of features in each instance to use to calculate f(x)
	model_matrix = np.zeros((len(vec_dict), len(unique_feats))) #initialize to 0
	test_matrix = np.zeros((len(test_dict), len(unique_feats))) #initialize to 0

	#populate matrices
	for ind, vector in vec_dict.items():
		for feat in vector:
			model_matrix[ind][mapping[feat]] = 1

	for ind, vector in test_dict.items():
		for feat in vector:
			feat_ind = mapping.get(feat)
			if feat_ind:
				test_matrix[ind][feat_ind] = 1

	#caluculate kernel matrix using kernel functions based on given kernel type
	if kernel_type == 'linear':
		kernel_matrix = linear(model_matrix, test_matrix)
	if kernel_type == 'polynomial':
		kernel_matrix = polynomial(model_matrix, test_matrix, gamma, coef0, degree)
	if kernel_type == 'rbf':
		kernel_matrix = radial_basis_function(model_matrix, test_matrix, gamma)
	if kernel_type == 'sigmoid':
		kernel_matrix = sigmoid(model_matrix, test_matrix, gamma, coef0)

	weights_arr = np.array(weights)

	#multiply kernel values by weights
	kernel_times_weights = kernel_matrix * weights_arr.reshape((weights_arr.size, 1))  
	
	#sum all kernel values (after multiplying by weights)
	kernel_sums = np.sum(kernel_times_weights, axis = 0)

	#subtract each by rho to get f(x) for each instance
	f_x = kernel_sums - rho

	#use f(x) to classify each instance
	total_count = len(test_dict) #total number of instances; to calculate accuracy
	correct_count = 0 #count number of correctly classified instances
	for i in range(len(f_x)): #iterate through f(x) values
		true_label = true_labels[i] #get true label for this instance
		if f_x[i] >= 0: 
			sys_label = 0
		else:
			sys_label = 1

		#print output
		sys_output.write(str(true_label) + ' ' + str(sys_label) + ' ' + str(f_x[i]) + '\n')
		if true_label == sys_label: #instance correctly classified
			correct_count += 1 

	#print accuracy
	accuracy = (correct_count/total_count) * 100
	print('Accuracy = ' + str(accuracy) + '% (' + str(correct_count) + '/' + str(total_count) + ')')

#read in and store model and test data files
def read_files():
	#read in model file
	#store parameters
	kernel_type = None
	degree = 0
	gamma = 0
	coef0 = 0
	rho = 0

	vec_dict = {} #store instances -> {index of instance: list of features}
	weights = [] #store weights for each instance
	unique_feats = set() #store set of all unique features that occur in model file

	header = True
	index = 0

	for line in model_file:
		line = line.strip()
		tokens = line.split(' ')
		
		if header: #get parameters
			if tokens[0] == 'kernel_type':
				kernel_type = tokens[-1]
			if tokens[0] == 'degree':
				degree = int(tokens[-1])
			if tokens[0] == 'gamma':
				gamma = float(tokens[-1])
			if tokens[0] == 'coef0':
				coef0 = float(tokens[-1])
			if tokens[0] == 'rho':
				rho = float(tokens[-1])
			if tokens[0] == 'SV':
				header = False
		else: #store instances
			weights.append(float(tokens[0])) #store weight for this instance
			feats = []
			for t in tokens[1:]: #iterate through features
				feat = int(t.split(':')[0])
				feats.append(feat) #add all features to list
				unique_feats.add(feat) #store in set
			vec_dict[index] = feats #store instance in dictionary
			index += 1

	unique_feats = sorted(list(unique_feats))
	unique_feats_mapping = {}
	for ind, feat in enumerate(unique_feats):
		unique_feats_mapping[feat] = ind

	#read in test file
	test_dict = {} #store instances -> {index of instance: list of features}
	true_labels = [] #store labels for each instance
	index = 0

	for line in test_data:
		line = line.strip()
		tokens = line.split(' ')
		true_labels.append(int(tokens[0])) #store label for this instance
		feats = []
		for t in tokens[1:]: #iterate through features
			feat = int(t.split(':')[0])
			feats.append(feat) #add all features to list
		test_dict[index] = feats #store instance in dictionary
		index += 1

	classify(vec_dict, test_dict, weights, unique_feats, unique_feats_mapping, true_labels, kernel_type, degree, gamma, coef0, rho)


read_files()












