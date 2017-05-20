
import pandas as pd 
import numpy as np 
import cPickle
import tensorflow as tf
import glob as glob
import data_utils as dsutils 
import time 
import math
import operator

def euclideanDistance(inst1, inst2, length):
    dist=0 
    for x in range(length):
        dist += pow((inst1[x]-inst2[x]),2)
    return math.sqrt(dist) 

def getNeighbors(trainingSet, testInstance,k):
    distances=[]
    length = len(testInstance)-1 
    for x in xrange(len(trainingSet)):
        dist= euclideanDistance(testInstance,trainingSet[x], length)
        distances.append((trainingSet[x],dist))
    #sorts the distances list based on the (0,1) 1 represents
    #the tuple position.
    distances.sort(key=operator.itemgetter(1))
    neighbors=[]
    for x in xrange(k):
        neighbors.append(distances[x][0]) 
    return neighbors

def getResponse(neighbors):
    classVotes= {}
    for x in xrange(len(neighbors)):
        resp= neighbors[x][-1]
        if resp in classVotes:
            classVotes[resp] +=1
        else:
            classVotes[resp]= 1 
    sortedVotes= sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True) 
    return sortedVotes[0][0] 

def getAccuracy(testSet, predictions):
    correct= 0.0 
    for x in xrange(len(testSet)):
        if testSet[x]==predictions[x]:
            correct +=1 
        else:
            pass
    return (correct/float(len(testSet))) * 100.0


if __name__ == '__main__':
	start_time = time.time()
	Xtr, Ytr, Xte, Yte= dsutils.load_CIFAR10('../data/cifar-10-batches-py/')

	#flatten out all images 
	Xtr_rows = Xtr.reshape(Xtr.shape[0], 32 * 32 * 3) # Xtr_rows becomes 50000 x 3072 
	Xte_rows = Xte.reshape(Xte.shape[0], 32 * 32 * 3) # Xte_rows becomes 10000 x 3072 

	#Xtr_rows, Ytr will be used for training and Xte_rows, Yte will be used for testing 
	#lets consider 1000 images first. 
	Xtr_rows_1000= Xtr_rows[:10000]
	Ytr_1000= Ytr[:10000]
	#adding the class to each image rows.
	Xtr_rows_1000_class= []
	for x in xrange(len(Xtr_rows_1000)):
		Xtr_rows_1000_class.append(np.hstack((Xtr_rows_1000[x],Ytr_1000[x])))
	
	#consider the 20 elements as validation data:
	validation_Xtr_last_1000= Xtr_rows[45600:45650]
	validation_Ytr_last_1000= Ytr[45600:45650] 

	validation_Xtr_last_1000_class=[]
	for x in xrange(len(validation_Xtr_last_1000)):
		validation_Xtr_last_1000_class.append(np.hstack((validation_Xtr_last_1000[x],validation_Ytr_last_1000[x]))) 


	#main 
	#generate predictions
	pred= []
	k=3
	for x in xrange(len(validation_Xtr_last_1000_class)):
		neighbors= getNeighbors(Xtr_rows_1000_class, validation_Xtr_last_1000_class[x],5)
		result= getResponse(neighbors)
		pred.append(result)
		print ('predicted='+repr(result) + ',actual='+repr(validation_Xtr_last_1000_class[x][-1]))
	print getAccuracy(validation_Ytr_last_1000, pred) 

	
	




	



