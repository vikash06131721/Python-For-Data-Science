
#Implementing knn from scratch for better understanding to image problems.
#libraries 
import numpy as np 
import csv
import random
import math
import operator
#handle data:
# Open the dataset and split into train and test sets 
def loaddataset(filename, split, trainingset=[], testset=[]):
    with open(filename,'rb') as csvfile:
        lines= csv.reader(csvfile)
        dataset= list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y]= float(dataset[x][y])
                print dataset[x][y]
            if random.random() < split :
                trainingset.append(dataset[x])
            else:
                testset.append(dataset[x])
#In order to make predictions we need to calc the similarity
#between any two given data instances.
def euclideanDistance(inst1, inst2, length):
    dist=0 
    for x in range(length):
        dist += pow((inst1[x]-inst2[x]),2)
    return math.sqrt(dist) 

#calculating the neighbors
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

#calculating the response 
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

#get accuracy 
def getAccuracy(testSet, predictions):
    correct= 0.0 
    for x in xrange(len(testSet)):
        if testSet[x][-1]==predictions[x]:
            correct +=1 
        else:
            pass
    return (correct/float(len(testSet))) * 100.0 

#putting all together 
def main():
    #prep data 
    trainSet = []
    testSet= []
    split= 0.67 
    loaddataset('../data/iris.data.txt',split,trainSet,testSet)
    print 'train set:' + repr(len(trainSet))
    print 'test set:' + repr(len(testSet))

    #generate predictions 
    pred= []
    k=3 
    for x in xrange(len(testSet)):
        neighbors= getNeighbors(trainSet, testSet[x],k)
        result= getResponse(neighbors)
        pred.append(result)
        print ('predicted='+repr(result) + ',actual='+repr(testSet[x][-1]))
    print getAccuracy(testSet, pred)
    
    # accuracy= getAccuracy(testSet, pred)
    # print('Accuracy'+repr(accuracy)+'%')
main()














