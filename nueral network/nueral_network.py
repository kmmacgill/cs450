import random
import csv
from sklearn.utils import shuffle
from numpy.random import random_integers
from sklearn import preprocessing
import math

class DataLoader:
    def loadDataset(self, filename, splitter, trainingData, trainingTarget, testData, testTarget):
        # open it up and read it into a file
        f = open(filename)
        lines = csv.reader(f)
        dataSet = list(lines)
        f.close()
        # time to randomize the data...
        dataSet = shuffle(dataSet, random_state=random_integers(1, 1000000))

        # now divy up the file according to the file name
        for x in range(len(dataSet)):
            for y in range(len(dataSet[x]) - 1):  # -1 is all but last?
                dataSet[x][y] = float(dataSet[x][y])
            if random.random() < splitter:
                trainingData.append(dataSet[x][:-1])
                trainingTarget.append(dataSet[x][-1])
            else:
                testData.append(dataSet[x][:-1])
                testTarget.append(dataSet[x][-1])


class Neuron:
    """neuron class for networks"""
    bias = 1
    def __init__(self, inputNums):
        self.weight = []
        for i in range(0, inputNums):
            randWeight = random.random()
            self.weight.append(randWeight)
        self.weight.append(random.random()) #for bias node

    def neuronBoom(self, inputs):
        totalVal = 0
        for i in range(len(inputs)):
            totalVal += (inputs[i] * self.weight[i])
        totalVal += (self.bias * self.weight[-1])  # for bias node
        return 1/(1 + math.e**totalVal)

class networkOfNodes:

    def __init__(self):
        self.layers = []
    def addLayer(self, howManyNeurons, howManyInputs):
        neurons = []
        for i in range(0, howManyNeurons):
            neurons.append(Neuron(howManyInputs))
        self.layers.append(neurons)

    def predict(self, data):
        predictions = []
        for currentLayer in range(len(self.layers)):
            for row in range(len(data)):
                explodedNeurons = []
                for node in range(len(self.layers[currentLayer])):
                    neuron = self.layers[currentLayer][node]
                    explodedNeurons.append(neuron.neuronBoom(data[row]))
                predictions.append(explodedNeurons)
            data = predictions
        return predictions

def normalize(trainingData, testData):
    std_scale = preprocessing.StandardScaler().fit(trainingData)
    trainingData = std_scale.transform(trainingData)
    testData = std_scale.transform(testData)
    return trainingData, testData



def main(filename):
    dl = DataLoader()
    trainingData = []
    trainingTarget = []
    testData = []
    testTarget = []
    split = .7
    dl.loadDataset(filename, split, trainingData, trainingTarget, testData, testTarget)

    trainingData, testData = normalize(trainingData, testData)

    theNet = networkOfNodes()

    if filename == 'iris.csv':
        theNet.addLayer(3, len(trainingData))
    elif filename == 'diabetes.csv':
        theNet.addLayer(2, len(trainingData))

    predictions = theNet.predict(trainingData)
    print(predictions)


print("running Iris dataset")
main('iris.csv')
print("Running diabetes dataset")
main('diabetes.csv')
