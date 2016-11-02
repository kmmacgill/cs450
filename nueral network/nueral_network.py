import random
import csv
from sklearn.utils import shuffle
from numpy.random import random_integers
from sklearn import preprocessing
from sklearn import datasets as ds
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

        if filename == 'iris.csv':
            for x in range(len(dataSet)):
                if dataSet[x][-1] == 'Iris-setosa':
                    dataSet[x][-1] = 0
                elif dataSet[x][-1] == 'Iris-versicolor':
                    dataSet[x][-1] = 1
                else:
                    dataSet[x][-1] = 2

        # now divy up the file according to the file name
        for x in range(len(dataSet)):
            for y in range(len(dataSet[x]) - 1):  # -1 is all but last
                dataSet[x][y] = float(dataSet[x][y])
            if random.random() < splitter:
                trainingData.append(dataSet[x][:-1])
                trainingTarget.append(dataSet[x][-1])
            else:
                testData.append(dataSet[x][:-1])
                testTarget.append(dataSet[x][-1])


class Neuron:
    """neuron class for networks"""
    bias = -1
    def __init__(self, inputNums):
        self.Weight = []
        self.error = 0.0

        for i in range(0, inputNums):
            randWeight = random.random()
            self.Weight.append(randWeight)
        self.Weight.append(random.random()) #for bias node

    def neuronBoom(self, inputs):
        totalVal = 0
        for i in range(len(inputs)):
            totalVal += (inputs[i] * self.Weight[i])
        totalVal += (self.bias * self.Weight[-1])  # for bias node
        return 1 / (1 + math.e ** totalVal)

class networkOfNodes:

    def __init__(self):
        self.layers = []

    def thereAndBackAgain(self, trainingTarget, trainingData):
        for epoch in range(100):
            for row in range(len(trainingData)):
                predictions = []
                rowsData = trainingData[row]
                for layer in range(len(self.layers)):
                    # the feed forward bit
                    explodedNeurons = []
                    for node in range(len(self.layers[layer])):
                        neuron = self.layers[layer][node]
                        explodedNeurons.append(neuron.neuronBoom(rowsData))
                    predictions.append(explodedNeurons)
                    rowsData = explodedNeurons

            CurrentLayer = -1
            for _ in range(len(self.layers)):
                # Set all errors
                nodesPerLayer = len(self.layers[CurrentLayer])
                for indexOfNueron in range(nodesPerLayer):
                    # Setting errors for each node in the layer
                    if CurrentLayer == -1:
                        # set the prediction to the highest values column number the set output to a one or zero
                        high = 0
                        predict = -1
                        for col in range(len(predictions[-1])):
                            if predictions[-1][col] > high:
                                high = predictions[-1][col]
                                predict = col
                        # is the prediction right or wrong
                        output = 0
                        if trainingTarget[row] == predict:
                            output = 1

                        nodeOutput = predictions[CurrentLayer][indexOfNueron]
                        newError = nodeOutput * (1 - nodeOutput) * (nodeOutput - output)
                        self.layers[CurrentLayer][indexOfNueron].error = newError

                    else:
                        # hidden layer

                        errorSum = 0.0
                        nextNode = len(self.layers[CurrentLayer + 1])
                        for nextOne in range(nextNode):
                            # this sums up the weight * error of all nodes in the layer to the right
                            nodesError = self.layers[CurrentLayer + 1][nextOne].error
                            nodesWeight = self.layers[CurrentLayer + 1][nextOne].Weight[nextOne]
                            errorSum += (nodesError * nodesWeight)

                        # Calculate and set this nodes error
                        nodeOutput = predictions[CurrentLayer][indexOfNueron]
                        newError = nodeOutput * (1 - nodeOutput) * errorSum
                        self.layers[CurrentLayer][indexOfNueron].error = newError
                CurrentLayer -= 1

            for eachLayer in range(len(self.layers)):
                # update weights
                learning_rate = .1
                for neurons in range(len(self.layers[eachLayer])):
                    numbWeights = len(self.layers[eachLayer][neurons].Weight)
                    for eachWeight in range(numbWeights):
                        bias = self.layers[eachLayer][neurons].bias
                        if eachLayer == 0:
                            if eachWeight != (numbWeights - 1):
                                thisInput = trainingData[eachLayer][eachWeight]
                            else:
                                thisInput = bias
                        else:
                            if eachWeight != (numbWeights - 1):
                                thisInput = predictions[eachLayer - 1][eachWeight]
                            else:
                                thisInput = bias

                        oldWeight = self.layers[eachLayer][neurons].Weight[eachWeight]
                        error = self.layers[eachLayer][neurons].error
                        new_weight = oldWeight - learning_rate * error * thisInput
                        self.layers[eachLayer][neurons].Weight[eachWeight] = new_weight

        thisPrediction = self.payItForward(trainingData)
        predicts = self.classifyPredictions(thisPrediction)
        percent = getAccuracy(predicts, trainingTarget)
        print("%i%%" % percent)

    def addLayer(self, howManyNeurons, howManyInputs = 0):
        if len(self.layers) != 0:
            howManyInputs = len(self.layers[-1])
        neurons = []
        for i in range(0, howManyNeurons):
            neurons.append(Neuron(howManyInputs))
        self.layers.append(neurons)

    def payItForward(self, data):
        for layer in range(len(self.layers)):
            prediction = []
            for row in range(len(data)):
                neuronsThatWentOff = []
                for node in range(len(self.layers[layer])):
                    neuron = self.layers[layer][node]
                    neuronsThatWentOff.append(neuron.neuronBoom(data[row]))
                prediction.append(neuronsThatWentOff)
            data = prediction
        return prediction

    def classifyPredictions(self, predictions):
        new_predictions = []
        for row in range(len(predictions)):
            predict = 0
            high = 0
            for col in range(len(predictions[row])):
                # find the highest prediction of the 3.
                if predictions[row][col] > high:
                    high = predictions[row][col]
                    predict = col
            new_predictions.append(predict)
        return new_predictions


def normalize(trainingData, testData):
    std_scale = preprocessing.StandardScaler().fit(trainingData)
    trainingData = std_scale.transform(trainingData)
    testData = std_scale.transform(testData)
    return trainingData, testData


def getAccuracy(predictions, targets):
    correctGuesses = 0
    for item in range(len(predictions)):
        if predictions[item] == targets[item]:
            correctGuesses += 1
    return (correctGuesses / len(targets)) * 100


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
        theNet.addLayer(3, len(trainingData[0]))
        predictions = theNet.payItForward(trainingData)
        print("One layer predictions:")
        print(predictions)
        theNet.addLayer(3)
        predictions = theNet.payItForward(trainingData)
        print("Predictions of same data, 2 layers:")
        print(predictions)
        predictions = theNet.classifyPredictions(predictions)
        accuracy = int(getAccuracy(predictions, trainingTarget))
        print("Accuracy before back-propogation: ", accuracy, "%")
        print(" ")
        print("Back propogating...")
        theNet.thereAndBackAgain(trainingTarget, trainingData)
        print("making new predictions with new weights...")
        predictions = theNet.payItForward(testData)
        predictions = theNet.classifyPredictions(predictions)
        accuracy = int(getAccuracy(predictions, trainingTarget))
        print("Accuracy: ", accuracy, "%")


    elif filename == 'diabetes.csv':
        print("adding layers")
        theNet.addLayer(2, len(trainingData[0]))
        theNet.addLayer(3)
        theNet.addLayer(1)
        theNet.addLayer(2)
        theNet.addLayer(2)
        print("finished adding layers")
        predictions = theNet.payItForward(trainingData)
        predictions = theNet.classifyPredictions(predictions)
        accuracy = int(getAccuracy(predictions, trainingTarget))
        print("Accuracy: ", accuracy, "%")

print("running Iris dataset")
main('iris.csv')
# print("Running diabetes dataset")
# main('diabetes.csv')
