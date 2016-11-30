import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import random
from sklearn.utils import shuffle


def loadCsvFile(fileName):
    f = open(fileName)
    lines = csv.reader(f)
    dataset = list(lines)
    f.close()
    return dataset

def shuffleItUp(data, target):
    # shuffle the data using a random number
    data, target = shuffle(data, target, random_state=int(random.random() * 100))
    trainingData = data[:int(len(data)*.7)]
    trainingTarget = target[:int(len(data)*.7)]
    testData = data[int(len(data)*.7):]
    testTarget = target[int(len(data)*.7):]
    return trainingData, trainingTarget, testData, testTarget

def normalize(trainingData, testData):
    """Scale data"""
    std_scale = preprocessing.StandardScaler().fit(trainingData)
    trainingData = std_scale.transform(trainingData)
    testData = std_scale.transform(testData)
    return trainingData, testData


def getTrainingNTestingSets(dataSetName, dataset, trainingSet=[], testSet=[]):
    if dataSetName == "abalone":
        for x in range(len(dataset)):
            entry = []
            for y in range(len(dataset[x])):
                if y == 0:
                    if dataset[x][y] == "M":
                        testSet.append(0)
                    elif dataset[x][y] == "F":
                        testSet.append(1)
                    else:
                        testSet.append(2)
                else:
                    entry.append(float(dataset[x][y]))
            trainingSet.append(entry)
    else:
        for x in range(len(dataset)):
            for y in range(len(dataset[x])):
                dataset[x][y] = float(dataset[x][y])
            trainingSet.append(dataset[x][:-1])
            testSet.append(dataset[x][-1])
    return trainingSet, testSet


def main():
    print("running abalone data...")
    abaloneTestSet = []
    abaloneTrainSet = []
    abalone = loadCsvFile("abalone.csv")
    abaloneTrainSet, abaloneTestSet = getTrainingNTestingSets("abalone", abalone, abaloneTrainSet, abaloneTestSet)
    abatrainingData, abatrainingTarget, abatestData, abatestTarget = shuffleItUp(abaloneTrainSet, abaloneTestSet)

    print("k nearest neighbors:")
    abaPredictions = []
    neigh = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree').fit(abatrainingData, abatrainingTarget)

    for i in range(len(abatestData)):
        abaPredictions.append(neigh.predict(abatestData[i]))
    accuracy = 0
    for i in range(len(abaPredictions)):
        if abaPredictions[i] == abatestTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(abatestTarget) * 100)
    print("knn accuracy: ", predictAccuracy, "%")

    print("decision tree:")
    abaPredictions.clear()
    decTree = DecisionTreeClassifier().fit(abatrainingData, abatrainingTarget)

    for i in range(len(abatestData)):
        abaPredictions.append(decTree.predict(abatestData[i]))
    print("neural net:")

    print("running contraceptive Data...")
    contraTestSet = []
    contraTrainSet = []
    contra = loadCsvFile("contraceptiveData.csv")
    contraTrainSet, contraTestSet = getTrainingNTestingSets("contra", contra, contraTrainSet, contraTestSet)
    conTrainingData, conTrainingTarget, conTestData, ConTestTarget = shuffleItUp(contraTrainSet, contraTestSet)

    print("k nearest neighbors:")

    print("decision tree:")

    print("neural net:")

    print("running glass prediction data...")
    glassesTestSet = []
    glassesTrainSet = []
    glasses = loadCsvFile("glassTypes.csv")
    glassesTrainSet, glassesTestSet = getTrainingNTestingSets("glasses", glasses, glassesTrainSet, glassesTestSet)
    trainingData, trainingTarget, testData, testTarget = shuffleItUp(glassesTrainSet, glassesTestSet)

    print("k nearest neighbors:")

    print("decision tree:")

    print("neural net:")

main()

