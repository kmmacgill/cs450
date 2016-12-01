import csv
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
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
        for row in range(len(dataset)):
            entry = []
            for col in range(len(dataset[row])):
                if col == 0:
                    if dataset[row][col] == "M":
                        testSet.append(0)
                    elif dataset[row][col] == "F":
                        testSet.append(1)
                    else:
                        testSet.append(2)
                else:
                    entry.append(float(dataset[row][col]))
            trainingSet.append(entry)
    else:
        for row in range(len(dataset)):
            for col in range(len(dataset[row])):
                dataset[row][col] = float(dataset[row][col])
            trainingSet.append(dataset[row][:-1])
            testSet.append(dataset[row][-1])
    return trainingSet, testSet


def main():
    print("running abalone data...")
    testSet = []
    trainSet = []
    dataset = loadCsvFile("abalone.csv")
    trainSet, testSet = getTrainingNTestingSets("abalone", dataset, trainSet, testSet)
    trainingData, trainingTarget, testData, testTarget = shuffleItUp(trainSet, testSet)

    print("k nearest neighbors:")
    predictions = []
    neigh = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree').fit(trainingData, trainingTarget)

    for i in range(len(testData)):
        predictions.append(neigh.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("knn accuracy: ", predictAccuracy, "%")

    print("decision tree:")
    predictions.clear()
    decTree = DecisionTreeClassifier(max_depth=5).fit(trainingData, trainingTarget)

    for i in range(len(testData)):
        predictions.append(decTree.predict(testData[i]))

    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("DecisionTree accuracy: ", predictAccuracy, "%")

    print("neural net:")
    predictions.clear()
    nutnet = MLPClassifier(max_iter=1000000).fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(nutnet.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Neural Net accuracy: ", predictAccuracy, "%")

    print("Bagging:")
    predictions.clear()
    bilbo = BaggingClassifier(base_estimator=neigh).fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(bilbo.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Bagging accuracy: ", predictAccuracy, "%")

    print("Boosting:")
    predictions.clear()
    superBooster = AdaBoostClassifier().fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(superBooster.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Boosting accuracy: ", predictAccuracy, "%")

    print("Random Forests:")
    predictions.clear()
    ranFor = RandomForestClassifier().fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(ranFor.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Boosting accuracy: ", predictAccuracy, "%")

##############################################################################
    print("running contraceptive Data...")
    testSet = []
    trainSet = []
    contra = loadCsvFile("contraceptiveData.csv")
    trainSet, testSet = getTrainingNTestingSets("contra", contra, trainSet, testSet)
    trainingData, trainingTarget, testData, testTarget = shuffleItUp(trainSet, testSet)

    print("k nearest neighbors:")
    predictions = []
    neigh = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree').fit(trainingData, trainingTarget)

    for i in range(len(testData)):
        predictions.append(neigh.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("knn accuracy: ", predictAccuracy, "%")

    print("decision tree:")
    predictions.clear()
    decTree = DecisionTreeClassifier(max_depth=5).fit(trainingData, trainingTarget)

    for i in range(len(testData)):
        predictions.append(decTree.predict(testData[i]))

    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("DecisionTree accuracy: ", predictAccuracy, "%")

    print("neural net:")
    predictions.clear()
    nutnet = MLPClassifier(max_iter=1000000).fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(nutnet.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Neural Net accuracy: ", predictAccuracy, "%")

    print("Bagging:")
    predictions.clear()
    bilbo = BaggingClassifier(base_estimator=neigh).fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(bilbo.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Bagging accuracy: ", predictAccuracy, "%")

    print("Boosting:")
    predictions.clear()
    superBooster = AdaBoostClassifier().fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(superBooster.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Boosting accuracy: ", predictAccuracy, "%")

    print("Random Forests:")
    predictions.clear()
    ranFor = RandomForestClassifier().fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(ranFor.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Boosting accuracy: ", predictAccuracy, "%")

###########################################################################
    print("running glass prediction data...")
    testSet = []
    trainSet = []
    glasses = loadCsvFile("glassTypes.csv")
    trainSet, testSet = getTrainingNTestingSets("glasses", glasses, trainSet, testSet)
    trainingData, trainingTarget, testData, testTarget = shuffleItUp(trainSet, testSet)

    print("k nearest neighbors:")
    predictions = []
    neigh = KNeighborsClassifier(n_neighbors=3, algorithm='ball_tree').fit(trainingData, trainingTarget)

    for i in range(len(testData)):
        predictions.append(neigh.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("knn accuracy: ", predictAccuracy, "%")

    print("decision tree:")
    predictions.clear()
    decTree = DecisionTreeClassifier(max_depth=5).fit(trainingData, trainingTarget)

    for i in range(len(testData)):
        predictions.append(decTree.predict(testData[i]))

    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("DecisionTree accuracy: ", predictAccuracy, "%")

    print("neural net:")
    predictions.clear()
    nutnet = MLPClassifier(max_iter=1000000).fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(nutnet.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Neural Net accuracy: ", predictAccuracy, "%")

    print("Bagging:")
    predictions.clear()
    bilbo = BaggingClassifier(base_estimator=neigh).fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(bilbo.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Bagging accuracy: ", predictAccuracy, "%")

    print("Boosting:")
    predictions.clear()
    superBooster = AdaBoostClassifier().fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(superBooster.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Boosting accuracy: ", predictAccuracy, "%")

    print("Random Forests:")
    predictions.clear()
    ranFor = RandomForestClassifier().fit(trainingData, trainingTarget)
    for i in range(len(testData)):
        predictions.append(ranFor.predict(testData[i]))
    accuracy = 0
    for i in range(len(predictions)):
        if predictions[i] == testTarget[i]:
            accuracy += 1
    predictAccuracy = int(accuracy / len(testTarget) * 100)
    print("Boosting accuracy: ", predictAccuracy, "%")


main()

