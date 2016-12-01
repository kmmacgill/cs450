import csv
import matplotlib.pyplot as plt
import numpy as np
import scipy.cluster.hierarchy as hac
from sklearn.cluster import KMeans


def loadcsv(filename):
    f = open(filename)
    lines = csv.reader(f)
    dataset = list(lines)
    f.close()
    return dataset
#grab the name of the team and shove it into the first entry of the table
# for each entry of the season, count the wins, the losses
# send the number of wins and losses into the table for the names entry


def buildTableOfWinLoss(seasonResults):
    namesofTeams = loadcsv("teams.csv")
    # declare a table, that will hold the name of team, their wins, and their losses
    tableOfWinnersAndLosers = []
    tableForIdTeams = []
    # for each team in the teams csv file
    for row in range(len(namesofTeams)):
        ID = namesofTeams[row][0]
        teamName = namesofTeams[row][1]
        wins = 0
        losses = 0
        totalPlayed = 0
        ratio = 0
        for col in range(len(seasonResults)):
            if seasonResults[col][2] == ID:
                wins += 1
            elif seasonResults[col][4] == ID:
                losses += 1
        totalPlayed = wins + losses
        if totalPlayed != 0:
            ratio = int(wins / totalPlayed * 100)
        elif wins > 0:
            ratio = 100
        else:
            ratio = 0
        entry1 = [losses, wins] # took out : ratio, totalPlayed
        entry2 = [ID, teamName]
        if totalPlayed == 0:
            pass
        else:
            tableOfWinnersAndLosers.append(entry1)
            tableForIdTeams.append(entry2)

    return tableOfWinnersAndLosers, tableForIdTeams

def main(): #TODO: get season from user, teams too.
    Results = loadcsv("tourney_results.csv")
    table, teamIDs = buildTableOfWinLoss(Results)
    npTable = np.array(table)
    kmeans = KMeans(init='k-means++', n_clusters=4, n_init=10).fit(table)

################################################################################################################
    #KMEANS PLOT
    # Step size of the mesh. Decrease to increase the quality of the VQ.
    h = .1  # point in the mesh [x_min, x_max]x[y_min, y_max].
    xmax = table[0][0]
    xmin = table[0][0]
    ymax = table[0][1]
    ymin = table[0][1]
    for i in range(len(table)):
        if table[i][0] > xmax:
            xmax = table[i][0]
        if table[i][0] < xmin:
            xmin = table[i][0]
        if table[i][1] > ymax:
            ymax = table[i][1]
        if table[i][1] < ymin:
            ymin = table[i][1]

    # Plot the decision boundary. For that, we will assign a color to each
    x_min, x_max = xmin, xmax
    y_min, y_max = ymin, ymax
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    # Obtain labels for each point in mesh. Use last trained model.
    Z = kmeans.predict(np.c_[xx.ravel(), yy.ravel()])  #hac.linkage(npTable)#

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1)
    plt.clf()
    plt.imshow(Z, interpolation='nearest',
               extent=(xx.min(), xx.max(), yy.min(), yy.max()),
               cmap=plt.cm.Paired,
               aspect='auto', origin='lower')

    for row in range(len(table)):
        plt.plot(table[row][0], table[row][1], 'bo', markersize=5)
    # Plot the centroids as a white X
    centroids = kmeans.cluster_centers_
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=169, linewidths=5,
                color='r', zorder=5)
    plt.title('Tournament Results 1995 - 2013')
    plt.show()                                  #KMEANS PLOT
    ###############################################################################################################
    #************************************************************************
    # these were used to construct the csv's, can be uncommented to do so again.
    #************************************************************************
    #seasonResults = loadcsv("regular_season_results.csv")
    #tourneyResults = loadcsv("tourney_results.csv")
    #tableToWrite = buildTableOfWinLoss(seasonResults)
    #anotherTableToWrite = buildTableOfWinLoss((tourneyResults))

    # This code writes the table generated above, to a csv file.
    # The csv was created before using the UI but the content was
    # added with this code.            "a" is for append, w is write, but overwrites csv data
    # with open("tourneyTeamTotals.csv", "a") as textFile: # change this to whatever csv file you want, after you'd added the file to the project
    #     textfileWriter = csv.writer(textFile, lineterminator='\n') #without this lineterminator set, it skips a line between entries.
    #     for row in anotherTableToWrite:
    #         textfileWriter.writerow(row)
    #
    # textFile.close()


main()