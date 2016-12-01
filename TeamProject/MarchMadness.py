import csv


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
        entry = [teamName, wins, losses, totalPlayed, ratio]
        tableOfWinnersAndLosers.append(entry)

    return tableOfWinnersAndLosers

def main(): #TODO: get season from user, teams too.
    #************************************************************************
    # these were used to construct the csv's, can be uncommented to do so again.
    #************************************************************************
    #seasonResults = loadcsv("regular_season_results.csv")
    tourneyResults = loadcsv("tourney_results.csv")
    #tableToWrite = buildTableOfWinLoss(seasonResults)
    anotherTableToWrite = buildTableOfWinLoss((tourneyResults))

    # This code writes the table generated above, to a csv file.
    # The csv was created before using the UI but the content was
    # added with this code.            "a" is for append, w is write, but overwrites csv data
    with open("tourneyTeamTotals.csv", "a") as textFile: # change this to whatever csv file you want, after you'd added the file to the project
        textfileWriter = csv.writer(textFile, lineterminator='\n') #without this lineterminator set, it skips a line between entries.
        for row in anotherTableToWrite:
            textfileWriter.writerow(row)

    textFile.close()


main()