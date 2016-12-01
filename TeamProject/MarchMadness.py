import csv

def loadcsv(filename):
    f = open(filename)
    lines = csv.reader(f)
    dataset = list(lines)
    f.close()
    return dataset

seasonResult = read.csv("Rstudio/MarchMadness/season S results.csv", header=TRUE)
namesOfTeams = read.csv("Rstudio/MarchMadness/teams.csv", header=TRUE)
# declare a table, that will hold the name of team, their wins, and their losses
tableOfWinnersAndLosers < - list()
# for each team in the teams csv file
for (i in 1: nrow(namesOfTeams)){
    ID = namesOfTeams[i, "id"]
teamName = namesOfTeams[i, "name"]
print(teamName)
wins = 0
losses = 0
for (j in 1: nrow(seasonResult)){
if (seasonResult[j, "wteam"] == ID)
{
    wins = wins + 1
}
else if (seasonResult[j, "lteam"] == ID){
losses = losses + 1
}
}
entry < - list(teamName, wins, losses)
tableOfWinnersAndLosers[i] < - entry
}

for (thing in 1: nrow(tableOfWinnersAndLosers)){
    print(thing)
}
# grab the name of the team and shove it into the first entry of the table
# for each entry of the season, count the wins, the losses
# send the number of wins and losses into the table for the names entry
