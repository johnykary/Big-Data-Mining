import csv
import sys
import pandas as pd
import math

file = open("trips.csv", "w")

df = pd.read_csv("train_set.csv")
df.sort_values(by=['vehicleID', 'timestamp'], ascending=True, inplace=True)

file.write("TripId;JourneyPatternId;JourneyPoints\n")

journeyId = str(df.iat[0,0])

if journeyId == "nan":
	journeyId = "null"
	
detailsOfJournie = ""
autoIncrement = 0

for row in df.itertuples():

	ji = str(row[1])
	
	if ji == "nan":
		ji = "null"
	
	if journeyId == ji:
		detailsOfJournie = detailsOfJournie + "[" + str(row[3]) + "," + str(row[4]) + "," + str(row[5]) + "]" + ","
	else:
		detailsOfJournie = detailsOfJournie[:-1]
		file.write(str(autoIncrement) + ";" + str(journeyId) + ";" + "[" + detailsOfJournie + "]" + '\n')
		journeyId = str(row[1])
		
		if journeyId == "nan":
			journeyId = "null"
			
		autoIncrement = autoIncrement + 1
		detailsOfJournie = "[" + str(row[3]) + "," + str(row[4]) + "," + str(row[5]) + "]" + ","