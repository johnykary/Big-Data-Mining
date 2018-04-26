import csv
import sys
import pandas as pd
import math
from math import sin, cos, sqrt, atan2, radians
import json

file = open("tripsClean.csv", "w")
file.write("TripId;JourneyPatternId;JourneyPoints\n")

df = pd.read_csv("trips.csv",sep=';')

def calculateDistanceHaversine(lon1, lat1, lon2, lat2):
    R = 6373.0
    lat1 = radians(lat1)
    lon1 = radians(lon1)
    lat2 = radians(lat2)
    lon2 = radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))

    distance = R * c
    return distance

countNull = 0	
countKm1 = 0
countKm2 = 0

for row in df.itertuples():
	
	if str(row[2]) == "nan":
		countNull = countNull + 1
		continue
	
	j = row[3]
	jtoJson = json.loads(j)
	
	TotalDist = 0
	check = True
	for i in range(0, len(jtoJson)-1):
		lon1 = jtoJson[i][1]
		lat1 = jtoJson[i][2]
		lon2 = jtoJson[i+1][1]
		lat2 = jtoJson[i+1][2]
		distCalc = calculateDistanceHaversine(lon1, lat1, lon2, lat2)
		
		if distCalc > 2:
			check = False
			countKm1 = countKm1 + 1
			break
		
		TotalDist = TotalDist + distCalc
	
	
	if check == True:
		if TotalDist > 2:
			file.write(str(row[1]) + ";" + str(row[2]) + ";" + str(row[3]) + '\n')
		else:
			countKm2 = countKm2 + 1

print countNull, countKm1, countKm2			
 
