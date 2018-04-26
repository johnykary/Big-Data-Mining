import csv
import sys
import pandas as pd
import math
import numpy
import time
import json
import gmplot
from math import sin, cos, sqrt, atan2, radians


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

def DTW(search, target):
	n = len(search)
	m = len(target)
	dtw = numpy.zeros((n,m))
	
	for i in range(1,n):
		dtw[i][0] = float("inf")
	for i in range(1,m):
		dtw[0][i] = float("inf")
	dtw[0][0] = 0
	
	
	for i in range(1, n):
		for j in range(1, m):
			cost = calculateDistanceHaversine(search[i][1], search[i][2], target[j][1], target[j][2])
			dtw [i][j] = cost + min(dtw[i - 1][j], dtw[i][j - 1], dtw[i - 1][j - 1])
			
	return dtw[n-1][m-1]
	
def mapCreationTest(testTraj):
	
	for row in testTraj.itertuples():
	
		j = row[1]
		jtoJson = json.loads(j)
		
		longitudes = []
		latitudes = []
		filename = "mapTest" + str(row[0]) + ".html"
		
		for i in range(0, len(jtoJson)):
			lon = jtoJson[i][1]
			lat = jtoJson[i][2]
			
			longitudes.append(lon)
			latitudes.append(lat)
			
		gmap = gmplot.GoogleMapPlotter(lat, lon, 16)
		gmap.plot(latitudes, longitudes, 'green', edge_width=5)
		gmap.draw(filename)
	
def mapCreation(testTrip, testNumber):
	
	for row in testTrip.itertuples():
		
		j = row[4]
		filename = "mapNeighbor" + str(row[1]) + "for" + testNumber + ".html"
			
		jtoJson = json.loads(j)
		
		longitudes = []
		latitudes = []
		
		for i in range(0, len(jtoJson)):
			lon = jtoJson[i][1]
			lat = jtoJson[i][2]
			
			longitudes.append(lon)
			latitudes.append(lat)
			
		gmap = gmplot.GoogleMapPlotter(lat, lon, 16)
		gmap.plot(latitudes, longitudes, 'green', edge_width=5)
		gmap.draw(filename)

start_time = time.time()	
df1 = pd.read_csv("test_set_a1.csv",sep=';')
df2 = pd.read_csv("tripsClean.csv",sep=';')

mapCreationTest(df1)

for row1 in df1.itertuples():
	
	trajectory = row1[1]
	strs = trajectory.replace('[','').split('],')
	trajectorySearchList = [map(float, s.replace(']', '').split(',')) for s in strs]
	
	DTWdf = pd.DataFrame(columns=['Id','JourneyPatternId', 'DTW', 'JourneyPoints'])

	for row2 in df2.itertuples():
		
		search = row2[3]
		strs = search.replace('[','').split('],')
		trajectoryTargetList = [map(float, s.replace(']', '').split(',')) for s in strs]
		x = DTW(trajectorySearchList, trajectoryTargetList)

		d = {'Id' : row2[1],'JourneyPatternId':row2[2], 'DTW' : x, 'JourneyPoints':search}
		DTWdf.at[row2[0], : ] = d
		
	DTWdf.sort_values(by=['DTW'], ascending = True, inplace = True)
	DTWdf.drop(DTWdf.index[5:], inplace = True)
	mapCreation(DTWdf, str(row1[0]))
	
	
	endtime = time.time()
	print(DTWdf)
	print(endtime - start_time)
	



