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
	
def mapCreation(trajFirst, testTrip, testNumber):
	 
	for row in testTrip.itertuples():
		
		j = row[4]
		filename = "mapSubNeighbor" + str(row[1]) + "for" + testNumber + ".html"
			
		jtoJson = json.loads(j)
		
		longitudes = []
		latitudes = []
		
		for i in range(0, len(jtoJson)):
			lon = jtoJson[i][1]
			lat = jtoJson[i][2]
			
			longitudes.append(lon)
			latitudes.append(lat)
			
		longitudesTest = []
		latitudesTest = []
		
		for i in range(0, len(trajFirst)):
			for j in range(0, len(jtoJson)):
				if calculateDistanceHaversine(longitudes[j], latitudes[j], trajFirst[i][1], trajFirst[i][2]) <= 0.2:
					longitudesTest.append(trajFirst[i][1]) 
					latitudesTest.append(trajFirst[i][2])
					
		gmap = gmplot.GoogleMapPlotter(lat, lon, 16)

		gmap.plot(latitudes, longitudes, 'green', edge_width=5)
		gmap.plot(latitudesTest, longitudesTest, 'red', edge_width=5)
		gmap.draw(filename)

def lcs(search , target):
    
	m = len(search)
	n = len(target)
	
	
	L = [[None]*(n+1) for i in xrange(m+1)]
	
	for i in range(m+1):
		for j in range(n+1):
			x = calculateDistanceHaversine(search[i-1][1], search[i-1][2], target[j-1][1], target[j-1][2]) 
            
			if i == 0 or j == 0 :
				L[i][j] = 0
			elif x <= 0.2:
				L[i][j] = L[i-1][j-1]+1
			else:
				L[i][j] = max(L[i-1][j] , L[i][j-1])
    
	return L[m][n]

start_time = time.time()
	
df1 = pd.read_csv("test_set_a2.csv",sep=';')
df2 = pd.read_csv("tripsClean.csv",sep=';')

mapCreationTest(df1)

for row1 in df1.itertuples():   
    
	trajectory = row1[1]
	strs = trajectory.replace('[','').split('],')
	trajectorySearchList = [map(float, s.replace(']', '').split(',')) for s in strs]
	
	LCSSdf = pd.DataFrame(columns=['Id','JourneyPatternId', 'MatchingPoints', 'JourneyPoints'])

	for row2 in df2.itertuples():
	
		search = row2[3]
		strs = search.replace('[','').split('],')
		trajectoryTargetList = [map(float, s.replace(']', '').split(',')) for s in strs]

		matchingPoints = lcs(trajectorySearchList, trajectoryTargetList)

		d = {'Id' : row2[1],'JourneyPatternId' : row2[2], 'MatchingPoints' : matchingPoints, 'JourneyPoints' : search}
		LCSSdf.at[row2[0], : ] = d
		
		
	endtime = time.time()
	LCSSdf.sort_values(by=['MatchingPoints'], ascending=False, inplace=True)
	LCSSdf = LCSSdf.drop_duplicates(subset='JourneyPatternId', keep = 'first')
	LCSSdf.drop(LCSSdf.index[5:], inplace=True)
	
	mapCreation(trajectorySearchList, LCSSdf, str(row1[0]))
	
	print(LCSSdf)
	print(endtime - start_time)
	
	
	
	
	
	