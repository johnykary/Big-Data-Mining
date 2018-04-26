import csv
import sys
import pandas as pd
import math
import numpy
import scipy
import time
import json
from math import sin, cos, sqrt, atan2, radians
from sklearn.model_selection import KFold
from sklearn.cross_validation import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
#This script is using Bag of Words for features
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
	
def convertCell(df, minlong, minlat, distance):
	
	cellsDf = pd.DataFrame(columns=['TripId','JourneyPatternId', 'JourneyCells'])
	
	maxc1 = 0
	maxc2 = 0
	
	for row in df.itertuples():
		
		j = row[3]
		jtoJson = json.loads(j)
		
		trajCells = ""
		previous = ""
		
		for i in range(0, len(jtoJson)):
			lon = jtoJson[i][1]
			lat= jtoJson[i][2]
			
			x = calculateDistanceHaversine(minlong, minlat, lon, minlat)
			y = calculateDistanceHaversine(minlong, minlat, minlong, lat)
			
			c1 = int(x/distance)
			c2 = int(y/distance)
			
			if c1 > maxc1:
				maxc1 = c1
			if c2 > maxc2:
				maxc2 = c2
			
			new = 'C' + str(c2) + ',' + str(c1) + ';'
			
			if new == previous :
				continue
			trajCells = trajCells + new
			previous = new
			
		d = {'TripId' : row[1],'JourneyPatternId' : row[2], 'JourneyCells' : trajCells}
		cellsDf.at[row[0], : ] = d
		
	return maxc1, maxc2, cellsDf

def coverCellTest(df, minlong, minlat, distance):
	
	cells = pd.DataFrame(columns=['JourneyCells'])
	
	for row in df.itertuples():
		
		j = row[2]
		jtoJson = json.loads(j)
		
		trajCells = ""
		previous = ""
		
		for i in range(0, len(jtoJson)):
			lon = jtoJson[i][1]
			lat= jtoJson[i][2]
			
			x = calculateDistanceHaversine(minlong, minlat, lon, minlat)
			y = calculateDistanceHaversine(minlong, minlat, minlong, lat)
			
			c1 = int(x/distance)
			c2 = int(y/distance)
			
			new = 'C' + str(c2) + ',' + str(c1) + ';'
			if new == previous :
				continue
			trajCells = trajCells + new
			previous = new
			
		d = {'JourneyCells' : trajCells}
		cells.at[row[0], : ] = d
	
	return cells	
	
def convertCellTo1D(df, pos, c1, c2):
	
	newArray = []
	
	for row in df.itertuples():
	
		j = row[pos]
		cellsArray = numpy.zeros(((c1 + 1)*(c2 + 1)), int)
		
		strs = j.replace('C','').split(';')
		cellsList = [map(str, s.split(';')) for s in strs]

		cellsList.pop()
		
		if len(cellsList) > 1:
			West, East, South, North, Circle = direction(cellsList)
		
		for i in cellsList:
			temp = [map(int,s.split(',')) for s in i]
			cellsArray[(c1 * (c2 - temp[0][0])) + temp[0][1]] = cellsArray[(c1 * temp[0][0]) + temp[0][1]] + 1
				
		newArray.append(cellsArray)
		
	return newArray

def createClassifiers(features, labels):
	
	kf = KFold(n_splits = 10)
	kf.get_n_splits(X)
	knn = KNeighborsClassifier(n_neighbors = 5)
	logreg = LogisticRegression()
	randomFore = RandomForestClassifier()

	knnScores = cross_val_score(knn, X, Y, cv = 10, scoring = 'accuracy')   
	logisticRegressionsScores = cross_val_score(logreg, X, Y, cv=10, scoring = 'accuracy')
	randomForestScores = cross_val_score(randomFore, X, Y, cv=10, scoring= 'accuracy')
	
	print "KNN score = " + str(knnScores.mean()) + " Logistic Regression = " + str(logisticRegressionsScores.mean()) + " Random Forest score = " + str(randomForestScores.mean())
	
	
	
	if (knnScores.mean()>logisticRegressionsScores.mean()) and (knnScores.mean() > randomForestScores.mean()):
		print "Best Classifier is KNNeighbors"
		return knn
	elif(logisticRegressionsScores.mean() > knnScores.mean()) and (logisticRegressionsScores.mean() > randomForestScores.mean()):
		print "Best Classifier is Logistic Regression"
		return logreg
	elif(randomForestScores.mean() > knnScores.mean()) and (randomForestScores.mean() > logisticRegressionsScores.mean()):
		print "Best Classifier is Random Forest"
		return randomFore

start_time = time.time()
		
df = pd.read_csv("tripsClean.csv", sep=';')

longitudes = []
latitudes = []

for row in df.itertuples():
	
	j = row[3]
	jtoJson = json.loads(j)
	
	for i in range(0, len(jtoJson)):
		lon = jtoJson[i][1]
		lat= jtoJson[i][2]
		
		longitudes.append(lon)
		latitudes.append(lat)
		
maxlong = max(longitudes)
maxlat = max(latitudes)
minlong = min(longitudes)
minlat = min(latitudes)

x1 = calculateDistanceHaversine(minlong, minlat, maxlong, minlat)
y1 = calculateDistanceHaversine(minlong, minlat, minlong, maxlat)

distance = y1/30

maxC1, maxC2, cellsDF = convertCell(df, minlong, minlat, distance)

print "The grid size is: " + str(maxC1) + "x" + str(maxC2)

cellsDF.drop(columns=['TripId'], inplace = True)
finalArrayOfCells = []
finalArrayOfCells = convertCellTo1D(cellsDF, 2, maxC1, maxC2)

X = finalArrayOfCells

Y = numpy.array(cellsDF['JourneyPatternId'])
classif = createClassifiers(X, Y)


predict = pd.read_csv("test_set.csv", sep=';')
testDF = coverCellTest(predict, minlong, minlat, distance)
testArrayOfCells = []
testArrayOfCells = convertCellTo1D(testDF, 1, maxC1, maxC2)

classif.fit(X,Y)
final = []
final = classif.predict(testArrayOfCells)

toCsv = pd.DataFrame(columns=['Test_Trip_ID','Predicted_JourneyPatternID'])

for i in range(0, len(final)):
	t = {'Test_Trip_ID' : i, 'Predicted_JourneyPatternID' : final[i]}
	toCsv.at[i, : ] = t

toCsv.to_csv('testSet_JourneyPatternIDs.csv', sep='\t', index=False)

end_time = time.time()

print (end_time - start_time)



