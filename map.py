import csv
import sys
import pandas as pd
import json
import gmplot

df = pd.read_csv("tripsClean.csv",sep=';')
df = df.drop_duplicates(subset='JourneyPatternId', keep = 'first')

df.drop(df.index[5:], inplace=True)

for row in df.itertuples():
	
	j = row[3]
	jtoJson = json.loads(j)
	
	longitudes = []
	latitudes = []
	filename = "map" + str(row[2]) + ".html"
	
	for i in range(0, len(jtoJson)):
		lon = jtoJson[i][1]
		lat= jtoJson[i][2]
		
		longitudes.append(lon)
		latitudes.append(lat)
		
	gmap = gmplot.GoogleMapPlotter(lat, lon, 16)
	gmap.plot(latitudes, longitudes, 'green', edge_width=5)
	gmap.draw(filename)