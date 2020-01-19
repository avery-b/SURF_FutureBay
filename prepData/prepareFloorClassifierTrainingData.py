import arcpy
import os
import csv
import pandas as pd

def shapefileToDF(buildings, pathOutput):
	"""
	Converts building footprint shapefile to a CSV, then a Pandas dataframe.
	"""
	fieldList = arcpy.ListFields(buildings)
	fieldNames = [field.name for field in fieldList]
	pathCSV = pathOutput + '/' + os.path.basename(buildings) + '_toDF' + '.csv'
	with open(pathCSV, 'wb') as csvFile:
		writer = csv.writer(csvFile)
		writer.writerow(fieldNames)
		with arcpy.da.SearchCursor(buildings, '*') as cursor:
			for row in cursor:
				writer.writerow(row)
		del cursor
		print(pathCSV + " CREATED")
	csvFile.close()

	buildingDF = pd.DataFrame(pd.read_csv(pathCSV))

	os.remove(pathCSV)

	return buildingDF

def extractData(buildingDF):
	buildingDF = buildingDF[['USE_INT', 'Height_ft', 'Area_ft', 'LAND_SQUAR', 'YEAR_BUILT','STORIES_NU']].astype('int')
	buildingDF = buildingDF.dropna(subset=['USE_INT', 'Height_ft', 'Area_ft', 'LAND_SQUAR', 'YEAR_BUILT', 'STORIES_NU'])
	buildingDF = buildingDF[(buildingDF != 0).all(1)]
	return buildingDF

if __name__ == '__main__':

	shapefile = "D:\_SURF_Files\surfData\publicationGeodatabase.gdb\smcBuildings_Microsoft2017_projected_oneBuilding_parcelData"
	pathOutput = "D:\_SURF_Files\publicationFiles"

	buildingDF = shapefileToDF(shapefile, pathOutput)
	buildingDF = extractData(buildingDF)
	buildingDF.to_csv(pathOutput + '\\' + 'floorTrainingData', index=False)

