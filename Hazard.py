"""

~~~~~~~~~~~~~~~~~~~~~~~
S.U.R.F.
	Hazard Functions
~~~~~~~~~~~~~~~~~~~~~~~

Written by Ian Avery Bick 
October 2018

"""

import os
import arcpy
from arcpy.sa import *
import pandas as pd
import numpy as np
import pysal as ps
import csv
import sys
import gc

arcpy.env.overwriteOutput = True
arcpy.env.extent = "MINOF"

if arcpy.CheckExtension("Spatial") == "Available":
    arcpy.AddMessage("Checking out Spatial")
    arcpy.CheckOutExtension("Spatial")
else:
    arcpy.AddError("Unable to get spatial analyst extension")
    arcpy.AddMessage(arcpy.GetMessages(0))
    sys.exit(0)

def zonalStatistics(buildings, hazardRastersDF, IDField, statistic, pathOutput, aggregationTableDF, pathGDB, buildingUseCodeField, buildingAreaField, depthUnit):
	"""
	Performs zonal statistics of each hazard raster on building shapefiles - based on given statistic type.
	Creates a new inundation amount field for each hazard raster based on names given from inputHazardRasters().
	Outputs a list of CSVs.
	"""
	# Create blank zonalOutput list to store zonalDF dataframes for each hazard raster
	scenarioOutputs = []
	# Create blank zonalField list to store names of new zonal statistics fields
	scenarioFields = []
	# Create dataframe of sea level rise scenarios, return periods, and associated filepaths and field names
	hazardRastersDF = inputHazardRasters(hazardRastersDF)

	# Call ShapefileToDF to convert buildingShapefile into a pandas dataframe
	buildingDF = ShapefileToDF(buildings, pathOutput)

	# Perform zonal statistics by iterating through hazardMapsDF with rows as named tuples
	for row in hazardRastersDF.itertuples(index=False, name='Pandas'):
		# Create output names for zonal statistics tables
		raster = row[2]
		pathOutTable = pathOutput + '//ZonalStatisticsTable_' + os.path.basename(raster)[:-4] + '.dbf'
		# Perform zonal statistics and output database file
		print('Performing Zonal Statistics for: ', os.path.basename(raster))

		arcpy.sa.ZonalStatisticsAsTable(buildings, IDField, raster, pathOutTable, 'DATA', 'ALL')
		# Convert database file into a pandas dataframe with pysal module
		scenarioDF = dbf2DF(pathOutTable)
		# Delete the zonal statistics shapefile from memory
		arcpy.Delete_management(pathOutTable)
		# Find name of zonal statistics field from hazardMapsDF
		fieldName = row[3]
		# Rename scenario statistic field in scenarioDF and append field name to a list
		scenarioDF[fieldName] = scenarioDF[statistic]
		# Append field name to list
		scenarioFields.append(fieldName)
		# Append dataframe to list
		scenarioOutputs.append(scenarioDF)

	# Iterate through the zonal outputs to create a single building footprint dataframe with zonal statistics fields
	# For each zonal output, create a temporary IDField for the join
	counter = 0
	for output in scenarioOutputs:
		scenarioField = scenarioFields[counter]
		tempDF = output[[IDField, scenarioField]]
		buildingDF = pd.merge(buildingDF, tempDF, left_on=IDField, right_on=IDField, how='outer')#.drop(IDField, axis=1)
		del tempDF
		counter +=1

	# Fill empty cells in zonal statistics dataframe with zeros
	buildingDF[scenarioFields] = buildingDF[scenarioFields].apply(pd.to_numeric)
	buildingDF[scenarioFields] = buildingDF[scenarioFields].fillna(0)

	# Adjust units of flood depth to feet
	buildingDF = adjustDepthUnits(buildingDF, depthUnit, scenarioFields)

	# Delete all buildings incurring no flood depth
	buildingDF['floodSum'] = buildingDF[scenarioFields].sum(axis=1)
	buildingDF = buildingDF.query('floodSum != 0')

	return {'buildingDF':buildingDF, 'scenarioFields':scenarioFields}

def adjustDepthUnits(buildingDF, depthUnit, scenarioFields):
	if depthUnit == 'FT':
		pass
	elif depthUnit == 'IN':
		buildingDF[scenarioFields] = buildingDF[scenarioFields] / 12
	elif depthUnit == 'M':
		buildingDF[scenarioFields] = buildingDF[scenarioFields] * 3.28084
	elif depthUnit == 'CM':
		buildingDF[scenarioFields] = buildingDF[scenarioFields] * 0.0328084	
	return buildingDF


def inputHazardRasters(hazardRastersDF):
	"""
	Input the flood map dataframe and creates/returns a dataframe with the following columns: 
	Sea Level Rise (in), Return Period (yr), File Path, Field Name

	The field name is a portmanteau of the SLR and return period. 
	Ex: For 0 in. SLR and 1 year return period, field name would equal 0_1
	"""
	# Create a temporary dataframe with SLR and return period fields
	tempDF = hazardRastersDF[['Sea Level Rise (in)', 'Return Period (yr)']]
	# Calculate field name
	tempDF['Field Name'] = tempDF['Sea Level Rise (in)'].astype(str) + '_' + tempDF['Return Period (yr)'].astype(str)
	# Append the temporary dataframe to the hazard map dataframe
	hazardRastersDF['Field Name'] = tempDF['Field Name']
	return hazardRastersDF

def ShapefileToDF(buildings, pathOutput):
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

### Legacy code - keeping for now for testing purposes
def appendCSVtoDF(scenarioOutputs, footprintDF, IDField):
	finalZonalDF = pd.DataFrame(pd.read_csv(scenarioOutputs[0])[IDField])
	for output in scenarioOutputs:
		columnName = os.path.basename(output)[:-4]
		outputDF = pd.DataFrame(pd.read_csv(output))
		finalZonalDF = pd.merge(finalZonalDF, outputDF, on=IDField)
	finalZonalDF = pd.merge(finalZonalDF, footprintDF, on=IDField)
	return finalZonalDF

def dbf2DF(dbfile):
	"""
	Reads in DBF files and returns Pandas DF.

	dbfile  : DBF file - Input to be imported
	upper   : Condition - If true, make column heads upper case

	This block of code adapted from https://gist.github.com/ryan-hill/f90b1c68f60d12baea81 
	"""
	#Pysal to open DBF
	db = ps.open(dbfile)
	#Convert dbf to dictionary
	d = {col: db.by_col(col) for col in db.header}
	#Convert to Pandas DF
	pandasDF = pd.DataFrame(d)
	db.close() 
	return pandasDF

# Execute as module

if __name__ == '__main__':
	pass
else:
	pass
