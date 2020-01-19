"""

~~~~~~~~~~~~~~~~~~~~~~~
S.U.R.F.
	Data Prep
~~~~~~~~~~~~~~~~~~~~~~~

Written by Ian Avery Bick 
November 2018

"""


# Import arcpy module
import arcpy
import os
import numpy as np
import pandas as pd
import math
import csv
from Hazard import *
import sys

arcpy.env.overwriteOutput = True

def dataPrep(pathBuildingFootprints, pathParcels, aggregationTableDF, pathGDB, pathTempFiles, IDField, buildingAreaField, buildingHeightField):

	## Prepare raw Corelogic Parcels for processing
	# Removes duplicate parcels
	# Creates field resUnits for number of residential units in a given Parcel
	parcels = processParcels(pathParcels)

	## Delete parcels where 'PUC' use code field is not equal to an integer
	# parcels = deleteNullParcels(parcels)

	## Spatial join parcels to building footprints to get PUC attribute
	buildingFootprints = joinParcelsToBuildings(pathBuildingFootprints, parcels, pathGDB)

	## Calculate total floor area based on building footprint area and height
	# buildingFootprints = calculateLivingArea(buildingFootprints, buildingAreaField, buildingHeightField)

	## Join all aggregation levels (tracts, blocks, etc.) to give the buildings GEOID attributes
	spatialJoinDictionary = spatialJoinAggregations(buildingFootprints, aggregationTableDF, pathGDB, IDField)
	geoidFieldDict = spatialJoinDictionary['geoidFieldDict']
	buildingFootprints = spatialJoinDictionary['buildingFootprints']

	## Get a list of unique tract GEOIDs
	tractList = getUniqueTracts(buildingFootprints)
	
	## Split all buildings by their tract ID and create a dictionary sorted by tract GEOID

	buildingDictionary = splitBuildingShapefiles(buildingFootprints, tractList, pathTempFiles)

	return {'buildingDictionary':buildingDictionary, 'geoidFieldDict':geoidFieldDict}

def processParcels(pathParcels):
	# Copy Parcels File
	parcels = os.path.join(pathGDB, os.path.splitext(pathParcels)[0] + '_processedParcels')
	print('Path of processed parcels: ', parcels)
	arcpy.CopyFeatures_management(pathParcels, parcels)

	return parcels

def joinParcelsToBuildings(pathBuildingFootprints, parcels, pathGDB):
	tempFeaturesList = []
	# Process: Copy Footprint File
	print('Copying input building footprints.')
	buildingFootprints = os.path.join(pathGDB, os.path.splitext(pathBuildingFootprints)[0] + '_Copy')
	print('Building footprint copy path: ', buildingFootprints)
	arcpy.CopyFeatures_management(pathBuildingFootprints, buildingFootprints)
	tempFeaturesList.append(buildingFootprints)

	# Process: Spatial Join
	print('Spatial joining buildings to parcels.')
	buildingFootprintsAPN = os.path.join(pathGDB, os.path.splitext(pathBuildingFootprints)[0] + '_ParcelJoin')
	arcpy.SpatialJoin_analysis(buildingFootprints, parcels, buildingFootprintsAPN, "JOIN_ONE_TO_ONE", "KEEP_COMMON", "", "WITHIN", "", "")

	# Process: Summary Statistics
	print('Performing summary statistics of buildings in parcels.')
	buildingSummaryStatistics = os.path.join(pathGDB, os.path.splitext(buildingFootprintsAPN)[0] + '_summstats')
	arcpy.Statistics_analysis(buildingFootprintsAPN, buildingSummaryStatistics, "Area_ft SUM", "APN")
	tempFeaturesList.append(buildingSummaryStatistics)

	# Process: Join Field for Summary Statistics
	print('Joining summary statistics to building footprints.')
	arcpy.JoinField_management(buildingFootprintsAPN, "APN", buildingSummaryStatistics, "APN", "FREQUENCY;SUM_AREA")

	# Delete temporary files
	for tempFeature in tempFeaturesList:
		arcpy.Delete_management(tempFeature)

	return buildingFootprintsAPN

def spatialJoinAggregations(buildingFootprints, aggregationTableDF, pathGDB, IDField):	
	print('Spatial joining aggregation features to buildings.')
	# Create a list of lists of the rows from the aggregationTableDF
	aggregationList = aggregationTableDF.values.tolist()
	# Declare a blank list to which the new GEOID fields will be appended along with their field length for zfill
	geoidFieldDict = {}
	# Declare a blank list to which the spatial join outputs will be appended	
	tempSpatialJoinFeatures = []

	# Create spatial join output feature class
	tempSpatialOutput = os.path.join(pathGDB, 'tempSpatialOutput')
	# Iterate through aggregation levels
	for aggregation in aggregationList:
		# Create variables for each aggregations name, abbreviation, and feature path
		aggregationName = aggregation[0]
		aggregationAbbrev = aggregation[1]
		aggregationFieldLength = aggregation[2]
		aggregationFeature = aggregation[3]

		# Create spatial join output feature class
		tempAggregationOutput = os.path.join(pathGDB, tempSpatialOutput + aggregationAbbrev)
		arcpy.Copy_management(buildingFootprints, tempSpatialOutput)

		# Create a new field name for the aggregation geoid and append to field list
		geoidField = aggregationAbbrev + 'GEOID'
		geoidFieldDict[geoidField] = aggregationFieldLength

		# Field Mappings to allow only moving the 'GEOID' field from the aggregation shape to the building shape
		fieldmappings = arcpy.FieldMappings()

		# Add all fields from inputs.
		# fieldmappings.addTable(buildingFootprints)
		print(aggregationFeature)
		fieldmappings.addTable(aggregationFeature)
		
		# Spatial join the aggregation features to the buildingFootprints with the fieldmappings
		# Deleted field mappings for now
		arcpy.SpatialJoin_analysis(buildingFootprints, aggregationFeature, tempAggregationOutput, 'JOIN_ONE_TO_ONE', 'KEEP_COMMON', fieldmappings, 'INTERSECT')
		
		# Create a new GEOID field after the aggregation which it identifies
		newGEOID = aggregationAbbrev + 'GEOID'
		arcpy.AddField_management(tempAggregationOutput, newGEOID, 'TEXT')
		arcpy.CalculateField_management(tempAggregationOutput, newGEOID, '!'+'GEOID'+'!', "PYTHON_9.3")

		# Join the spatial join outputs containing GEOIDs to buildings
		arcpy.JoinField_management(buildingFootprints, IDField, tempAggregationOutput, IDField, [newGEOID])
		arcpy.Delete_management(tempAggregationOutput)

	# Create spatial join check output for building feature class
	geoidJoinOutput = os.path.join(pathGDB, buildingFootprints+'_GeoidJoin')
	arcpy.Copy_management(buildingFootprints, geoidJoinOutput)

	return {'buildingFootprints':buildingFootprints, 'geoidFieldDict':geoidFieldDict}

def getUniqueTracts(buildingFootprints):
	uniqueValues = set(row[0] for row in arcpy.da.SearchCursor(buildingFootprints, "tractGEOID"))

	tractList = list(uniqueValues)
	return tractList


def deleteNullParcels(parcels):
	expression = """ "PUC" IS NULL """
	cursor = arcpy.UpdateCursor(parcels, expression)  
	for row in cursor:  
		if not row.getValue('PUC'):  
			cursor.deleteRow() 

	return parcels

def splitBuildingShapefiles(buildingFootprints, tractList, pathTempFiles):
	buildingDictionary = {}
	for tract in tractList:
		if tract is not None:
			outName = 'buildings_' + tract 
			outLocation = pathTempFiles
			outFeature = os.path.join(outLocation, outName)
			tractGEOIDField = 'tractGEOID'
			expression = """ "tractGEOID" = '{}' """.format(tract)
			arcpy.FeatureClassToFeatureClass_conversion(buildingFootprints, outLocation, outName, expression)

			buildingDictionary[tract] = outFeature
		else:
			pass

	return buildingDictionary

def splitBuildingDataframe():
	### Split buildingDF by tractID into many buildingDFs
	## Delete all buildings in tracts with no floodings
	# Returns a dictionary with tract GEOID as the key containing a building dataframe value
	buildingDictionary = {}
	tracts = buildingDF['tractGEOID'].unique()
	for tract in tracts:
		tractDF = buildingDF.loc[buildingDF['tractGEOID']==tract]
		depthSum = tractDF[ffeFields].values.sum()
		if depthSum == 0:
			del tractDF
			pass
		else:
			buildingDictionary[tract] = tractDF
	del buildingDF

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

	buildingDF = pd.DataFrame(pd.read_csv(pathCSV, dtype={buildingUseCodeField: object}))

	os.remove(pathCSV)

	return buildingDF

if __name__ == '__main__':
	## Set file paths
	# Geodatabase file containing all required data
	pathGDB = "exampleGeodatabase.gdb"

	# Building polygon feature class, must contain fields for footprint Area and structure Height in feet	
	pathBuildingFootprints = "exampleGeodatabase.gdb/SanMateoBuildings_Microsoft2017_sample"

	# Tax assessor parcel polygon feature class, must contain fields for property use code, APN, and number of residential units.	
	pathParcels = "exampleGeodatabase.gdb/SanMateoParcels_Corelogic2017"

	# Directory to store intermediate files during run
	pathTempFiles = "exampleGeodatabase.gdb/temporaryFiles"

	# Directory to write output building data
	pathOutput = "outputData"

	# CSV containing the sea level rise scenarios, return periods, and raster file paths for hazards
	pathHazardRasters = "hazardData//floodMapDatabase_OCOF.csv"

	# CSV containing Census blocks, blockgroups, and tract filepaths
	pathAggregationTable = "prepData//aggregationShapes.csv"

	# CSV containing table relating property use code to building properties including depth damage curves 
	pathBuildingLookup = "exposureData/buildingData//smcParcelsSfqCurves.csv"

	# Training data for floor classifier model
	trainingData = "prepData//floorClassifierTrainingData.csv"

	## Instantiate Fragility Curves, Lookup Tables, and Databases as Pandas DataFrames:
	aggregationTableDF = pd.DataFrame(pd.read_csv(pathAggregationTable))
	hazardRastersDF = pd.DataFrame(pd.read_csv(pathHazardRasters))
	buildingLookupDF = pd.DataFrame(pd.read_csv(pathBuildingLookup))

	## Field in building shapefile describing the county building use type
	# This field allows a join with the 'PUC' field in the lookup table
	buildingUseCodeField = 'PUC'

	## Unique ID Field for building footprints
	# Ex: 'OBJECTID', 'FID', 'OID'
	IDField = 'buildingID'

	## Define fields for buliding properties
	# Area and height must be in feet
	buildingAreaField = 'Area_ft'
	buildingHeightField = 'Height_ft'

	## Statistic type for raster zonal statistics
	# Defining hazard value by mean or max value across building footprint
	# Ex: 'MEAN', 'MAX'
	zonalStatistic = 'MEAN'

	## Unit of depth for flood maps, will be converted to feet
	# Accepted: 'FT', 'IN', 'CM', 'M'
	depthUnit = 'FT'

	### Module 0 - Data Prep
	# Create a dictionary tract GEOIDs as keys and shapefiles of buildings in that tract as the value
	dataPrepDictionary = dataPrep(pathBuildingFootprints, pathParcels, aggregationTableDF, pathGDB, pathTempFiles, IDField, buildingAreaField, buildingHeightField)
	buildingDictionary = dataPrepDictionary['buildingDictionary']
	resultsDictionary = {}

	## Iterate over tracts to run model
	for tract in buildingDictionary.keys():
		print('Calculating flood exposure in Census Tract: ', tract)
		### Module 1 - Zonal Statistics
		buildings = buildingDictionary[tract]
		zonalDictionary = zonalStatistics(buildings, hazardRastersDF, IDField, zonalStatistic, pathOutput, aggregationTableDF, pathGDB, buildingUseCodeField, buildingAreaField, depthUnit)

#		# If there are no buildings in the tract after deleting buildings with no damage or use code, skip to next tract
		print('Valid buildings in tract: ', zonalDictionary['buildingDF'].shape[0])
		if zonalDictionary['buildingDF'].shape[0] == 0:
			print('No complete building data in Census Tract, moving to next Tract')
			del buildingDictionary[tract]
			continue

		buildingDF = zonalDictionary['buildingDF']
		resultsDictionary[tract] = buildingDF

	## Combine all entries in buildingDictionary into a single building dataframe
	resultsDataframeList = []

	print('Converting shapefiles to Pandas dataframes')
	for tract, buildings in resultsDictionary.items():
		resultsDataframeList.append(buildings)

	buildingDF = pd.concat(resultsDataframeList, axis=0)

	# Add building lookup table to buildingDF to give depth damage curve IDs
	# Delete any buildings with no use code
	buildingDF = buildingDF[np.isfinite(buildingDF[buildingUseCodeField])].reset_index()
	buildingDF = pd.merge(buildingDF, buildingLookupDF, on = buildingUseCodeField, how='left')

	## Can run a floor prediction model or assume a standard floor height to predict floor count
	# Broad assumption relating max building footprint height to an integer number of floors
#	buildingDF['nFloors'] = buildingDF['Height_ft'].div(25).round(0)

	# Run building floor classifier model
	Classifier = floorClassifier.trainClassifier(trainingData)
	buildingDF = buildingDF.dropna(axis = 0, subset = ['Height_ft', 'Area_ft', 'PUC'])
	buildingDF['nFloors'] = buildingDF.apply(lambda x: floorClassifier.floorCountPrediction(x['PUC'], x['Height_ft'], x['Area_ft'], Classifier), axis=1)

	# Add a field to store total floor area
	buildingDF['floorArea'] = buildingDF['nFloors'] * buildingDF[buildingAreaField]

	## Export buildings to CSV
	buildingDF.to_csv("output/exampleDataPrepResults.csv")


else:
	pass
