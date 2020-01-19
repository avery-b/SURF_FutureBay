"""

~~~~~~~~~~~~~~~~~~~~~~~
S.U.R.F.
	Tools
~~~~~~~~~~~~~~~~~~~~~~~

Written by Adrian F. Santiago Tate and Ian Avery Bick
November 2018 - May 2019


"""

import os
import pandas as pd
import numpy as np
import time
import re

def readDatabase(database, rowKey, firstDataColumn):
	"""

	Database reader. Assumes metadata to left of firstDataColumn, where data starts.
	Column labels of data are integers of xValues, data are yValues.

	Args:
	-	database (Pandas DataFrame):
	-	rowKey (integer): used to select desired row by matching to values in first column.
		-	typically the Stru_ID or Cont_ID
	-	firstDataColumn (integer): index for column where data begins

	Returns: [xValues, yValues]
	-	xValues (list): 
	-	yValues (list): 

	"""
	try:
		#	from database select row with corresponding Stru_ID
		maskRow = database.iloc[:,0] == rowKey
		row = database.loc[maskRow,:]
		#	create list of all xValues and yValues
		xValues = np.array([int(x) for x in row.columns.values[firstDataColumn:]])
		yValues = np.array(row.values[0,firstDataColumn:])

		#	remove potential nans
		maskNaNs = pd.isnull(yValues)
		xValues = np.compress(~maskNaNs, xValues)
		yValues = np.compress(~maskNaNs, yValues)
		#	return list ready for linearInterpolation
		return [xValues, yValues]
	#	Handle exception, return None
	except Exception as e:
		print("Error in readDatabase:\n{}".format(e))
		return None


def getDfFromCsvList(pathCsvFiles, save=True):
	"""

	Creates dataframe similar to buildingDF from cvs files in given dir.
	-	Assumes filename is building ID and all files are similar.

	"""

	resultsDF = pd.DataFrame()

	completedResults = sorted([f[:-4] for f in os.listdir(pathCsvFiles) if not f.startswith('.')])

	for buildingID in completedResults:
		resultsDfRow = pd.read_csv(pathCsvFiles + buildingID + ".csv")
		resultsDfRow.insert(0, "TARGET_FID", np.full(1,buildingID))
		resultsDF = resultsDF.append(resultsDfRow)

	if save is True:
		resultsDF.to_csv(pathCsvFiles[:-1] + ".csv", index=False)

	return resultsDF


def saveSumStats(pathResults, ID, result, mean=True, stdev = False, varName="", bkgp = None):
	"""
	Saves one csv for each building containing mean and stdev for given column in result

	varName is an optional string to specify variable name if needed.
	"""
	#	For each column in original result dataframe, create new column in resultDF for mean and stdev
	resultDF = pd.DataFrame()
	for column in result.columns:
		if mean is True: resultDF[column + varName + "_mean"] = np.full(1, np.round(np.mean(result[column]), 2))
		if stdev is True: resultDF[column + varName + "_stdev"] = np.full(1, np.round(np.std(result[column]), 2))
	#	Add blockgroup
	if bkgp is not None: resultDF.insert(0, "bkgpGEOID", np.full(1,bkgp))
	#	Save to csv
	resultDF.to_csv(os.path.join(pathResults, "{}.csv".format(ID)), index = False)


def linearInterpolation(x, xValues, yValues):
	"""

	Linear interpolation function. Called after readDatabase.

	Args:
	-	x: float corresponds to column labes of lookup table
	-	xValues: list of column labels of lookup table
	-	yValues: list of values in lookup table

	Returns:
	-	Lowest value if x < min(xValues)
	-	Highest value if x > max(xValues)
	-	Exact value if found
	-	Linearly interpolated value if x between xValues
	-	None otherwise

	"""
	nBounds = len(xValues)
	try:
		#	exact match
		if x in xValues:
			exactValueFound = yValues[x == xValues]
			return exactValueFound[0]
		#	values outside bounds
		elif x < xValues[0]:
			return yValues[0]
		elif x > xValues[nBounds - 1]:
			return yValues[nBounds - 1]
		#	perform linear interpolation otherwise
		else:
		# iterate until bounds are found
			for bound in range(nBounds):
				if x > xValues[bound] and x < xValues[bound + 1]:
					#	n is the lower bound weight
					n = (x - xValues[bound]) / (xValues[bound + 1] - xValues[bound])
					#	linear interpolation
					yInterpolated = yValues[bound] * (1 - n) + yValues[bound + 1] * n
					return yInterpolated
	#	Handle exception, return None
	except Exception as e:
		print("There was an error in linearInterpolation for  x: {}  xValues: {} yValues: {} \n{}".format(x, xValues, yValues, e))
		return None


def calculateSLRAAL(xValues, yValues):
	"""
	Imports lists of X values and Y values and performs trapezoidal integration
	List order must correspond
	
	xValues: return periods (1, 20, 100 years etc.)
	yValues: DEL associated with return periods

	"""

	#convert return periods to occurrence rates
	xValues = [float((1/float(x))) for x in xValues]

	# Sort occurrence rates and associated DELs together
	xValues, yValues = (list(t) for t in zip(*sorted(zip(xValues, yValues))))

	# Perform trapezoidal integration
	# Account for greater return period storms by assuming constant DEL from largest available storm
	AAL = np.trapz(yValues, x=xValues) + np.trapz([yValues[0],yValues[0]], [0,xValues[0]])

	# Alternative trapezoidal integration to include a 1-year storm at $0 damage if not given
	# returnPeriod = 1
	# AAL = np.trapz(yValues, x=xValues) + np.trapz([yValues[0],yValues[0]], [0,xValues[0]]) + np.trapz([yValues[-1], 0], [xValues[-1], returnPeriod])

	# Not accounting for greater return period storms by assuming AAL from largest storm
#	AAL = np.trapz(yValues, x=xValues)

	return AAL

def calculateYearAALs(xValues, yValues, years, projectionDF=None):
	"""
	List order must correspond
	
	xValues: SLR amounts (0, 6, 12 etc.)
	yValues: AAL associated with SLR amounts
	years = list of years

	Output: List of year AALs in sorted order of years ex. [2020AAL, 2030AAL, 2040AAL]

	"""
	# Sort SLR amounts and associated AALs together
	xValues, yValues = (list(t) for t in zip(*sorted(zip(xValues, yValues))))

	# Create lists of lists in order to perform integrations
	xValues = [xValues[i:i+2] for i in range(0, len(xValues), 2-1)]
	yValues = [yValues[i:i+2] for i in range(0, len(yValues), 2-1)]

	# Create a list to store output AAL results for each year
	yearAALs = []

	# Iterate through SLR amounts
	# Need to deal with case of last list only having 1 entry
	for year in years:
		# Create a list to store temporary AAL results for each year
		tempAALs = []
		for (slrAmounts, AALs) in zip(xValues, yValues):
#			print('integrating between {} inches of SLR and {} AAL'.format(slrAmounts, AALs))
			if len(AALs) > 1:
				occurrenceRate = projectionDF[str(slrAmounts[0])].loc[year]
				integrationSum = occurrenceRate * ((AALs[0] + AALs[1])/2)
#				print('occurrence rate: {}'.format(occurrenceRate))
#				print('integration: {}'.format(integrationSum))
			else:
				occurrenceRate = projectionDF[str(slrAmounts[0])].loc[year]
				integrationSum = occurrenceRate * ((AALs[0]))
#				print('occurrence rate: {}'.format(occurrenceRate))
#				print('integration: {}'.format(integrationSum))
			tempAALs.append(integrationSum)

		yearAAL = sum(tempAALs)
		yearAALs.append(yearAAL)


#	print(years)	
#	print(yearAALs)

	return yearAALs

def calculateProjectedAAL(xValues, yValues):
	"""
	Calculates averaged annualized loss over a range of years
	List order must correspond
	
	xValues: years (2020, 2030, 2040 etc.)
	yValues: AAL associated with years
	"""
	xValues, yValues = (list(t) for t in zip(*sorted(zip(xValues, yValues))))

	projAAL = (np.trapz(yValues, x=xValues))/(xValues[-1]-xValues[0])

	return projAAL

def convertFieldNames(inputFields, newSuffix):
	outputFields = []

	for field in inputFields:
		integers = [int(s) for s in re.findall(r'\d+', field)]
		slrScenario = str(integers[0])
		returnPeriod = str(integers[1])
		outputFields.append(slrScenario + '_' + returnPeriod + '_' + newSuffix)

	return outputFields

def createSeaLevelRiseLookup(fields, values):
	"""
	Input: 
	List of fieldnames and corresponding values
	Field names store slrAmount_returnPeriod in name

	Output:
	Returns as an ordered dictionary of format:
	{'Sea Level Rise Scenario':[[returnPeriod1, DEL1], [returnPeriod2, DEL2], [returnPeriod3, DEL3]]}
	ex: {'0':[[1, 0_1_DEL], [20, 0_20_DEL], [100, 0_100_DEL]]}
		{'12':[[1, 12_1_DEL], [20, 12_20_DEL], [100, 12_100_DEL]]}

	"""

	dictionary = {}
	for field, value in zip(fields, values):
		integers = [int(s) for s in re.findall(r'\d+', field)]
		slrScenario = str(integers[0])
		returnPeriod = str(integers[1])

		if slrScenario in dictionary:
			dictionary[slrScenario].append([returnPeriod, value])
		else:
			dictionary[slrScenario] = [[returnPeriod, value]]

	# Sorts the dictionary entry lists so that they are in order of return period
	for key in dictionary:
		dictionary[key].sort(key=lambda x: int(x[0]))

	return dictionary

if __name__ == '__main__':

	#	Time it
	start = time.time()

	#	Test SLR AAL

	returnPeriods = [1, 100, 20]
	DELs = [0, 206525, 161150]

	slrAALs = calculateSLRAAL(returnPeriods, DELs)

	#	Test Year AAL

	slrScenarios = [0, 6, 12, 18]
	slrAALs = [85965, 197719, 275088, 429825]
	years = [2020, 2030, 2040]

	projectionDF = pd.DataFrame(pd.read_csv('/Users/ianbick/Desktop/DerekProjectionsTest.csv'))
	projectionDF.set_index(projectionDF.columns[0], inplace=True)
	projectionDF.rename(columns=projectionDF.iloc[0]).drop(projectionDF.index[0])

	yearAALs = calculateYearAALs(slrScenarios, slrAALs, years, projectionDF)

	#	Test projection AAL
	yearAALs =  [141936, 161846, 217444]

	projAAL = calculateProjectedAAL(years, yearAALs)

	end = time.time()
	
	print("{} seconds".format((end - start)))

else:
	pass