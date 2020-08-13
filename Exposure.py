"""

~~~~~~~~~~~~~~~~~~~~~~~
S.U.R.F.
	Exposure Model
~~~~~~~~~~~~~~~~~~~~~~~

Written by Adrian F. Santiago Tate
-	October, 2018, to August, 2019

"""


# Import Libraries
from surfTools import *
import os
import sys
import numpy as np
import pandas as pd
import time
from copy import copy
from collections import OrderedDict
from multiprocessing import Pool

np.random.seed(0)

class ExposureModel(object):

	def __init__(self, pathBuildingDF, simName,

		#	kwargs
		years = [2020, 2030, 2040, 2050, 2060],
		repCostLocationFactor = 1.,
		nIterations = 1000,
		saveResults = False,
		#	Hazard data paths
		pathFloodScenarios = "hazardData/floodScenarios_OCOF.txt",
		pathProjections = "hazardData/occurrenceRateTables/OCOF/",
		#	Exposure data paths
		pathBuildingDatabase = "exposureData/buildingData/smcParcelsSfqCurves.csv",
		pathRepCostDatabase_mean = "exposureData/repCostData/rsMeans2017_mean.csv",
		pathRepCostDatabase_stdev = "exposureData/repCostData/rsMeans2017_stdev.csv",
		#	Vulnerability data paths
		pathFloodFragDatabaseStru_mean = "vulnerabilityData/floodFragDatabaseStru_mean.csv",
		pathFloodFragDatabaseCont_mean = "vulnerabilityData/floodFragDatabaseCont_mean.csv"):

		#~~~~~~~~~~~~~~~~~
		#	Options
		#~~~~~~~~~~~~~~~~~

		#	Save csv files of results
		self.saveResults = saveResults
		#	Number of Monte Carlo iterations
		self.nIterations = nIterations
		#	Years for AAL calculation
		self.years = years
		#	Simulation name
		self.simName = simName
		#	Factor to adjust local repCost
		self.repCostLocationFactor = repCostLocationFactor
		#	Temporary path to projections, need to modify Risk so file is read here instead
		self.pathProjections = pathProjections

		#~~~~~~~~~~~~~~~~~~
		#	Architecture
		#~~~~~~~~~~~~~~~~~~

		#	Dir to save results
		self.pathResults = os.path.join(os.path.split(pathBuildingDF)[0],"results_" + simName + "/")
		if os.path.exists(self.pathResults) is False and saveResults is not False: os.mkdir(self.pathResults)

		#~~~~~~~~~~~~~~~
		#	Load data
		#~~~~~~~~~~~~~~~

		#	Input data
		self.buildingDF = pd.DataFrame(pd.read_csv(pathBuildingDF))

		#	List of floodScenarios and corresponding rasters
		with open(pathFloodScenarios, 'r') as f: self.floodScenarios = f.read().splitlines()
#		self.hazardRasters = pd.DataFrame(pd.read_csv(pathHazardRasters))

		#	Exposure databases
		self.buildingDatabase = pd.DataFrame(pd.read_csv(pathBuildingDatabase))
		self.repCostDatabase_mean = pd.DataFrame(pd.read_csv(pathRepCostDatabase_mean))
		#	-	Handle not having stdev data for repCost
		if pathRepCostDatabase_stdev is None: self.repCostDatabase_stdev = None
		else: self.repCostDatabase_stdev = pd.DataFrame(pd.read_csv(pathRepCostDatabase_stdev))

		#	Vulnerability databases
		self.floodFragDatabaseStru_mean = pd.DataFrame(pd.read_csv(pathFloodFragDatabaseStru_mean))
		self.floodFragDatabaseCont_mean = pd.DataFrame(pd.read_csv(pathFloodFragDatabaseCont_mean))


	# ~~~~~~~~~~~~~~~~~~~~~
	# ExposureModel methods
	# ~~~~~~~~~~~~~~~~~~~~~

	def runModel(self, implementation = "parallel", nodes = 2, nBuildings = None, overwrite=False):
		"""

		Runs the Exposure Model by creating all buildings. Provides the user with high-level control.

		kwargs:
		-	implementation: ["parallel", "serial"]
			-	"parallel": multiprocessing.
			-	"serial": buildings created in sequence by iterating through rows of buildingDF.
		-	nodes: integer value specifies number of nodes for parallel processing.
		-	nBuildings: integer value specifies number of buildings to create - usually for testing.
		-	overwrite: specify whether or not to overwrite previous results in same directory

		"""
		try: self.saveResults is not False
		except:
			print("ExposureModel.saveResults is False. User must specify values to save to create buildings.")
			raise

		#	Process entire buildingDF if no value is specified by user
		if nBuildings is None: nBuildings = self.buildingDF.shape[0]

		#	Overwrite files by default, or only create buildings that have not been created yet if specified.
		if overwrite is True:
			buildingDFRows = range(nBuildings)
			#	Implement building creation
			print("Creating {} buildings".format(nBuildings))
			bldgs = [i for i in range(nBuildings)]
		elif overwrite is False:
			checkFolder = self.pathResults + self.saveResults[0]
			if os.path.exists(checkFolder) is True:
				bldgs = []
				completedResults = sorted([int(f[:-4]) for f in os.listdir(checkFolder) if not f.startswith('.')])
				for buildingDFRow in range(nBuildings):
					if self.buildingDF["buildingID"].loc[buildingDFRow] not in completedResults:
						bldgs.append(buildingDFRow)

		#	Create result dirs
		for resultName in self.saveResults:
			if os.path.exists(self.pathResults + resultName) is False: os.mkdir(self.pathResults + resultName)

		#	Implement building creation
#		print("Creating {} buildings".format(nBuildings))
#		bldgs = [i for i in range(nBuildings)]

		if implementation is "parallel":
			Pool(processes=nodes).map(self.floodBuilding, bldgs)
		elif implementation is "serial":
			for bldgDFRow in bldgs: self.floodBuilding(bldgDFRow)

		#	Compile results from csv result files into buildingDF-like csv
		for resultName in self.saveResults:
			if resultName in ["summaryStatsAAL", "percDmgStru", "buildingInfo"]: getDfFromCsvList(self.pathResults + resultName + "/")

	def floodBuilding(self, buildingDFRowNumber):

		#~~~~~~~~~~~~~~~~
		#	buildingDF
		#~~~~~~~~~~~~~~~~

		#	Read row of building DF
		building = self.buildingDF.iloc[buildingDFRowNumber]
		#	Get essential building attributes (unique identifier, area/livingArea in SF, and structure/content ID)
		buildingIDKey = "buildingID" # (can be "buildingID", "TARGET_FID", etc. depends on what Arc does)

		try:
			buildingID = int(building[buildingIDKey])

		except ValueError:
			print(building[buildingIDKey])
			print(building['APN'])

		livingArea = building["floorArea"]
		struID = int(building["Stru_ID"])
		contID = int(building["Cont_ID"])
		PUC = int(building["PUC"])

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#	buildingDatabase lookup
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		#	Find relevant row in buildingDatabase by matching to PUC
		maskBuildingID = self.buildingDatabase["PUC"] == PUC
		buildingDatabaseRow = self.buildingDatabase.loc[maskBuildingID,:]

		#	CSVR mean (ratio)
		CSVR = buildingDatabaseRow["CSVR"].values[0]
		#	CSVR standard deviation (ratio)
		CSVR_stdev = buildingDatabaseRow["CSVR_stdev"].values[0]
		#	FFE mean (feet)
		FFE = buildingDatabaseRow["FFE"].values[0]
		#	FFE standard deviation (feet)
		FFE_stdev = buildingDatabaseRow["FFE_stdev"].values[0]

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#	repCost and FC databases
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		#	RS Means 2017 database
		[livingAreas, repCosts] = readDatabase(self.repCostDatabase_mean, struID, firstDataColumn=5)
		
		#	-	Interpolate repCost
		repCost = linearInterpolation(livingArea, livingAreas, repCosts)

		#	USACE fragility curve databases for structures and contents
		[depthsStru, percDmgsStru] = readDatabase(self.floodFragDatabaseStru_mean, struID, firstDataColumn=4)
		[depthsCont, percDmgsCont] = readDatabase(self.floodFragDatabaseCont_mean, contID, firstDataColumn=4)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#	Randomize Input Parameters
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		#	repCost ($ per square feet of living area)
		if self.repCostDatabase_stdev is not None:
			[livingAreas_stdev, repCosts_stdev] = readDatabase(self.repCostDatabase_stdev, struID, firstDataColumn=5)
			repCost_stdev = linearInterpolation(livingArea, livingAreas_stdev, repCosts_stdev)
		else: repCost_stdev = 0

		randRepCost = self.getPositiveRandomNormalVariable(repCost, repCost_stdev)
		#	CSVR (content structure value ratio)
		randCSVR = self.getPositiveRandomNormalVariable(CSVR, CSVR_stdev, decimalPlaces=5)
		#	FFE (first floor elevation)
		randFFE = self.getPositiveRandomNormalVariable(FFE, FFE_stdev, decimalPlaces=5)

		#~~~~~~~~~~~~~~~~~~~
		#	Calculate DEL
		#~~~~~~~~~~~~~~~~~~~

		#	Value of structure and contents
		randValueStru = livingArea * randRepCost * self.repCostLocationFactor
		randValueCont = livingArea * randRepCost * randCSVR

		#	Random distribution of FFE-adjusted flood depths
		randFFEDepths = pd.DataFrame(columns=self.floodScenarios)
		for floodScenario in self.floodScenarios:
			#	Add FFE-adjusted depth from flooding (feet) column for given floodScenario
			randFFEDepths[floodScenario] = np.full(self.nIterations, building[floodScenario]) - randFFE
			#	Clip non-real depths of flooding
			randFFEDepths[floodScenario] = np.clip(randFFEDepths[floodScenario], 0, None)

		#	Random distribution of floodDamageStru, floodDamageCont
		#	-	Initialize by copying so randFFEDepths isn't being modified repeatedly (oh Python)
		randFldDmgStru = copy(randFFEDepths)
		randFldDmgCont = copy(randFFEDepths)

		#	-	Iterate through flood scenarios, apply floodDamage function, clip nonreal values
		for floodScenario in self.floodScenarios:
			#	Interpolate flood damage to structure and contents from fragility curves
			randFldDmgStru[floodScenario] = randFldDmgStru[floodScenario].apply(lambda x: linearInterpolation(x, depthsStru, percDmgsStru))
			randFldDmgCont[floodScenario] = randFldDmgCont[floodScenario].apply(lambda x: linearInterpolation(x, depthsCont, percDmgsCont))

		#	DEL calculation for structures and contents separately
		#	-	DEL scenarios, argument of riskCalculation FLAG TO LOOK INTO THIS
		delScenarios = [f + "_DEL" for f in self.floodScenarios]
		#	-	Initialize dataframes
		randStruDEL = pd.DataFrame(columns = delScenarios)
		randContDEL = pd.DataFrame(columns = delScenarios)
		#	-	Calculate DEL iteratively for all flood scenarios
		for floodScenario in self.floodScenarios:
			randStruDEL[floodScenario + "_DEL"] = randValueStru * randFldDmgStru[floodScenario]/100
			randContDEL[floodScenario + "_DEL"] = randValueCont * randFldDmgCont[floodScenario]/100

		#~~~~~~~~~~~~~~~~~~~
		#	Calculate AAL
		#~~~~~~~~~~~~~~~~~~~

		#	Iterate through each monte carlo realization

		slrProjectionCSVs = [self.pathProjections + f for f in os.listdir(self.pathProjections) if not f.startswith(".")]
		projectionDFs = {}

		for csv in slrProjectionCSVs:
			projectionName = os.path.basename(csv)[:-4]
			projectionDF = pd.DataFrame(pd.read_csv(csv, header=0))
			projectionDF.set_index(projectionDF.columns[0], inplace=True)
			projectionDF.rename(columns=projectionDF.iloc[0]).drop(projectionDF.index[0])
			projectionDFs[projectionName] = projectionDF

		projections = [os.path.basename(csv)[:-4] for csv in slrProjectionCSVs]
		projectionFields = [projection + "_AAL" for projection in projections]

		yearELFields = []
		yearFields = [str(year) + "_EL" for year in self.years]
		for projection in projections:
			yearELFields.extend([projection + "_" + year for year in yearFields])

		slrELFields = []
		for scenario in self.floodScenarios:
			integers = [int(s) for s in re.findall(r'\d+', scenario)]
			slr = str(integers[0])
			slrELFields.append(slr + '_EL')

		slrELFields = list(sorted(set(slrELFields)))

		randStruAAL = pd.DataFrame(columns=(projectionFields + yearELFields + slrELFields))
		randContAAL = pd.DataFrame(columns=(projectionFields + yearELFields + slrELFields))

		###	Structure AAL

		for index, row1 in randStruDEL.iterrows():
			row2 = randContDEL.iloc[index]

			struDict = createSeaLevelRiseLookup(delScenarios, row1)
			contDict = createSeaLevelRiseLookup(delScenarios, row2)

			struSlrELs = []
			contSlrELs = []

			projectionAALs = []
			slrScenarios = []

			# Calculate the Expected Loss for each Sea Level Rise Scenario
			for slrScenario in struDict:
				returnPeriods = [item[0] for item in struDict[slrScenario]]
				struDELs = [item[1] for item in struDict[slrScenario]]
				contDELs = [item[1] for item in contDict[slrScenario]]

				struSlrEL = calculateSLRAAL(returnPeriods, struDELs)
				struSlrELs.append(struSlrEL)

				contSlrEL = calculateSLRAAL(returnPeriods, contDELs)
				contSlrELs.append(contSlrEL)

				slrScenarios.append(slrScenario)
				slrField = slrScenario + '_EL'

				randStruAAL.at[index,slrField] = struSlrEL
				randContAAL.at[index,slrField] = contSlrEL

			# Calculate the Expected Loss for each Year and Sea Level Rise Projection
			for projection in projectionDFs:

				projectionDF = projectionDFs[projection]

				projectionField = projection + "_AAL"
				projectionYearFields = [projection + "_" + yearField for yearField in yearFields]

				struYearELs = calculateYearAALs(slrScenarios, struSlrELs, self.years, projectionDF)
				contYearELs = calculateYearAALs(slrScenarios, contSlrELs, self.years, projectionDF)

				randStruAAL.at[index, projectionYearFields] = struYearELs
				randContAAL.at[index, projectionYearFields] = contYearELs

				struProjectionAAL = calculateProjectedAAL(self.years, struYearELs)
				contProjectionAAL = calculateProjectedAAL(self.years, contYearELs)

				randStruAAL.at[index, projectionField] = struProjectionAAL
				randContAAL.at[index, projectionField] = contProjectionAAL

		#	-	Combine structure and content AAL to get total AAL
		randAAL = randStruAAL + randContAAL

		randStruResults = randStruAAL
		randContResults = randContAAL
		randAALResults = randAAL

		randStruAAL = randStruAAL[projectionFields]
		randContAAL = randStruAAL[projectionFields]
		randAAL = randAAL[projectionFields]

		#	-	Rename columns to differentiate stru, cont, and total AAL
		randStruAAL.columns = [column + "_Stru" for column in randStruAAL.columns]
		randContAAL.columns = [column + "_Cont" for column in randContAAL.columns]
		randStruResults.columns = [column + "_Stru" for column in randStruResults.columns]
		randContResults.columns = [column + "_Cont" for column in randContResults.columns]

		# ~~~~~~~~~~~~~~~~
		# Save results
		# ~~~~~~~~~~~~~~~~

		for resultName in self.saveResults:
			#	Path to results for given resultName
			pathResults = self.pathResults + resultName

			#	Save full Monte Carlo results for total AAL to csv
			if resultName == "monteCarloAAL":
				monteCarloAAL = pd.concat([randAAL, randStruAAL, randContAAL], axis = 1)
				monteCarloAAL.to_csv(self.pathResults + "monteCarloAAL/" + str(buildingID) + ".csv", index=False)

			#	Save summary statistics results for total AAL and Expected Losses to csv
			if resultName == "summaryStatsAAL":
				monteCarloAAL = pd.concat([randAALResults, randStruResults, randContResults], axis = 1)
				saveSumStats(pathResults, buildingID, monteCarloAAL, stdev=True)

			#	Save percent structural damage for equity
			if resultName == "percDmgStru": 
				saveSumStats(pathResults, buildingID, randFldDmgStru, varName="_dmgStru")

			#	Save replacement costs
			if resultName == "buildingInfo":
				data = {'randRepCost': randRepCost, 'randValueStru': randValueStru, 'randValueCont': randValueCont, 'randCSVR':randCSVR, 'randFFE':randFFE}
				buildingInfo = pd.DataFrame.from_dict(data).astype('float64')
				saveSumStats(pathResults, buildingID, buildingInfo, stdev=True)


	def getPositiveRandomNormalVariable(self, mean, stdev, decimalPlaces = 2):
		#	Normal distribution
		randVar = np.random.normal(mean, stdev, self.nIterations)
		#	Clip negative values
		randVar = np.clip(randVar, 0, None)
		#	Round to given number of decimal places
		return np.round(randVar, decimalPlaces)


# ~~~~~~~~~~~~~~~~~~~~~~
# Execute Exposure Model
# ~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

	#	Time it
	start = time.time()

	#	Exposure model
	exposure = ExposureModel(
		pathBuildingDF = "prepData/prepResults/residentialSample.csv", 
		simName = "test",
		#	kwargs
		repCostLocationFactor = 1.33,
		saveResults = ["monteCarloAAL", "percDmgStru", "summaryStatsAAL", "buildingInfo"])

	#	Create all buildings
	exposure.runModel(implementation = "parallel", nBuildings=None, nodes=2, overwrite=True)

	end = time.time()
	print("{} seconds".format((end - start)))

else:
	pass
