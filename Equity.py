"""

~~~~~~~~~~~~~~~~~~~~~~~
S.U.R.F. 
	Equity Model
~~~~~~~~~~~~~~~~~~~~~~~

Written by Ian Avery Bick and Adrian F. Santiago Tate
November 2018 - August 2019

"""

## Import Libraries
from surfTools import *
import time
import os
import pandas as pd
import numpy as np
from copy import copy
from multiprocessing import Pool
import pickle
import sys

np.warnings.filterwarnings('ignore')

class EquityModel(object):
	
	def __init__(self, pathBuildingDF, simName,
		fractionUncompensated = 1,
		structureDamageFields = None,
		years = [2020, 2030, 2040, 2050, 2060],
		saveResults = True,
		regionalPriceParity = 1.309,
		pathCensusBlockgroups = "equityData/ACS2017_BayArea_Demographics.csv",
		pathIncomeBracketLookup = "equityData/censusIncomeBracketLookup.csv",
		pathDemographicsLookup = "equityData/censusDemographicsLookup.csv",
		pathDiscretionaryIncomeLookup = "equityData/discretionaryIncomeTable2017.csv",
		pathProjections = "hazardData/occurrenceRateTables/OCOF/RCP85_DP16.csv"):

		#~~~~~~~~~~~~~~~~~
		#	Options
		#~~~~~~~~~~~~~~~~~
		#	Attributes from User Inputs
		self.fractionUncompensated = fractionUncompensated
		#	Simulation name
		self.simName = simName
		#	Years
		self.years = years

		#~~~~~~~~~~~~~~~~~~~~
		#	Architecture
		#~~~~~~~~~~~~~~~~~~~~
		#	Dir to save results
		self.pathResults = os.path.join(os.path.split(pathBuildingDF)[0],"results_" + simName + "/")
		if os.path.exists(self.pathResults) is False and saveResults is not False: os.mkdir(self.pathResults)
		#	Save csv files of results
		self.saveResults = saveResults
		#	Create result dirs
		for resultName in self.saveResults:
			if os.path.exists(self.pathResults + resultName) is False: os.mkdir(self.pathResults + resultName)
		#	Folder containing Monte Carlo iteration results from Exposure.py
		self.exposureResultsPath = exposureResultsPath
		# 	Folder containing structure damage results from Exposure.py
		self.structureDamagesPath = structureDamagesPath
		#	Path to projections is needed as attribute for methods
		self.pathProjections = pathProjections
		self.projectionDF = pd.DataFrame(pd.read_csv(self.pathProjections))
		self.projectionDF.set_index(self.projectionDF.columns[0], inplace=True)
		self.projectionDF.rename(columns=self.projectionDF.iloc[0]).drop(self.projectionDF.index[0])

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#	Load residential building data
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#	Create pandas dataframe from building CSV
		buildingDF = pd.DataFrame(pd.read_csv(pathBuildingDF))
		#	Get slice of residential buildings that have units
		self.resBldgs = copy(buildingDF.loc[buildingDF.useType.isin(["RESIDENTIAL"])]).dropna(subset=['nResUnits'])
		#	Distribute units in given APN among buildings in that parcel
		self.resBldgs["nBldgUnits"] = self.distributeParcelUnits()
		#	Remove buildings with zero units
		self.resBldgs = self.resBldgs[self.resBldgs["nBldgUnits"]!=0]

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#	Load blockgroup level data
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		#	Read census and income blockgroup data
		censusBkgpDF = pd.DataFrame(pd.read_csv(pathCensusBlockgroups))
		incmLkup = pd.DataFrame(pd.read_csv(pathIncomeBracketLookup))
		demoLkup = pd.DataFrame(pd.read_csv(pathDemographicsLookup))
		#	Get column labels for census income brackets and demographics
		incmFields = incmLkup.HouseholdCount.values.tolist()+incmLkup.MarginError.values.tolist()
		demoFields = demoLkup.Estimate.values.tolist()+demoLkup.MarginError.values.tolist()
		censusFields =  ["bkgpGEOID"] + incmFields + demoFields
		#	Get unique geoid's
		bkgpGEOIDs = self.resBldgs.bkgpGEOID.unique()
		#	Slice blockgroup dataframe by census fields and unique bkgpGEOIDs in buildingDF
#		print('blockgroup GEOIDs', bkgpGEOIDs)
		self.bkgpDF = censusBkgpDF[censusFields].loc[censusBkgpDF["bkgpGEOID"].isin(bkgpGEOIDs)]
#		print('blockgroupDF', self.bkgpDF)

		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
		#	Load Discretionary Income data
		#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

		self.discretionaryIncomeDF = pd.DataFrame(pd.read_csv(pathDiscretionaryIncomeLookup))
		self.discretionaryIncomeDF['regionalNonDisExpenses'] = \
			self.discretionaryIncomeDF['nonDiscretionaryExpenses'].apply(lambda x: x * regionalPriceParity).astype('int64')
		self.discretionaryIncomeDF['regionalDisIncome'] = \
			self.discretionaryIncomeDF['incomeBeforeTaxes'] - self.discretionaryIncomeDF['regionalNonDisExpenses'].astype('int64')

		#~~~~~~~~~~~~~~~~~~~~~~~~
		#	Load Exposure data
		#~~~~~~~~~~~~~~~~~~~~~~~~

		# Identify the structural flood damage fields which are used to calculate displacement time
		if structureDamageFields is None: 
			self.structureDamageFields = \
				list(pd.DataFrame(pd.read_csv(self.structureDamagesPath + os.listdir(self.structureDamagesPath)[0])))
		else: self.structureDamageFields = structureDamageFields	


	def runModel(self):
		"""
		Runs Equity model by iterating over blockgroups
		"""
		for index, blockgroup in self.bkgpDF.iterrows():
			#	Create folders to save results
			for resultName in self.saveResults:
				if resultName in ["monteCarloADI", "sumStats_unitADI", "sumStats_householdADI"]:
					pathBkgpResults = os.path.join(self.pathResults,resultName,str(blockgroup["bkgpGEOID"]))
					if os.path.exists(pathBkgpResults) is False: os.mkdir(pathBkgpResults)
			#	Calculate owner/renter/empty fraction for each blockgroup
			bkgpVacantUnits = blockgroup.B25004e1
			bkgpOccupiedUnits = blockgroup.B25009e1
			bkgpCensusTotalUnits = bkgpVacantUnits + bkgpOccupiedUnits
			bkgpVacantFraction = bkgpVacantUnits / bkgpCensusTotalUnits
			bkgpRentFraction = blockgroup.B25009e10 / bkgpCensusTotalUnits
			bkgpOwnFraction = blockgroup.B25009e2 / bkgpCensusTotalUnits
			# Calculate fraction of households in each income bracket for each blockgroup
			bkgpIncome_0_9999 = blockgroup['B19001e2'] / bkgpOccupiedUnits
			bkgpIncome_10000_14999 = blockgroup['B19001e3'] / bkgpOccupiedUnits
			bkgpIncome_15000_19999 = blockgroup['B19001e4'] / bkgpOccupiedUnits
			bkgpIncome_20000_24999 = blockgroup['B19001e5'] / bkgpOccupiedUnits
			bkgpIncome_25000_29999 = blockgroup['B19001e6'] / bkgpOccupiedUnits
			bkgpIncome_30000_34999 = blockgroup['B19001e7'] / bkgpOccupiedUnits
			bkgpIncome_35000_39999 = blockgroup['B19001e8'] / bkgpOccupiedUnits
			bkgpIncome_40000_44999 = blockgroup['B19001e9'] / bkgpOccupiedUnits
			bkgpIncome_45000_49999 = blockgroup['B19001e10'] / bkgpOccupiedUnits
			bkgpIncome_50000_59999 = blockgroup['B19001e11'] / bkgpOccupiedUnits
			bkgpIncome_60000_74999 = blockgroup['B19001e12'] / bkgpOccupiedUnits
			bkgpIncome_75000_99999 = blockgroup['B19001e13'] / bkgpOccupiedUnits
			bkgpIncome_100000_124999 = blockgroup['B19001e14'] / bkgpOccupiedUnits
			bkgpIncome_125000_149999 = blockgroup['B19001e15'] / bkgpOccupiedUnits
			bkgpIncome_150000_199999 = blockgroup['B19001e16'] / bkgpOccupiedUnits
			bkgpIncome_200000_1000000 = blockgroup['B19001e17'] / bkgpOccupiedUnits
			#	Dataframe it
			incomeDF = pd.DataFrame([
					[bkgpIncome_0_9999, 0, 9999, '0_9999'], 
					[bkgpIncome_10000_14999, 10000, 14999, '10000_14999'],
					[bkgpIncome_15000_19999, 15000, 19999, '15000_19999'],
					[bkgpIncome_20000_24999, 20000, 24999, '20000_24999'],
					[bkgpIncome_25000_29999, 25000, 29999, '25000_29999'],
					[bkgpIncome_30000_34999, 30000, 34999, '30000_34999'],
					[bkgpIncome_35000_39999, 35000, 39999, '35000_39999'],
					[bkgpIncome_40000_44999, 40000, 44999, '40000_44999'],
					[bkgpIncome_45000_49999, 45000, 49999, '45000_49999'],
					[bkgpIncome_50000_59999, 50000, 59999, '50000_59999'],
					[bkgpIncome_60000_74999, 60000, 74999, '60000_74999'],
					[bkgpIncome_75000_99999, 75000, 99999, '75000_99999'],
					[bkgpIncome_100000_124999, 100000, 124999, '100000_124999'],
					[bkgpIncome_125000_149999, 125000, 149999, '125000_149999'],
					[bkgpIncome_150000_199999, 150000, 199999, '150000_199999'],
					[bkgpIncome_200000_1000000, 200000, 1000000, '200000_1000000']])
			#	Get buildings in blockgroup
			bkgpBldgs = self.resBldgs.loc[self.resBldgs["bkgpGEOID"].isin([blockgroup["bkgpGEOID"]])]
			bkgpCorelogicTotalUnits = bkgpBldgs['nBldgUnits'].sum()
			bldgUnits = bkgpBldgs['nBldgUnits'].values
			#	Ensure that there are equal numbers of housing units and census units in a blockgroup
			normalizedUnits = self.normalizeHousingUnits(bldgUnits, bkgpCorelogicTotalUnits, bkgpCensusTotalUnits)
			bkgpBldgs['correctedBldgUnits'] = self.correctHousingUnits(normalizedUnits, bkgpCorelogicTotalUnits, bkgpCensusTotalUnits)
			#	If any buildings end up with zero units then drop that building
			bkgpBldgs = bkgpBldgs[bkgpBldgs.correctedBldgUnits != 0]
			#	Iterate through buildings
			for index, building in bkgpBldgs.iterrows():
				#	Open pandas dataframe (Exposure results) for given building
				exposureResults = pd.DataFrame(pd.read_csv(self.exposureResultsPath + str(building['buildingID']) + '.csv'))
				nMonteCarloIterations = exposureResults.shape[0]
				#	Obtain structure percent damages 
				structureDamageDF = pd.DataFrame(pd.read_csv(self.structureDamagesPath + str(building['buildingID']) + '.csv'))
				structureDamages = structureDamageDF.iloc[0].values.tolist()
				#	Calculate displacement time and cost based on structure damage results from Exposure.py
				displacementTimes = [self.calculateDisplacementTime(damage) for damage in structureDamages]
				displacementCosts = [self.calculateDisplacementCost(time) for time in displacementTimes]
				#	Create new displacement time and displacement cost fields
#				displacementTimeFields = convertFieldNames(self.structureDamageFields, 'disTime')
				displacementCostFields = convertFieldNames(self.structureDamageFields, 'disCost')
				#	Create a dicitonary linking sea level rise amounts, return periods, and displacement field names
				displacementCostDict = createSeaLevelRiseLookup(displacementCostFields, displacementCosts)
				#	Calculate the annualized displacement costs for the building
				#	Calculate annualized displacement for each sea level rise scenario
				slrADCs = []
				slrScenarios = []
				for slrScenario in displacementCostDict:
					returnPeriods = [item[0] for item in displacementCostDict[slrScenario]]
					dispCosts = [item[1] for item in displacementCostDict[slrScenario]]
					slrADC = calculateSLRAAL(returnPeriods, dispCosts)
					slrADCs.append(slrADC)
					slrScenarios.append(slrScenario)
				#	Calculate annualized displacement cost for each year
				yearADCs = calculateYearAALs(slrScenarios, slrADCs, self.years, projectionDF=self.projectionDF)
				#	Calculate total annualized displacement
				buildingADC = calculateProjectedAAL(self.years, yearADCs)
				#	Obtain structure and content AALs
				RCP85Fields = [s for s in list(exposureResults) if "RCP26" in s]
				RCP85StruField = [s for s in RCP85Fields if "Stru" in s][0]
				RCP85ContField = [s for s in RCP85Fields if "Cont" in s][0]
				RCP85Fields = [RCP85StruField, RCP85ContField]
				#	Assign unit floors from bottom-up
				unitFloors = self.getUnitFloors(building["correctedBldgUnits"], building["nFloors"])
				#	Randomly assign whether units are vacant (0), renter-occupied (1), or owner-occupied (2)
				randUnitOwnership = pd.DataFrame(np.random.choice(range(3),
					p = [bkgpVacantFraction, bkgpRentFraction, bkgpOwnFraction],
					size=(nMonteCarloIterations, building["correctedBldgUnits"])))
				#	Randomly assign income from income data
				randomUnitIncomes = pd.DataFrame(np.random.choice(list(range(incomeDF.shape[0])),
					p = incomeDF[0].values.tolist(),
					size=(nMonteCarloIterations, building["correctedBldgUnits"])))\
					.apply(lambda x: self.getRandomUnitIncome(x, incomeDF))
				if isinstance(randomUnitIncomes, pd.DataFrame):
					unitDiscretionaryIncomes = copy(randomUnitIncomes).applymap(lambda x: self.getDiscretionaryIncome(x))
				else:
					unitDiscretionaryIncomes = copy(randomUnitIncomes).map(lambda x: self.getDiscretionaryIncome(x))
					break
				# For each unit, iterate over RCP AAL results and append new ADI and ADI results to lists
				buildingResultsADI = []
				buildingResultsDI = []
				for iteration in range(nMonteCarloIterations):
					structureAAL = RCP85Fields[0]
					contentAAL = RCP85Fields[1]
					rcpName= RCP85Fields[0][0:5]
					unitOwnership = randUnitOwnership.iloc[[iteration]].values.tolist()[0]
					unitDiscretionaryIncome = unitDiscretionaryIncomes.iloc[[iteration]].values.tolist()[0]
					buildingResultsDI.append(unitDiscretionaryIncome)
					#	Return lists of AALs of same length as number of units
					bldgContentAALs = [int(exposureResults[contentAAL].iloc[[iteration]])] * len(unitFloors)
					bldgStructureAALs = [int(exposureResults[structureAAL].iloc[[iteration]])] * len(unitFloors)
					#	Get ADI rows
					unitADIs = self.runADI(unitFloors, unitOwnership, unitDiscretionaryIncome, bldgContentAALs, bldgStructureAALs, buildingADC)
					buildingResultsADI.append(unitADIs)
				buildingResultsADI = pd.DataFrame(buildingResultsADI)
				buildingResultsDI = pd.DataFrame(buildingResultsDI)
				#	If unit has null ADI, make DI null
				nanMask = buildingResultsADI.isnull()
				unitDiscretionaryIncomes = unitDiscretionaryIncomes.mask(nanMask, np.nan)


				# ~~~~~~~~~~~~~
				# Save results
				# ~~~~~~~~~~~~~
				for resultName in self.saveResults: 
					#	Path to results for given blockgroup
					pathBkgpResults = os.path.join(self.pathResults,resultName,str(blockgroup["bkgpGEOID"]))
					#	Create dataframes for each unit
					for unitNumber in range(building["correctedBldgUnits"]):
						monteCarloADI = pd.DataFrame({"Income": randomUnitIncomes[unitNumber], \
							"DI": unitDiscretionaryIncomes[unitNumber], "ADI": buildingResultsADI[unitNumber]})
						#	Save Income, DI, and ADI monte carlo results for each unit
						if resultName == "monteCarloADI":
							monteCarloADI.to_csv(pathBkgpResults + \
								"/{}_{}.csv".format(building["buildingID"],unitNumber+1), index=False)
						#	Save sumarry statistics of Income, DI, and ADI for each unit
						elif resultName == "sumStats_unitADI":
							saveSumStats(pathBkgpResults,"{}_{}".format(building["buildingID"],unitNumber+1), \
								monteCarloADI, stdev=True, bkgp = building["bkgpGEOID"])
						else: pass


	def aggregateResults(self):
		"""
		Create summary statistics files for housing units and households
		"""
		#	Column labels for each income bracket
		incomeBracketColumns = ["0-9999", "10000-14999", "15000-19999", "20000-24999", "25000-29999",\
							"30000-34999", "35000-39999", "40000-44999", "45000-49999", "50000-59999",\
							"60000-74999", "75000-99999", "100000-124999", "125000-149999", "150000-199999", "200000-1000000"]
		#	Hard-code dataframe with income brackets
		incomeDF = pd.DataFrame([
					[0, 9999], 
					[10000, 14999],
					[15000, 19999],
					[20000, 24999],
					[25000, 29999],
					[30000, 34999],
					[35000, 39999],
					[40000, 44999],
					[45000, 49999],
					[50000, 59999],
					[60000, 74999],
					[75000, 99999],
					[100000, 124999],
					[125000, 149999],
					[150000, 199999],
					[200000, 1000000]],
					columns = ["lo", "hi"])
		#	Compile results from csv result files into buildingDF-like csv
		for resultName in self.saveResults:
			# ~~~~~~
			# Units
			# ~~~~~~
			if resultName == "sumStats_unitADI":
				#	Initialize empty dataframe
				sumStats_unitADI_DF = pd.DataFrame()
				#	Iterate through folders containing results for each blockgroup
				for bkgp in self.resBldgs.bkgpGEOID.unique():
					pathBkgpResults = os.path.join(self.pathResults,resultName,"{}/".format(bkgp))
					resultDF = getDfFromCsvList(pathBkgpResults, save=False)
					sumStats_unitADI_DF = sumStats_unitADI_DF.append(resultDF)
				#	Save
				sumStats_unitADI_DF.to_csv(self.pathResults + "sumStats_unitADI.csv", index=False)
			# ~~~~~~~~~~~
			# Households
			# ~~~~~~~~~~~
			elif resultName == "sumStats_householdADI":

				def sumStatsForHouseholdADI(bkgp):
					"""
					Performs summary statistics on ADI, DI, and Income results for households at the block group scale
					"""
					#	Get list of Monte Carlo result files
					pathMonteCarloADI = os.path.join(self.pathResults,"monteCarloADI", "{}/".format(bkgp))
					monteCarloCsvFiles = sorted([f[:-4] for f in os.listdir(pathMonteCarloADI) if not f.startswith('.')])
					#	Fill dictionary with dataframes of ADI Monte Carlo results
					householdADI = {}
					for monteCarloResult in monteCarloCsvFiles:
						#	Read csv
						householdADI[monteCarloResult] = pd.DataFrame(pd.read_csv(pathMonteCarloADI + monteCarloResult + ".csv"))
						#	Create mask for each income bracket
						for incomeBracket in range(incomeDF.shape[0]):
							income = householdADI[monteCarloResult]["Income"]
							ADI = householdADI[monteCarloResult]["ADI"]
							DI = householdADI[monteCarloResult]["DI"]
							maskIncome = np.logical_and(np.greater_equal(income,incomeDF.loc[incomeBracket].lo),\
								np.less_equal(income,incomeDF.loc[incomeBracket].hi))
							# 	Create mask for ADI and DI incomes in this bracket below zero
							maskADI_HighRisk = np.logical_and(np.less_equal(ADI, 0), maskIncome)
							maskDI_HighRisk = np.logical_and(np.less_equal(DI, 0), maskIncome)
						
							#	Store masked household results (for sum) and mask (for count) to dictionary
							householdADI[monteCarloResult + incomeBracketColumns[incomeBracket]] = \
								householdADI[monteCarloResult].mask(~maskIncome)
							householdADI[monteCarloResult + incomeBracketColumns[incomeBracket] + "_Mask"] = \
								maskIncome			
							householdADI[monteCarloResult + incomeBracketColumns[incomeBracket] + "_MaskADI_HighRisk"] = \
								maskADI_HighRisk
							householdADI[monteCarloResult + incomeBracketColumns[incomeBracket] + "_MaskDI_HighRisk"] = \
								maskDI_HighRisk
					#	Now iterate through income brackets and add up household results for each bkgp
					for incomeBracket in range(incomeDF.shape[0]):
						#	In the future, come back to iterative stdev calculation
						#	-	http://datagenetics.com/blog/november22017/index.html
						#	Initialize dataframes for...
						#	-	ADI, DI, and Income for all households
						incomeBracketHouseholds = pd.DataFrame(columns = ['ADI','DI','Income'])
						incomeBracketCount = pd.Series()
						#	-	Counts of high, mid, and low risk households
						incomeBracketADI_HighRiskCount = pd.Series()
						incomeBracketDI_HighRiskCount = pd.Series()
						incomeBracketDIs = pd.Series()
						incomeBracketADIs = pd.Series()
						incomeBracketDIs = []
						incomeBracketADIs = []
						#	Iterate through households
						for monteCarloResult in monteCarloCsvFiles:
							#	Concatenate all households in a given income bracket
							incomeBracketHouseholds = pd.concat([incomeBracketHouseholds, \
								householdADI[monteCarloResult + incomeBracketColumns[incomeBracket]].dropna()], sort=False)

							unitDIs = incomeBracketHouseholds['DI'].tolist()
							unitADIs = incomeBracketHouseholds['ADI'].tolist()
							incomeBracketDIs.extend(unitDIs)
							incomeBracketADIs.extend(unitADIs)
							#	Add up counts for all households in the same income bracket
							incomeBracketCount = incomeBracketCount.add( \
								householdADI[monteCarloResult + incomeBracketColumns[incomeBracket] + "_Mask"],
								fill_value = False)
							#	Add up counts of all households below $0 DI and $0 ADI in the same income bracket
							incomeBracketADI_HighRiskCount = incomeBracketADI_HighRiskCount.add( \
								householdADI[monteCarloResult + incomeBracketColumns[incomeBracket] + "_MaskADI_HighRisk"],
								fill_value = False)
							#	Add up counts of all households...
							incomeBracketDI_HighRiskCount = incomeBracketDI_HighRiskCount.add( \
								householdADI[monteCarloResult + incomeBracketColumns[incomeBracket] + "_MaskDI_HighRisk"],
								fill_value = False)

						#	Aggregate counts into one dataframe - we can clean this up later
						householdCountsDF = pd.DataFrame()
						householdCountsDF["Total_Households"] = incomeBracketCount
						householdCountsDF["ADI_HighRisk"] = incomeBracketADI_HighRiskCount
						householdCountsDF["DI_HighRisk"] = incomeBracketDI_HighRiskCount
						householdCountsDF["newHighRiskHouseholds"] = incomeBracketADI_HighRiskCount - incomeBracketDI_HighRiskCount

						#	Rename columns to include income bracket
						incomeBracketHouseholds.columns = \
							[f + "_" + incomeBracketColumns[incomeBracket] for f in incomeBracketHouseholds.columns]
						householdCountsDF.columns = \
							[f + "_" + incomeBracketColumns[incomeBracket] for f in householdCountsDF.columns]

						# ~~~~~~~~~~~~~
						# Save sumStats
						# ~~~~~~~~~~~~~
						pathIncomeBracketResults = os.path.join(self.pathResults,resultName,str(bkgp))
						#	-	Add column for blockgroups
						if incomeBracket != 0: bkgpColumn = None
						else: bkgpColumn = bkgp
						#	-	Household ADI, DI, and Income
						saveSumStats(pathIncomeBracketResults, incomeBracketColumns[incomeBracket], \
							incomeBracketHouseholds, stdev = True, median = True, bkgp = bkgpColumn)
						#	-	Household high, mid, and low risk counts
						saveSumStats(pathIncomeBracketResults, incomeBracketColumns[incomeBracket]+"_Counts", \
							householdCountsDF, stdev = True, median = True, bkgp = bkgpColumn)

						# ~~~~~~~~~~~~~~~~~~~~~
						# Save DI, ADI, Results
						# ~~~~~~~~~~~~~~~~~~~~~
						#	Save all DI results for this block group in a pickle file - high storage usage
						# adiFile =os.path.join(pathIncomeBracketResults, "{}.csv".format(incomeBracketColumns[incomeBracket]+"_ADIs"))
						# with open(adiFile[:-4] + '.data', 'wb') as filehandle:
						#	pickle.dump(incomeBracketADIs, filehandle)
						#	Save all DI results for this block group in a pickle file - high storage usage
						#diFile = os.path.join(pathIncomeBracketResults, "{}.csv".format(incomeBracketColumns[incomeBracket]+"_DIs"))
						#with open(diFile[:-4] + '.data', 'wb') as filehandle:
						#	pickle.dump(incomeBracketADIs, filehandle)

				#	Parallelized function above by blockgroup
				print("Creating sumStats_householdADI files for blockgroups...")
#				Pool().map(sumStatsForHouseholdADI, self.resBldgs.bkgpGEOID.unique().tolist())
				for bkgp in self.resBldgs.bkgpGEOID.unique().tolist():
					sumStatsForHouseholdADI(bkgp)

				#	Initialize empty dataframe for all bkgps and income brackets
				sumStats_householdADI_DF = pd.DataFrame()
				sumStats_householdADI_DF_Counts = pd.DataFrame()
				#	Iterate through folders containing results for each blockgroup
				for bkgp in self.resBldgs.bkgpGEOID.unique():
					print(bkgp)
					pathBkgpResults = os.path.join(self.pathResults,resultName,"{}/".format(bkgp))
					#	Concatenate bkgp level results for all income brackets
					bkgpResultsDF = pd.DataFrame()
					bkgpCountsDF = pd.DataFrame()
					for incomeBracket in incomeBracketColumns:
						resultsDFRow = pd.read_csv(pathBkgpResults + incomeBracket + ".csv")
						countsDFRow = pd.read_csv(pathBkgpResults + incomeBracket + "_Counts.csv")
						bkgpResultsDF = pd.concat([bkgpResultsDF, resultsDFRow], axis=1)
						bkgpCountsDF = pd.concat([bkgpCountsDF, countsDFRow], axis=1)
						# Here is where to pull data to test distribution of Equity Results - Uniform/Normal etc...
					#	Append results for each blockgroup
					sumStats_householdADI_DF = sumStats_householdADI_DF.append(bkgpResultsDF)
					sumStats_householdADI_DF_Counts = sumStats_householdADI_DF_Counts.append(bkgpCountsDF)

				#	Save
				sumStats_householdADI_DF.to_csv(self.pathResults + "sumStats_householdADI.csv", index=False)
				sumStats_householdADI_DF_Counts.to_csv(self.pathResults + "sumStats_householdADI_Counts.csv", index=False)


	def normalizeHousingUnits(self, buildingUnits, bkgpCorelogicTotalUnits, bkgpCensusTotalUnits):	
		"""
		Normalize the count of housing units in each building indicated by Corelogic data to that indicated by Census data.

		kwargs:
		- buildingUnits: A list of the number of housing units in all building
		- bkgpCorelogicTotalUnits: Total integer number of housing units in block group indicated by summed Corelogic Units
		- bkgpCensusTotalUnits: Total integer number of housing units in block group indicated by Census data

		returns:
		- revisedUnits: A normalized list of the number of housing units in all building

		"""
		#	If census units > corelogic units, add units from buildings weighted by existing units
		#	Correct for small errors in weighted addition by randomly adding buildings to units
		if bkgpCensusTotalUnits > bkgpCorelogicTotalUnits:
			missingUnits = bkgpCensusTotalUnits - bkgpCorelogicTotalUnits
			unitFractions = np.divide(buildingUnits, bkgpCorelogicTotalUnits)
			additionalUnits = np.round(np.multiply(missingUnits, unitFractions))
			revisedUnits = np.add(buildingUnits, additionalUnits).astype(int)
		#	If census units < corelogic units, subtract units from buildings weighted by existing units
		#	Correct for small errors in weighted subtraction by randomly adding buildings to units
		if bkgpCensusTotalUnits < bkgpCorelogicTotalUnits:
			extraUnits = bkgpCorelogicTotalUnits - bkgpCensusTotalUnits
			unitFractions = np.divide(buildingUnits, bkgpCorelogicTotalUnits)
			subtractedUnits = np.round(np.multiply(extraUnits, unitFractions))
			revisedUnits = np.subtract(buildingUnits, subtractedUnits).astype(int)
		#	If census units == corelogic units, no change in building units
		if bkgpCensusTotalUnits == bkgpCorelogicTotalUnits:
			revisedUnits = buildingUnits
		return revisedUnits


	def correctHousingUnits(self, normalizedUnits, bkgpCorelogicTotalUnits, bkgpCensusTotalUnits):
		"""
		Correct the rounding errors from normalization of housing units in each building.
		Randomly removes units from buildings until the sum of housing units in a given block group equals that indicated by the Census.

		kwargs:
		- normalizedUnits: A normalized list of the number of housing units in all building
		- bkgpCorelogicTotalUnits: Total integer number of housing units in block group indicated by summed Corelogic Units
		- bkgpCensusTotalUnits: Total integer number of housing units in block group indicated by Census data

		returns:
		- correctedUnits: A corrected list of the number of housing units in all building

		"""
		#	While there are more census units than normalized units, add units to random buildings
		totalNormalizedUnits = np.sum(normalizedUnits)
		if bkgpCensusTotalUnits > totalNormalizedUnits:
			while bkgpCensusTotalUnits > totalNormalizedUnits:
				#	Choose random building and add a unit
				index = np.random.randint(0, np.shape(normalizedUnits)[0], 1)[0]
				normalizedUnits[index] = normalizedUnits[index] + 1
				totalNormalizedUnits = np.sum(normalizedUnits)
		#	While there are less census units than normalized units, subtract units from random buildings
		if bkgpCensusTotalUnits < totalNormalizedUnits:
			while bkgpCensusTotalUnits < totalNormalizedUnits:
				#	Choose random building and subtract a unit
				index = np.random.randint(0, np.shape(normalizedUnits)[0], 1)[0]
				if normalizedUnits[index] > 0:
					normalizedUnits[index] = normalizedUnits[index] - 1
				else:
					pass
				totalNormalizedUnits = np.sum(normalizedUnits)
		correctedUnits = normalizedUnits
		return correctedUnits


	def distributeParcelUnits(self):
		"""

		Distributes the housing units on a given parcel ("nResUnits") amongst buildings on the parcel.
		Units are distributed fractionally by total floor area of the buildings ("floorArea").

		Returns:
		- A list of building housing units for all buildings in the given dataframe

		"""
		nBldgUnits = []
		#	Iterate through unique APNs
		for APN in self.resBldgs.APN.unique(): 
			#	Get slice of buildings in each parcel
			parcelBldgs = self.resBldgs.loc[self.resBldgs.APN == APN]
			#	Get number of units in each building on the parcel
			if parcelBldgs.shape[0] == 1:
				# If a parcel has only one building, assign it the number of units in that parcel
				nBldgUnits = nBldgUnits + parcelBldgs["nResUnits"].values.tolist()
			else:
				# If a parcel has only multiple buildings, assign it a fraction of units in that parcel based on floor Area
				floorAreaFraction = np.divide(parcelBldgs["floorArea"], parcelBldgs["floorArea"].sum())
				nBldgUnits = nBldgUnits + \
				np.round(np.multiply(parcelBldgs["nResUnits"], floorAreaFraction)).values.tolist()
		return [int(units) for units in nBldgUnits]


	def getUnitFloors(self, nUnits, nFloors):
		"""
		Assigns the floor number to each housing unit in a given building from bottom floor upwards.

		kwargs:
		- nUnits: The integer number of housing units in a given building
		- nFloors: The integer number of floors in a given building

		returns:
		- floorList: A list which contains the floor assignment of the housing units
		
		"""
		if nUnits == 1: return [1]
		else:
			# Create a list of floors to iterate over and assign to units
			buildingFloors = [i for i in range(1,nFloors+1)]
			while nUnits > len(buildingFloors):
				buildingFloors.extend(buildingFloors)
			iteration = 1
			floorIndex = 0
			floorList = []
			# Iterate through units and assign floor value
			while iteration <= nUnits:
				floor = buildingFloors[floorIndex]
				floorList.append(floor)
				iteration += 1
				floorIndex += 1
		return floorList


	def getRandomUnitIncome(self, idx, incomeDF):
		"""
		Generates a random housing unit total income value between two given bounds.

		Kwargs:
		- idx: The row of incomeDF corresponding to where income bracket bounds are stored
		- incomeDF: Dataframe defined in runModel() which holds the upper and lower bounds of each income bracket

		"""
		return np.random.uniform(low = incomeDF[1].loc[idx], high = incomeDF[2].loc[idx])


	def getDiscretionaryIncome(self, unitIncome):
		"""
		Generates a discretionary income value given a household income value by running linearInterpolation()

		Kwargs:
		- unitIncome: Integer value of household income

		"""
		return linearInterpolation(unitIncome, self.discretionaryIncomeDF['incomeBeforeTaxes'].values, self.discretionaryIncomeDF['regionalDisIncome'].values)


	def runADI(self, unitFloors, unitOwnership, unitDiscretionaryIncomes, bldgContentAALs, bldgStructureAALs, bldgADC):
		"""
		Runs the risk-adjusted discretionary income (ADI) model by calling calculateAAI().
		Calculates ADI based on properties: floor #, renter v owner, discretionary income, AAL, and displacement cost

		kwargs:
		- unitsFloors: list of floor assignments across housing units
		- unitOwnership: list of renter v owner status across housing units
		- uniDiscretionaryIncome: list of discretionary income across housing units
		- bldgContentAALs: list of building content AALs across monte carlo iterations
		- bldgStructureAALs: list of structure & content AALs across monte carlo iterations
		- bldgADC: list of displacement costs across monte carlo iterations

		returns:
		- unitADIs: list of ADI across given housing units

		"""
		numFirstFloorUnits = [unitFloors.count(1)] * len(unitFloors)
		numOwners = [unitOwnership.count(2)] * len(unitFloors)
		bldgADC = [bldgADC] * len(unitFloors)
		unitData = [unitFloors, unitOwnership, unitDiscretionaryIncomes, bldgContentAALs, bldgStructureAALs, numOwners, numFirstFloorUnits, bldgADC]
		unitData = list(zip(*unitData))
		unitADIs = []
		for unit in unitData:
			unitADI = self.calculateADI(unit)
			if np.isnan(unitADI) is False: unitADI = int(unitADI)
			unitADIs.append(unitADI)
		return unitADIs


	def calculateADI(self, unit):
		"""
		Calculates ADI based on housing unit properties: floor #, renter v owner, discretionary income, AAL, and displacement cost

		kwargs:
		- unit: a list containing all housing unit properties of a single unit needed to calculate ADI

		returns:
		- unitADI: Integer value of risk-adjusted discretionary income for given housing unit

		"""
		# define variables
		floor = unit[0]
		ownership = unit[1]
		discretionaryIncome = unit[2]
		contentAAL = unit[3]
		structureAAL = unit[4]
		numOwners = unit[5]
		numFirstFloorUnits = unit[6]
		ADC = unit[7]
		# If ownership is vacant, unit gets NaN for ADI
		if ownership == 0: 
			unitADI = np.nan
		# If ownership is renter, unit gets no structure damages
		elif ownership == 1:
			if floor == 1:
				unitADI = discretionaryIncome - ((contentAAL * self.fractionUncompensated)/numFirstFloorUnits) - ADC
			else:
				unitADI = discretionaryIncome - ADC
		# If ownership is owner, responsible for structure damages
		elif ownership == 2:
			if floor == 1:
				unitADI = discretionaryIncome - ((contentAAL * self.fractionUncompensated)/numFirstFloorUnits) - ((structureAAL * self.fractionUncompensated)/numOwners) - ADC
			else:
				unitADI = discretionaryIncome - ((structureAAL * self.fractionUncompensated)/numOwners) - ADC
		return unitADI

	def calculateDisplacementCost(self, displacementTime):
		"""
		Calculates household displacement cost based on household displacement time for a given storm event.

		kwargs:
		- displacement Time: Household displacement time in days for a given storm event.

		Returns:
		- displacementCost: Household displacement cost in days for a given storm event.

		"""
		if displacementTime >= 365:
			displacementCost = 24900
		elif displacementTime > 0 and displacementTime < 365:
			displacementCost = 500
			displacementCost += ((displacementTime/30) * 2000)
		else:
			displacementCost = 0
		return int(displacementCost)


	def calculateDisplacementTime(self, struDmg):
		"""
		Calculates household displacement cost based on household displacement time for a given storm event.

		kwargs:
		- struDmg: building structural damage in percent for a given storm event

		Returns:
		- displacement Time: Household displacement time in days for a given storm event.
		"""

		if struDmg < 10:
			displacementTime = 0
		if struDmg == 10:
			displacementTime = 30
		if struDmg > 10:
			displacementTime = ((struDmg - 10) * 8) + 30
			if displacementTime > 365:
				displacementTime = 365
		return int(displacementTime)

# ~~~~~~~~~~~~~~~~~~~~~~
# Execute Equity Model
# ~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

	start = time.time()

	#	Equity model
	equity = EquityModel(

		pathBuildingDF = "output/exampleDataPrepResults.csv",
		simName = "exampleEquityResults",
		saveResults = ["monteCarloADI", "sumStats_householdADI"],
		years=[2020, 2030, 2040, 2050, 2060],
		fractionUncompensated = 0.372,
		exposureResultsPath = "output/exampleExposureResults/monteCarloAAL/",
		structureDamagesPath = "output/exampleExposureResults/percDmgStru/")

	equity.runModel()
	equity.aggregateResults()

	end = time.time()

	print("{} seconds".format((end - start)))

else:
	pass
