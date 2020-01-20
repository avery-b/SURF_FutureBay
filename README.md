# SURF - Stanford Urban Risk Framework

A python framework for flood risk analysis. It was designed to 1) project flooding damage at the building-scale due to coastal flooding and sea level rise 2) project financial impacts to households across incomes and demographics, and 3) account for uncertainty through monte carlo methodology.

Co-developed by Avery Bick (MS Environmental Engineering '18) and Adrian S. Santiago Tate (MS Geophysics '20) from mid 2018 to late 2019. This code supported the publication 'Rising seas, rising inequity? Communities at risk in the San Francisco Bay Area and implications for adaptation policy.' It began in a project learning course run by Stanford Future Bay Initiative in which we co-developed studies of sea level rise risk in the Bay Area with local stakeholders. Avery, Adrian, and another alum Arnav (MS Geophysics '19) went on to develop a flood risk data company, HighTide Intelligence.

Contact:
Avery@hightide.ai

# How to run SURF
## dataPrep.py
Joins flood rasters and tax assessor data to building footprints. Prepares data for Exposure.py. 

This module runs on Python 2 and requires a computer with an ArcGIS license.

1. Install requirements
- pip install -r dataPrepRequirements.txt on the system Python 2 installation
- Add the arcpy location to the computers PYTHONPATH along with the existing Python 2 library and script locations in Windows System Environment Variables, i.e. Path = C:\Python27\Lib; C:\Python27\ArcGIS10.5; C:\Python27\Scripts

2. Starting at if __name__ == '__main__', set your file paths, each file is described in the code and links to an example. 

3. Set names of specific fields in the building polygon (Area and Height in feet) and the tax parcel polygons (residential unit count, APN, property use code).

4. Within the assignResUnitCount(): function, set the detault values for # of residential units per each residential property use code. If the data for a given parcel does not contain the number of residential units on a given property, the code assign a default to buildings from this function.

5. Set path for floor count classifier training data
- Included training data is for San Mateo County, CA
- For other areas, create a csv of the same format as floorClassifierTrainingData.csv where each row provides a building's height, area, and property use code (see prepareFloorClassifierTrainingData.py).
- This data is applied in a random forest classifier in dataPrep to predict the number of floors in buildings where this value is unknown (see floorClassifier.py).
- Alternative is to use an assumed height for each floor

6. Run python dataPrep.py in Windows Powershell or dataprep.py in the Python shell

7. The output building CSV contains depth of flooding for each building and each flood (see Hazard.py) and is the input for the remainder of SURF.

## Exposure.py
Calculates building damages in each storm and average annualized losses for a given time period over multiple sea level rise projections. Performs Monte Carlo analysis to determine mean and standard deviation of damages.

This module runs on Python 3 and requires no licenses.

1. Install requirements
- pip install -r surfRequirements.txt on system Python 3, or create an environment with these dependencies

2. Set attribute 'pathBuildingDF' to output building csv file from dataPrep.py

3. Add a name for your model run (attribute 'simName') - a folder with this name storing results will be created in the same directory as your pathBuildingDF.

4. Set the attribute 'years' in decadal format. The AAL will be calculated over this time frame (i.e. [2020, 2030, 2040]).

5. In surfTools.py > calculateSLRAAL() set which type of trapezoidal integration to use when calculating AAL.

6. Set the filepaths for the building properties lookup table, the hazard raster lookup table, the replacement cost database, and the depth-damage curve databases.

7. Add a replacement cost factor for your region (see U.S. Bureau of Economic Analysis regional price parities).

8. Set your saveResults options. To run equity.py next you need ["monteCarloAAL", "percDmgStru"]. "SummaryStatsAAL" is also a useful setting and saves AAL predicted per building.

9. Set the number of iterations (attribute 'nIterations') of Monte Carlo to run. The more iterations, the longer the model can move towards a stable mean of AAL estimates. We find that 1,000 iterations is suitable.

10. Set attributes for exposure.createBuildings() class:
- nBuildings: Set as None to run all buildings in your pathBuildingDF, set to an integer to test Exposure with less buildings
- implementation: "serial" or "parallel" computing
- nodes: if using parallel computing, set number of cores to use. Using all available cores will slow your system significantly.
- overwrite: Set to True to overwrite previous results with same simName, set to False to skip results which have already been created

11. Run python Exposure.py

## Equity.py
Calculates impacts of building damages on residents in various income groups at the block group scale. Applies Monte Carlo analysis to determine mean and standard deviation of damages to households.

This module runs on Python 3 and requires no licenses.

1. Set attribute 'pathBuildingDF' to output building csv file from dataPrep.py

2. Add a name for your model run (see attribute 'simName') - a folder with this name storing results will be created in the same directory as your pathBuildingDF.

3. Set the 'exposureResultsPath' to the folder containing the "monteCarloAAL" results from the Exposure model.

4. Set the 'structureDamagesPath' to the folder containing the "percDmgStru" results from the Exposure model.

5. Set the years in decadal format that your analysis is focused on. Annual income losses will be calculated over this time frame (see attribute 'years').


6. Choose which sea level rise projection you are using (see attribute 'pathProjections')

7. Run python Equity.py