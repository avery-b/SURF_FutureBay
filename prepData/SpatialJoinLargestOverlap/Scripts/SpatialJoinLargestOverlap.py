'''----------------------------------------------------------------------------------
 Tool Name:   Spatial Join Largest Overlap
 Source Name: SpatialJoinLargestOverlap.py
 Version:     ArcGIS 10.1
 Author:      Esri, Inc.
 Required Arguments:
              Target Features (Feature Layer)
              Join Features (Feature Layer)
              Output Feature Class (Feature Class)
 Optional Arguments:
              Keep All (True|False)
              Spatial Relationship (String)

 Description: Joins attributes from one feature class to another based on the spatial
              relationship between the two. The target features and the
              attributes from the join features are written to the output feature
              class. This tool presents a new spatial relationship, Largest Overlap,
              where a target feature is joined to the join feature with the largest
              area or length of overlap.
----------------------------------------------------------------------------------'''

# Import system modules
import arcpy
import os

arcpy.env.overwriteOutput = True

# Main function, all functions run in SpatialJoinOverlapsCrossings
def SpatialJoinLargestOverlap(target_features, join_features, out_fc, keep_all, spatial_rel):
    if spatial_rel == "largest_overlap":
        # Calculate intersection between Target Feature and Join Features
        intersect = arcpy.analysis.Intersect([target_features, join_features], "in_memory/intersect", "ONLY_FID")
        # Save copy of intersect file for debugging
        arcpy.CopyFeatures_management(intersect, "C:/Users/IanAvery/Desktop/highTidePrototype/santaClaraCountyAnalysis.gdb/intersect")
        # Find which Join Feature has the largest overlap with each Target Feature
        # Need to know the Target Features shape type, to know to read the SHAPE_AREA oR SHAPE_LENGTH property
        geom = "AREA" if arcpy.Describe(target_features).shapeType.lower() == "polygon" and arcpy.Describe(join_features).shapeType.lower() == "polygon" else "LENGTH"
        fields = ["FID_{0}".format(os.path.splitext(os.path.basename(target_features))[0]),
                  "FID_{0}".format(os.path.splitext(os.path.basename(join_features))[0]),
                  "SHAPE@{0}".format(geom)]
        overlap_dict = {}

        with arcpy.da.SearchCursor(intersect, fields) as scur:
            for row in scur:
                try:
                    if row[2] > overlap_dict[row[0]][1]:
                        overlap_dict[row[0]] = [row[1], row[2]]
                except:
                    overlap_dict[row[0]] = [row[1], row[2]]

        # Copy the target features and write the largest overlap join feature ID to each record
        # Set up all fields from the target features + ORIG_FID
        fieldmappings = arcpy.FieldMappings()
        fieldmappings.addTable(target_features)
        fieldmap = arcpy.FieldMap()
        fieldmap.addInputField(target_features, arcpy.Describe(target_features).OIDFieldName)
        fld = fieldmap.outputField
        fld.type, fld.name, fld.aliasName = "LONG", "ORIG_FID", "ORIG_FID"
        fieldmap.outputField = fld
        fieldmappings.addFieldMap(fieldmap)
        # Perform the copy
        arcpy.conversion.FeatureClassToFeatureClass(target_features, os.path.dirname(out_fc), os.path.basename(out_fc), "", fieldmappings)
        # Add a new field JOIN_FID to contain the fid of the join feature with the largest overlap
        arcpy.management.AddField(out_fc, "JOIN_FID", "LONG")
        # Calculate the JOIN_FID field
        with arcpy.da.UpdateCursor(out_fc, ["ORIG_FID", "JOIN_FID"]) as ucur:
            for row in ucur:
                try:
                    row[1] = overlap_dict[row[0]][0]
                    ucur.updateRow(row)
                except:
                    if not keep_all:
                        ucur.deleteRow()
        # Join all attributes from the join features to the output
        joinfields = [x.name for x in arcpy.ListFields(join_features) if not x.required]
        arcpy.management.JoinField(out_fc, "JOIN_FID", join_features, arcpy.Describe(join_features).OIDFieldName, joinfields)


# Run the script
if __name__ == '__main__':
    # Get Parameters
    target_features = arcpy.GetParameterAsText(0)
    join_features = arcpy.GetParameterAsText(1)
    out_fc = arcpy.GetParameterAsText(2)
    keep_all = arcpy.GetParameter(3)
    spatial_rel = arcpy.GetParameterAsText(4).lower()

    SpatialJoinLargestOverlap(target_features, join_features, out_fc, keep_all, spatial_rel)
    print "finished"