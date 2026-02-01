from detection import LayoutInference

engine = LayoutInference()
# This will return the list of bounding boxes
engine.run("mef_2.tiff")
engine.run("goc_4.tiff") 
engine.run("pp_gov_4.tiff")

