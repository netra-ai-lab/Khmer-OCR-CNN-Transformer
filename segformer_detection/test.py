from detection import LayoutInference

engine = LayoutInference()
# This will return the list of bounding boxes
results = engine.run("pp_gov_4.tiff") 
