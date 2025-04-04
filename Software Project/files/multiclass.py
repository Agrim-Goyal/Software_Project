import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import feature_extract as ext
import shutil
import os
def multiclass():
    better_suited_images={}
    feature_vectors,classifications=ext.dataset_to_array()
    class_centroids=ext.centroids(feature_vectors,classifications)
    for img_name, features in feature_vectors.items():
        max_similarity = -1
        best_class = None
        original_class = next(cls for cls, imgs in classifications.items() if img_name in imgs)
        for target_class, centroid in class_centroids.items():    
            similarity = cosine_similarity([features], [centroid])[0][0]
            if similarity > max_similarity:
                max_similarity = similarity
                best_class = target_class
        if best_class!=original_class:
            better_suited_images[img_name] = (best_class,original_class)
    shutil.copytree("../dataset","../output",dirs_exist_ok=True)
    for img_name,(best_class,original_class)in better_suited_images.items():
        os.rename("../output/"+original_class+"/"+img_name,"../output/"+best_class+"/"+img_name)
    return better_suited_images