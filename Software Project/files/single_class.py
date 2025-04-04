import matplotlib.image as img
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import feature_extract as ext
import shutil
def single_class(k):
    feature_vectors,classifications=ext.dataset_to_array2()
    class_centroids=ext.centroids(feature_vectors,classifications)
    all_similarities=[]
    odd_images=[]
    for img_name, features in feature_vectors.items():
        similarity = cosine_similarity([features], [class_centroids["1"]])[0][0]
        all_similarities.append([similarity,img_name])
    all_similarities.sort()
    for i in range(0,k):
        odd_images.append(all_similarities[i])
    for odd_image in odd_images:
        shutil.copy("../dataset/"+odd_image[1],"../output")
    return odd_images