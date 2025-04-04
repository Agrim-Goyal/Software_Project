import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import feature_extract as ext
import os
import shutil
def classify(k):
    X,classifications=ext.dataset_to_array2()
    image_names = list(X.keys())
    vectors = np.array(list(X.values()))
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(vectors)
    cluster_assignments = kmeans.labels_
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(vectors_2d[:, 0], vectors_2d[:, 1], 
                           c=cluster_assignments, cmap='viridis', s=50, edgecolor='k')
    plt.title('K-Means Clustering Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.colorbar(scatter, label='Cluster ID')
    plt.savefig("../figures/plot.jpg")
    classes=[]
    for i in range(0,k):
        classes.append([])
    for img_name, cluster_id in zip(image_names, cluster_assignments):
        classes[cluster_id].append(img_name)
    for i,imgs in enumerate(classes):
        if not os.path.exists("../output/"+str(i)):
            os.mkdir("../output/"+str(i))
        for j in imgs:
            shutil.copy("../dataset/"+j,"../output/"+str(i))
    return classes