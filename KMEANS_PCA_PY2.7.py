import numpy as np
import matplotlib.pyplot as plt
from skimage import io

from sklearn.utils import shuffle

from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA


flower = io.imread('flower.jpg')
flower = np.array(flower, dtype=np.float64) / 255
plt.imshow(flower)


w, h, d = original_shape = tuple(flower.shape)
assert d == 3
image_array = np.reshape(flower, (w*h,d))

image_sample = shuffle(image_array, random_state=42)[:1000]

n_colors = 64

kmeans = KMeans(n_clusters=n_colors, random_state=42).fit(image_sample)

labels = kmeans.predict(image_array)


# Reconstruct_image
def reconstruct_image(cluster_centers, labels, w, h):
    d = cluster_centers.shape[1]
    image = np.zeros((w, h, d))
    label_index = 0
    for i in range(w):
        for j in range(h):
            image[i][j] = cluster_centers[labels[label_index]]
            label_index += 1
    return image

final = reconstruct_image(cluster_centers = image_sample, labels = labels, w = w, h = h )

plt.imshow(final)


# Load dataset
iris = load_iris()
X = iris.data
y = iris.target
target_names = iris.target_names

# Run PCA

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)

print('Explained variance ratio from PCA: {}').format(pca.explained_variance_ratio_)

colors = ['red', 'blue', 'yellow']
plt.figure()
for color, i, target_name in zip(colors, [0, 1, 2], target_names):
    plt.scatter(X_r[y == i, 0], X_r[y == i, 1], color = color, alpha = .8, lw = 2,
                label = target_name)
plt.legend(loc='best', shadow = False)
plt.title('PCA of IRIS dataset')
plt.show()







