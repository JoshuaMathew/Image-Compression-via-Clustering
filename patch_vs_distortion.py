import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# load image
img = cv2.imread('car.jpg')
# convert image to gray scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# select 512 * 1024 subset of image
img = img[50:562, 200:1224]
# get image shape
M = img.shape[0]
N = img.shape[1]

P_values = [2, 4, 8]
distortions = []

def reconstruct_image(patches, M, N, P):
    # function for reconstructing image from patches
    # M and N are size of image
    # P is patch size where each patch is P x P

    # initialize empty array which will be converted to reconstructed image
    empty_img = np.zeros((M, N), dtype=np.uint8)
    count = 0
    for x in range(0, img.shape[0], P):
        for y in range(0, img.shape[1], P):
            empty_img[x:x + P, y:y + P] = patches[count].reshape((P, P))
            count += 1

    return empty_img

def get_distortion(im1, im2, M, N):
    # calculate distortion between original image1 (original) and image2
    distortion = ((im1 - im2) ** 2).sum() / (M * N)
    return distortion

for P in P_values:
    R = 0.08  # rate
    C = int(2 ** (R * (P ** 2)))  # number of clusters
    # get patches
    patches = [img[x:x + P, y:y + P].flatten() for x in range(0, img.shape[0], P) for y in range(0, img.shape[1], P)]
    patches = np.array(patches)  # convert to array

    # perform k means clustering to get cluster labels and cluster centers
    kmeans = KMeans(n_clusters=C, random_state=0).fit(patches)

    # get new patches based on k means clustering
    new_patches = []  # initialize list of new patches

    for i in range(len(patches)):
        # assign each patch to equal the representative patch of its cluster
        new_patches.append(np.array(kmeans.cluster_centers_[kmeans.labels_[i]]))

    new_patches = np.array(new_patches)  # convert to array

    # get quantized image by reconstructing the quantized patches
    quantized_img = reconstruct_image(new_patches, M, N, P)

    # calc distortion
    distortion = get_distortion(img, quantized_img, M, N)
    distortions.append(distortion)


plt.plot(P_values, distortions)
plt.title("Patch Size vs. Distortion")
plt.xlabel("P")
plt.ylabel("Distortion")
plt.show()