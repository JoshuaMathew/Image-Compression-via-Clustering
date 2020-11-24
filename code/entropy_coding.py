import cv2
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter

# load image
img = cv2.imread('car.jpg')
# convert image to gray scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# select 512 * 1024 subset of image
img = img[50:562, 200:1224]
# get image shape
M = img.shape[0]
N = img.shape[1]

# display image
# cv2.imshow('image', img[200:400, 600:800])
# cv2.waitKey(0)

P = 2  # patch size
R = 1  # rate
C = int(2 ** (R*(P**2)))  # number of clusters
# get patches
patches = [img[x:x+P, y:y+P].flatten() for x in range(0, img.shape[0], P) for y in range(0, img.shape[1], P)]
patches = np.array(patches)  # convert to array

def reconstruct_image(patches, M, N, P):
    # function for reconstructing image from patches
    # M and N are size of image
    # P is patch size where each patch is P x P

    # initialize empty array which will be converted to reconstructed image
    empty_img = np.zeros((M,N), dtype=np.uint8)
    count = 0
    for x in range(0, img.shape[0], P):
        for y in range(0,img.shape[1],P):
            empty_img[x:x + P, y:y + P] = patches[count].reshape((P,P))
            count += 1

    return empty_img

def get_distortion(im1, im2, M, N):
    # calculate distortion between original image1 (original) and image2
    distortion = ((im1 - im2) ** 2).sum()/(M*N)
    return distortion

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


# count how many patches were assigned to each cluster
occurrences = Counter(kmeans.labels_)
# calculate encoding probabilities
probabilities = np.zeros(len(occurrences))
for i in occurrences:
    prob = occurrences[i] / ((M * N) / P ** 2)
    probabilities[i] = prob

# calculate the entropy of clusters
H = -(probabilities*np.log2(probabilities)).sum()
encoding_rate = H/P**2
print('The encoding rate is: ' + str(encoding_rate))

# display quantized image
cv2.imshow('img', img[200:400, 800:1000])
cv2.imshow('quantized_img', quantized_img[200:400, 800:1000])
cv2.waitKey(0)

