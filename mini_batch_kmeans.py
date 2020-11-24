import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
import time

# load image
img = cv2.imread('car.jpg')
# convert image to grayscale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# select 256 * 512 subset of image
img = img[70:326, 200:712]
# get image shape
M = img.shape[0]
N = img.shape[1]

# display image
#cv2.imshow('image', img)
#cv2.waitKey(0)

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

# perform k means clustering
start1 = time.time() # track runtime
kmeans = KMeans(n_clusters=C, random_state=0).fit(patches)
end1 = time.time()
# perform mini-batch k means clustering
start2 = time.time() # track runtime
miniKmeans = MiniBatchKMeans(n_clusters=C).fit(patches)
end2 = time.time()
# get new patches based on k means clustering
new_patches = []  # initialize list of new patches
new_patches2 = []

# quantization for k means
for i in range(len(patches)):
    # assign each patch to equal the representative patch of its cluster
    new_patches.append(np.array(kmeans.cluster_centers_[kmeans.labels_[i]]))
new_patches = np.array(new_patches)  # convert to array

# quantization for mini k means
for i in range(len(patches)):
    # assign each patch to equal the representative patch from the training image
    new_patches2.append(np.array(miniKmeans.cluster_centers_[miniKmeans.labels_[i]]))
new_patches2 = np.array(new_patches2)  # convert to array

# get quantized k means image by reconstructing the quantized patches
quantized_img = reconstruct_image(new_patches, M, N, P)
# get quantized mini k means image by reconstructing the quantized patches
quantized_img2 = reconstruct_image(new_patches2, M, N, P)

# display quantized images
cv2.imshow('quantized_img', quantized_img)
cv2.imshow('quantized_img2', quantized_img2)
cv2.waitKey(0)

distortion1 = get_distortion(img, quantized_img, M, N)
distortion2 = get_distortion(img, quantized_img2, M, N)

print("The distortion for Kmeans is: " + str(distortion1))
print("The distortion for miniKmeans is: " + str(distortion2))
print("Run time for Kmeans is: " + str(end1-start1))
print("Run time for miniKmeans is: " + str(end2-start2))