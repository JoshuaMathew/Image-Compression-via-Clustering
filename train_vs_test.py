import cv2
import numpy as np
from sklearn.cluster import KMeans

# load image
img = cv2.imread('car.jpg')
# convert image to gray scale
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# select 512 * 1024 subset of image
img = img[50:562, 200:1224]
# get image shape
M = img.shape[0]
N = img.shape[1]

# load test image
img2 = cv2.imread('car2.jpg')
# convert image to gray scale
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
# select 512 * 1024 subset of image
img2 = img2[80:592, 50:1074]

# display image
cv2.imshow('image', img2)
cv2.waitKey(0)

P = 2  # patch size
R = 1  # rate
C = int(2 ** (R*(P**2)))  # number of clusters
# get patches
patches = [img[x:x+P, y:y+P].flatten() for x in range(0, img.shape[0], P) for y in range(0, img.shape[1], P)]
patches = np.array(patches)  # convert to array

patches2 = [img2[x:x+P, y:y+P].flatten() for x in range(0, img2.shape[0], P) for y in range(0, img2.shape[1], P)]
patches2 = np.array(patches2)  # convert to array

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

# perform k means clustering on train image
kmeans = KMeans(n_clusters=C, random_state=0).fit(patches)
# perform k means clustering on test image
kmeans2 = KMeans(n_clusters=C, random_state=0).fit(patches2)

# get new patches based on k means clustering
new_patches = []  # initialize list of new patches
new_patches2 = []

# quantization for training image
for i in range(len(patches)):
    # assign each patch to equal the representative patch of its cluster
    new_patches.append(np.array(kmeans.cluster_centers_[kmeans.labels_[i]]))
new_patches = np.array(new_patches)  # convert to array

# quantization for testing image
for i in range(len(patches)):
    # assign each patch to equal the representative patch from the training image
    new_patches2.append(np.array(kmeans.cluster_centers_[kmeans2.labels_[i]]))
new_patches2 = np.array(new_patches2)  # convert to array

# get quantized image by reconstructing the quantized patches
quantized_img = reconstruct_image(new_patches, M, N, P)
# get quantized image by reconstructing the quantized patches
quantized_img2 = reconstruct_image(new_patches2, M, N, P)

# display quantized image
cv2.imshow('img2', img2)
cv2.imshow('quantized_img2', quantized_img2)
cv2.waitKey(0)

distortion1 = get_distortion(img, quantized_img, M, N)
distortion2 = get_distortion(img2, quantized_img2, M, N)

print("The distortion for the training image is: " + str(distortion1))
print("The distortion for the test image is: " + str(distortion2))
