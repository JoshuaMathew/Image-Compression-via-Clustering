******************************************************************************************
VECTOR QUANTIZATION VIA K-MEANS CLUSTERING:

This project demonstrates the use of k mean clustering for image compression

******************************************************************************************

We first start with an image of a car. The image is converted to grayscale and a 512 x 1024 subsection is shown below:

![Original Car IMAGE](https://github.com/JoshuaMathew/Image-Compression-via-Clustering/blob/main/car_original.JPG)

******************************************************************************************

After preprocessing the image, we define a "patch" in the image with dimensions PxP, where P=2. This value of will work well for the dimensions of the above M X N image since they are also powers of 2.

In total, there are (MxN)/P^2 patches. We then cluster these patches into 16 clusters using k-means clustering algorithm. We use the Kmeans function in scikit-learn for this. 
The value of the number of clusters (C) is chosen by choosing the value of the rate (R) as 1. This means, each pixel is represented by 2*P*R bits. The total number of clusters    C=2(RP^2). 

When the k-means algorithm is applied, each patch in the image is assigned to a particular cluster. Each patch is replaced by the values of the cluster to which it belongs. 
We observe that as the number of patches decreases the quality of the compressed image deteriorates(at cluster=1, the image is totally destroyed). 

The whole image is now represented using only those values in the cluster. Hence, we will be able to represent the whole image in fewer bits than before effectively compressing the image. However, this is a lossy image compression mehtod, decompressing the compressed image will create an image that is not identical to the original.

Below is part of the original image compared to the compressed version:

![Original vs. Qunatized IMAGE](https://github.com/JoshuaMathew/Image-Compression-via-Clustering/blob/main/orig_vs_quant.JPG)

******************************************************************************************

In this part, we vary the rate (R) value from 0 to 1. As we vary R, the number of clusters C also changes because C=2(R*P^2). For each value of R (and thereby C) we do the clustering and the Vector Quantization again. For each of these R we calculate the distortion as the mean squared error between the original image and the quantized image.

![Rate vs Distortion](https://github.com/JoshuaMathew/Image-Compression-via-Clustering/blob/main/rate_vs_distortion.JPG)

We observe that as we increase the rate R, the distortion is monotonically decreasing. This happens because as we increase R and keep patch size constant, the number of clusters C increase.
