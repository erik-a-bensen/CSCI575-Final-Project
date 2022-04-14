# import packages----------
import numpy as np
import matplotlib.pyplot as plt
import cv2
# --------------------------


def knn_segment(image_name, num_clusters, iterations):
# function: knn_segment
# inputs:
#       image: a string filename, make sure it is in the same folder as the project
#       num_clusters: the hyper-parameter number of clusters
#       iterations: number of iterations for the clustering process to stop at
# outputs as a tuple:
#       seg_image: the segmented image
#       square_ labels: an np array with the class number of each pixel
#### BEGIN FUNCTION ####
    # read in image
    image = cv2.imread(image_name)

    # change to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Reshaping the image into a 2D array of pixels and 3 color values (RGB)
    pixel_vals = image.reshape((-1, 3))

    # convert to float values
    pixel_vals = np.float32(pixel_vals)

    # criteria for the iterations to stop running
    criteria = (cv2.TERM_CRITERIA_MAX_ITER, iterations, .85)

    # then perform k-means clustering
    retval, labels, centers = cv2.kmeans(pixel_vals, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # convert data into 8-bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]

    # reshape data into the original image dimensions
    seg_image = segmented_data.reshape((image.shape))

    square_labels = np.reshape(labels, (seg_image.shape[0], seg_image.shape[1]))
    plt.imshow(seg_image)
    plt.show()

    return (seg_image,square_labels)

def compare_classes(cls, pix_list, labels):
# function: compare_classes
# inputs:
#       cls: a number denoting the "correct class" in this context
#       pix_list: the list of pixel locations we wish to compare to this class that should all be the same as "class"
# outputs as a tuple:
#       matches: an binary array of length len(pix_list) that shows the pixels in pix_list that match clas
#       accuracy: the percentage of ^^ that are correct matches
# the idea here is that all the pixels in pix_list we know should be ice or something, and we already know which
# class number corresponds to ice
#### BEGIN FUNCTION ####
    matches = []
    acc_count = 0
    for i in range(len(pix_list)):
        if labels[pix_list[i][0],pix_list[i][1]] == cls:
            matches.append(1)
            acc_count = acc_count + 1
        else:
            matches.append(0)

    accuracy = acc_count / len(pix_list)
    print(str(accuracy*100) + "% of the pixels identified match class " + str(cls))
    return (matches, accuracy)


# Example use of these functions:
mon, labs = knn_segment("1_SWIR_hist.png", 6, 100)
testpix = [[50,50], [50,50]]
compare_classes(3, testpix, labs)
