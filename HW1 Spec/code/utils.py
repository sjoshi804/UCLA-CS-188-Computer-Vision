import os
import cv2
import numpy as np
import timeit, time
from sklearn import svm, cluster, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, AgglomerativeClustering
import copy
from scipy.spatial import distance


def load_data():
    test_path = '../data/test/'
    train_path = '../data/train/'
    
    train_classes = sorted([dirname for dirname in os.listdir(train_path)], key=lambda s: s.upper())
    test_classes = sorted([dirname for dirname in os.listdir(test_path)], key=lambda s: s.upper())
    train_labels = []
    test_labels = []
    train_images = []
    test_images = []
    for i, label in enumerate(train_classes):
        for filename in os.listdir(train_path + label + '/'):
            image = cv2.imread(train_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            train_images.append(image)
            train_labels.append(i)
    for i, label in enumerate(test_classes):
        for filename in os.listdir(test_path + label + '/'):
            image = cv2.imread(test_path + label + '/' + filename, cv2.IMREAD_GRAYSCALE)
            test_images.append(image)
            test_labels.append(i)
            
    return train_images, test_images, train_labels, test_labels


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images

    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation and N is the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicating the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # num_neighbors is the number of neighbors for the KNN classifier

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    knn_classifier = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn_classifier.fit(train_features, train_labels)
    predicted_categories = knn_classifier.predict(test_features)
    return predicted_categories

def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an n x d matrix, where d is the dimensionality of
    # the feature representation.
    # train_labels is an n x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an m x d matrix, where d is the dimensionality of the
    # feature representation. (you can assume m=n unless you modified the 
    # starter code)
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # lambda is a scalar, the value of the regularizer for the SVMs
    # predicted_categories is an m x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    if is_linear:
        svclassifier = SVC(C=svm_lambda, kernel="linear", class_weight="balanced")
    else:
        svclassifier = SVC(C=svm_lambda, kernel="rbf", class_weight="balanced")
    svclassifier.fit(train_features, train_labels)
    return svclassifier.predict(test_features)


def imresize(input_image, target_size):
    dimension = (target_size, target_size)
    resized_image = cv2.resize(input_image, dimension)
    output_image = resized_image
    output_image = cv2.normalize(resized_image, output_image, -1, 1, cv2.NORM_MINMAX, cv2.CV_32F)
    return output_image



def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    num_correct = 0
    for i in range(0, len(true_labels)):
        if true_labels[i] == predicted_labels[i]:
            num_correct += 1
    accuracy = (num_correct / len(true_labels)) * 100
    return accuracy


def buildDict(train_images, dict_size, feature_type, clustering_type):
    # this function will sample descriptors from the training images,
    # cluster them, and then return the cluster centers.

    # train_images is a list of N images, represented as 2D arrays
    # dict_size is the size of the vocabulary,
    # feature_type is a string specifying the type of feature that we are interested in.
    # Valid values are "sift", "surf" and "orb"
    # clustering_type is one of "kmeans" or "hierarchical"

    # the output 'vocabulary' should be a list of length dict_size, with elements of size d, where d is the 
    # dimention of the feature. each row is a cluster centroid / visual word.

    #NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.

    #Initialize empty list for descriptors
    descriptors = []

    #Construct appropriate model object based on chosen feature detector
    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=10)
    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=10000)
    else:
        orb = cv2.ORB_create(nfeatures=10)

    #Detect descriptors using chosen method
    for image in train_images:
        if feature_type == "sift":
            des = sift.detectAndCompute(image, None)[1]
        elif feature_type == "surf":
            des = surf.detectAndCompute(image, None)[1]
        else:
            des = orb.detectAndCompute(image, None)[1]

        #Flatten by appending elements of des directly to descriptors
        if des is None:
            continue
        for element in des:
            descriptors.append(element)

    
    #Cluster according to chosen clustering method
    if clustering_type == "kmeans":
        kmeans = KMeans(n_clusters=dict_size, random_state=0).fit(descriptors)
        return kmeans.cluster_centers_
    else:
        hierarchical = AgglomerativeClustering(n_clusters=dict_size).fit(descriptors)

        #Get descriptors matching to each label
        des_mapping = []
        for label in range(0, dict_size):
            des_mapping.append([])
            for i in range(0, len(descriptors)):
                if hierarchical.labels_[i] == label:
                    des_mapping[label].append(descriptors[i])

        #Construct the 'average' descriptor for each label
        cluster_centers = []
        for label in range(0, dict_size):
            avg_descriptor = []
            for j in range(0, len(descriptors[0])):
                avg_value = 0
                for descriptor in des_mapping[label]:
                    avg_value += descriptor[j]
                avg_value /= len(des_mapping[label])
                avg_descriptor.append(avg_value)
            cluster_centers.append(avg_descriptor)

        #return clusters
        return cluster_centers
            

def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary
    #Construct appropriate model object based on chosen feature detector
    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=10)
        descriptors = sift.detectAndCompute(image, None)[1]
    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=10000)
        descriptors = surf.detectAndCompute(image, None)[1]
    else:
        orb = cv2.ORB_create(nfeatures=10)
        descriptors = orb.detectAndCompute(image, None)[1]

    # BOW is the new image representation, a normalized histogram
    bow = [0] * len(vocabulary)
    if descriptors is None:
        return bow

    dists = distance.cdist(descriptors, vocabulary)
    for dist_of_des in dists:
        bow[np.argmin(np.asarray(dist_of_des))] += 1
    
    #Normalize histogram
    bow = preprocessing.normalize([bow])

    return bow[0]


def tinyImages(train_features, test_features, train_labels, test_labels):
    # Resizes training images and flattens them to train a KNN classifier using the training labels
    # Classifies the resized and flattened testing images using the trained classifier
    # Returns the accuracy of the system, and the overall runtime (including resizing and classification)
    # Does so for 8x8, 16x16, and 32x32 images, with 1, 3 and 6 neighbors

    # train_features is a list of N images, represented as 2D arrays
    # test_features is a list of M images, represented as 2D arrays
    # train_labels is a list of N integers, containing the label values for the train set
    # test_labels is a list of M integers, containing the label values for the test set

    # classResult is a 18x1 array, containing accuracies and runtimes, in the following order:
    # accuracies and runtimes for 8x8 scales, 16x16 scales, 32x32 scales
    # [8x8 scale 1 neighbor accuracy, 8x8 scale 1 neighbor runtime, 8x8 scale 3 neighbor accuracy, 
    # 8x8 scale 3 neighbor runtime, ...]
    # Accuracies are a percentage, runtimes are in seconds
    #For different sizes of images
    results = []
    formatted_train_features = copy.deepcopy(train_features)
    formatted_test_features = copy.deepcopy(test_features)
    for size in (8,16,32):
        #Resize images 
        resize_start_time = time.time()     
        for i in range(0, len(train_features)):
            formatted_train_features[i] = np.ndarray.flatten(imresize(formatted_train_features[i], size))
        for i in range(0, len(test_features)):
            formatted_test_features[i] = np.ndarray.flatten(imresize(formatted_test_features[i], size))
        resize_end_time = time.time()
        #Run classifier with different numbers of neighbours
        for num_neighbours in (1,3,6):
            start = time.time()
            accuracy = reportAccuracy(KNN_classifier(formatted_train_features, train_labels, formatted_test_features, num_neighbours),test_labels)
            end = time.time()
            run_time = end - start + resize_end_time - resize_start_time
            results += [accuracy, run_time]
    return results
    
