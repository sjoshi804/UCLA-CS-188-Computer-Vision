import os
import cv2
import numpy as np
import timeit, time
from sklearn import svm, cluster, preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans


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

#def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
    # this function will train a linear svm for every category (i.e. one vs all)
    # and then use the learned linear classifiers to predict the category of
    # every test image. every test feature will be evaluated with all 15 svms
    # and the most confident svm will "win". confidence, or distance from the
    # margin, is w*x + b where '*' is the inner product or dot product and w and
    # b are the learned hyperplane parameters.

    # train_features is an N x d matrix, where d is the dimensionality of
    # the feature representation and N the number of training features.
    # train_labels is an N x 1 array, where each entry is an integer 
    # indicating the ground truth category for each training image.
    # test_features is an M x d matrix, where d is the dimensionality of the
    # feature representation and M is the number of testing features.
    # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
    # will use SVMs with a Radial Basis Function (RBF) Kernel.
    # svm_lambda is a scalar, the value of the regularizer for the SVMs

    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test feature.
    #return predicted_categories


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

    #Initialize a SIFT Object, keypoints and descriptors lists
    descriptors = []
    sift = cv2.xfeatures2d.SIFT_create()
    for image in train_images:
        des = sift.detectAndCompute(image, None)[1]
        print(len(des))
        print((len(des[0])))
        for i in range(0, 10):
            descriptors.append(des[i])
      
    kmeans = KMeans(n_clusters=dict_size, random_state=0).fit(descriptors)
    return kmeans.cluster_centers_




#def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary

    # BOW is the new image representation, a normalized histogram
    #return Bow


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
    formatted_train_features = train_features
    formatted_test_features = test_features
    for size in (8,16,32):
        #Resize images      
        for i in range(0, len(train_features)):
            formatted_train_features[i] = np.ndarray.flatten(imresize(formatted_train_features[i], size))
        for i in range(0, len(test_features)):
            formatted_test_features[i] = np.ndarray.flatten(imresize(formatted_test_features[i], size))
        #Run classifier with different numbers of neighbours
        for num_neighbours in (1,3,6):
            start = time.time()
            accuracy = reportAccuracy(KNN_classifier(formatted_train_features, train_labels, formatted_test_features, num_neighbours),test_labels)
            end = time.time()
            run_time = end - start
        results += [accuracy, run_time]
    return results
    
