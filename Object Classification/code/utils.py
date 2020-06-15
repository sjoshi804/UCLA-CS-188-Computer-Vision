import os
import cv2
import numpy as np
import timeit, time
from sklearn import neighbors, svm, cluster, preprocessing, metrics
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

    model = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, algorithm='kd_tree', metric='euclidean')
    model.fit(train_features, train_labels)
    predicted_categories = model.predict(test_features)
    return predicted_categories


def SVM_classifier(train_features, train_labels, test_features, is_linear, svm_lambda):
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
    if is_linear:
        classifier = svm.SVC(C=svm_lambda, kernel="linear", class_weight="balanced")
    else:
        classifier = svm.SVC(C=svm_lambda, kernel="rbf", class_weight="balanced")
    
    classifier.fit(train_features, train_labels)

    predicted_categories = classifier.predict(test_features)    
    
    return predicted_categories


def imresize(input_image, target_size):
    # resizes the input image, represented as a 2D array, to a new image of size [target_size, target_size]. 
    # Normalizes the output image to be zero-mean, and in the [-1, 1] range.
    output_image_unnormalized = cv2.resize(input_image, (target_size, target_size))
    output_image = output_image_unnormalized
    output_image = cv2.normalize(output_image_unnormalized, output_image, alpha=-1, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return output_image


def reportAccuracy(true_labels, predicted_labels):
    # generates and returns the accuracy of a model

    # true_labels is a N x 1 list, where each entry is an integer
    # and N is the size of the testing set.
    # predicted_labels is a N x 1 list, where each entry is an 
    # integer, and N is the size of the testing set. These labels 
    # were produced by your system.

    # accuracy is a scalar, defined in the spec (in %)
    accuracy = metrics.accuracy_score(true_labels, predicted_labels) * 100
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

    # NOTE: Should you run out of memory or have performance issues, feel free to limit the 
    # number of descriptors you store per image.

    features = []

    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=10)
    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=300)
    else:
        orb = cv2.ORB_create(nfeatures=10)

    for im in train_images:
        if feature_type == "sift":
            kp, descriptors = sift.detectAndCompute(im, None)
        elif feature_type == "surf":
            kp, descriptors = surf.detectAndCompute(im, None)
        else:
            kp, descriptors = orb.detectAndCompute(im, None)

        #Flatten by appending elements of des directly to descriptors
        for des in descriptors: 
            features.append(des)
    
    # kmeans clustering option
    if clustering_type == "kmeans":
        cluster_kmeans = cluster.KMeans(n_clusters=dict_size, random_state=0).fit(features)
        return cluster_kmeans.cluster_centers_
    # hierarchial clusterin option
    else:
        cluster_hierarchial = cluster.AgglomerativeClustering(n_clusters=dict_size).fit(features)
        
        # match feature to label
        feature_map = []
        for label in range(0, dict_size):
            feature_map.append([])
            for x in range(0, len(features)):
                if cluster_hierarchial.labels_[x] == label:
                    feature_map[label].append(features[x])

        # for each label gather "average" desciptor
        vocabulary = []
        for label in range(0, dict_size):
            feature_avg = []
            for y in range(0, len(features[0])):
                average = 0
                for x in feature_map[label]:
                    average += x[y]
                average /= len(feature_map[label])
                feature_avg.append(average)
            vocabulary.append(feature_avg)
    
    return vocabulary


def computeBow(image, vocabulary, feature_type):
    # extracts features from the image, and returns a BOW representation using a vocabulary

    # image is 2D array
    # vocabulary is an array of size dict_size x d
    # feature type is a string (from "sift", "surf", "orb") specifying the feature
    # used to create the vocabulary
    if feature_type == "sift":
        sift = cv2.xfeatures2d.SIFT_create(nfeatures=10)
    elif feature_type == "surf":
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=300)
    else:
        orb = cv2.ORB_create(nfeatures=10)

    # BOW is the new image representation, a normalized histogram
    if feature_type == "sift":
        kp, descriptors = sift.detectAndCompute(image, None)
    elif feature_type == "surf":
        kp, descriptors = surf.detectAndCompute(image, None)
    else:
        kp, descriptors = orb.detectAndCompute(image, None)
    
    bow_arr = [0] * len(vocabulary)
    distances = distance.cdist(descriptors, vocabulary)
    for dist in distances:
        min_ind = np.argmin(np.asarray(dist))
        bow_arr[min_ind] += 1

    Bow = preprocessing.normalize([bow_arr])
    
    return Bow[0]


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

    classResult = []

    resized_train_features = copy.deepcopy(train_features)
    resized_test_features = copy.deepcopy(test_features)

    for scale in (8, 16, 32):
        for neighbors in (1, 3, 6):
            start_time = time.time()
            for x in range(0, len(train_features)):
                resized_train_features[x] = np.ndarray.flatten(imresize(resized_train_features[x], scale))
            for x in range(0, len(test_features)):
                resized_test_features[x] = np.ndarray.flatten(imresize(resized_test_features[x], scale))
            pred = KNN_classifier(resized_train_features, train_labels, resized_test_features, neighbors)
            acc = reportAccuracy(pred, test_labels)
            end_time = time.time()
            classResult += [acc, end_time - start_time]

    return classResult
    