from utils import *
from sklearn.neighbors import KNeighborsClassifier


def KNN_classifier(train_features, train_labels, test_features, num_neighbors):
    # outputs labels for all testing images
    # train_features is an N x d matrix, where d is the dimensionality of the
    # feature representation.
    # train_labels is an N x 1 array, where each entry is an integer
    # indicatings the ground truth category for each training image.
    # test_features is an M x d array, where d is the dimensionality of the
    # feature representation. You can assume M = N unless you've modified the
    # starter code.
    # predicted_categories is an M x 1 array, where each entry is an integer
    # indicating the predicted category for each test image.
    knn_classifier = KNeighborsClassifier(n_neighbors=num_neighbors)
    knn_classifier.fit(train_features, train_labels)
    predicted_categories = knn_classifier.predict(test_features)
    return predicted_categories


# def SVM_classifier(train_features, train_labels, test_features, is_linear, lambda):
#     # this function will train a linear svm for every category (i.e. one vs all)
#     # and then use the learned linear classifiers to predict the category of
#     # every test image. every test feature will be evaluated with all 15 svms
#     # and the most confident svm will "win". confidence, or distance from the
#     # margin, is w*x + b where '*' is the inner product or dot product and w and
#     # b are the learned hyperplane parameters.

#     # train_features is an n x d matrix, where d is the dimensionality of
#     # the feature representation.
#     # train_labels is an n x 1 array, where each entry is an integer 
#     # indicating the ground truth category for each training image.
#     # test_features is an m x d matrix, where d is the dimensionality of the
#     # feature representation. (you can assume m=n unless you modified the 
#     # starter code)
#     # is_linear is a boolean. If true, you will train linear SVMs. Otherwise, you 
#     # will use SVMs with a Radial Basis Function (RBF) Kernel.
#     # lambda is a scalar, the value of the regularizer for the SVMs
#     # predicted_categories is an m x 1 array, where each entry is an integer
#     # indicating the predicted category for each test image.
#     return predicted_categories