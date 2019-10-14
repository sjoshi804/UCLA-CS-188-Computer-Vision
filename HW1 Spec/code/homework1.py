from classifiers import *
import cv2
import os
import numpy
# interpreting your performance with 100 training examples per category:
# accuracy  =   0 ->  your code is broken (probably not the classifier's
#                     fault! a classifier would have to be amazing to
#                     perform this badly).
#  accuracy ~= .07 -> your performance is chance.
#  accuracy ~= .20 -> rough performance with tiny images and nearest
#                     neighbor classifier.
#  accuracy ~= .20 -> rough performance with tiny images and linear svm
#                     classifier. the linear classifiers will have a lot of
#                     trouble trying to separate the classes and may be
#                     unstable (e.g. everything classified to one category)
#  accuracy ~= .50 -> rough performance with bag of sift and nearest
#                     neighbor classifier.
#  accuracy ~= .60 -> you've gotten things roughly correct with bag of
#                     sift and a linear svm classifier.
#  accuracy >= .70 -> you've also tuned your parameters well. e.g. number
#                     of clusters, svm regularization, number of patches
#                     sampled when building vocabulary, size and step for
#                     dense sift features.
#  accuracy >= .80 -> you've added in spatial information somehow or you've
#                     added additional, complementary image features. this
#                     represents state of the art in lazebnik et al 2006.
#  accuracy >= .85 -> you've done extremely well. this is the state of the
#                     art in the 2010 sun database paper from fusing many 
#                     features. don't trust this number unless you actually
#                     measure many random splits.
#  accuracy >= .90 -> you get to teach the class next year.
#  accuracy >= .96 -> you can beat a human at this task. this isn't a
#                     realistic number. some accuracy calculation is broken
#                     or your classifier is cheating and seeing the test
#                     labels.
TRAINING_DATA = "/Users/sjoshi/Documents/Classes/CS 188 - Computer Vision/HW1 Spec/data/train"
TESTING_DATA = "/Users/sjoshi/Documents/Classes/CS 188 - Computer Vision/HW1 Spec/data/test"
def read_data(path):
    images = []
    labels = []
    for dir in os.listdir(path):
        if dir == ".DS_Store":
            continue
        for file_name in os.listdir(path + "/" + dir):
            images.append(cv2.imread(path + "/" + dir + "/" + file_name, cv2.IMREAD_GRAYSCALE))
            labels.append(dir)
    return images, labels

if __name__ == "__main__":
    #Read and label images
    training_images, training_labels = read_data(TRAINING_DATA)
    testing_images, testing_labels = read_data(TESTING_DATA)

    #Resize and flatten images
    for i in range(0, len(training_images)):
        training_images[i] = numpy.ndarray.flatten(imresize(training_images[i], 16))
    for j in range(0, len(testing_images)):
        testing_images[j] = numpy.ndarray.flatten(imresize(testing_images[j], 16))

    predicted_labels = KNN_classifier(training_images, training_labels, testing_images, 3)
    acc = reportAccuracy(testing_labels, predicted_labels, None)
    print(acc)
