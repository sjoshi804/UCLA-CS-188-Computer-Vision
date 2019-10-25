from utils import *
import argparse

parser = argparse.ArgumentParser(description='CS188.2 - Fall 19 - Homework 1')
parser.add_argument("--tiny", "-t", type=bool, default=True, help='run Tiny Images')
parser.add_argument("--create-path", "-cp", type=bool, default=True, help='create the Results directory')
args = parser.parse_args()

# The argument is included as an idea for debugging, with a few examples in the main. Feel free to modify it or add arguments.
# You are also welcome to disregard this entirely
#DHRUV
#############################################################################################################################
# This file contains the main. All the functions you need to write are included in utils. You also need to edit the main.
# The main just gets you started with the data and highlights the high level structure of the project.
# You are free to modify it as you wish - the modifications you are required to make have been marked but you are free to make
# others.
# What you cannot modify is the number of files you have to save or their names. All the save calls are done for you, you
# just need to specify the right data.
#############################################################################################################################


if __name__ == "__main__":
    
    if args.create_path:
        # To save accuracies, runtimes, voabularies, ...
        if not os.path.exists('Results'):
            os.mkdir('Results') 
        SAVEPATH = 'Results/'

    print("Loading data...")
    # Load data, the function is written for you in utils
    train_images, test_images, train_labels, test_labels = load_data()
    print("Loaded data.")

    print("Running tiny images")
    if args.tiny:
        # You have to write the tinyImages function
        tinyRes = tinyImages(train_images, test_images, train_labels, test_labels)
    
        # Split accuracies and runtimes for saving  
        for element in tinyRes[::2]:
            # Check that every second element is an accuracy in reasonable bounds
            assert (7 < element and element < 21)
        acc = np.asarray(tinyRes[::2])
        runtime = np.asarray(tinyRes[1::2])
    
        # Save results
        np.save(SAVEPATH + 'tiny_acc.npy', acc)
        np.save(SAVEPATH + 'tiny_time.npy', runtime)
    print("Ran tiny images, accuracies are: ")
    print(acc)
    # Create vocabularies, and save them in the result directory
    # You need to write the buildDict function
    vocabularies = []
    vocab_idx = [] # If you have doubts on which index is mapped to which vocabulary, this is referenced here
    # e.g vocab_idx[i] will tell you which algorithms/neighbors were used to compute vocabulary i
    # This isn't used in the rest of the code so you can feel free to ignore it
    print("Constructing vocabularies")
    for feature in ['sift', 'surf', 'orb']:
        for algo in ['kmeans', 'hierarchical']:
            for dict_size in [20, 50]:
                vocabulary = buildDict(train_images, dict_size, feature, algo)
                filename = 'voc_' + feature + '_' + algo + '_' + str(dict_size) + '.npy'
                np.save(SAVEPATH + filename, np.asarray(vocabulary))
                vocabularies.append(vocabulary) # A list of vocabularies (which are 2D arrays)
                vocab_idx.append(filename.split('.')[0]) # Save the map from index to vocabulary
                print(feature + " " + algo + " " + str(dict_size) + " done")
    print("Constructed Vocabularies")            
    # Compute the Bow representation for the training and testing sets
    test_rep = [] # To store a set of BOW representations for the test images (given a vocabulary)
    train_rep = [] # To store a set of BOW representations for the train images (given a vocabulary)
    features = ['sift'] * 4 + ['surf'] * 4 + ['orb'] * 4 # Order in which features were used 
    
    # for vocabulary generation
    vocabularies = []
    for file_name in vocab_idx:
        vocabularies.append(np.load(SAVEPATH + file_name + '.npy'))
    print("Computing BOW")
    # You need to write ComputeBow()
    for i, vocab in enumerate(vocabularies):
        for image in train_images: # Compute the BOW representation of the training set
            rep = computeBow(image, vocab, features[i]) # Rep is a list of descriptors for a given image
            train_rep.append(rep)
        np.save(SAVEPATH + 'bow_train_' + str(i) + '.npy', np.asarray(train_rep)) # Save the representations for vocabulary i
        train_rep = [] # reset the list to save the following vocabulary
        for image in test_images: # Compute the BOW representation of the testing set
            rep = computeBow(image, vocab, features[i])
            test_rep.append(rep)
        np.save(SAVEPATH + 'bow_test_' + str(i) + '.npy', np.asarray(test_rep)) # Save the representations for vocabulary i
        print(feature + " " + algo + " " + str(dict_size) + " done")
        test_rep = [] # reset the list to save the following vocabulary
    print("Computed BOW.")
    
    # Use BOW features to classify the images with a KNN classifier
    # A list to store the accuracies and one for runtimes
    knn_accuracies = []
    knn_runtimes = []
    print("Classifying BOW using KNN")
    train_representation = []
    test_representation = []
    # Your code below, eg:
    for i, vocab in enumerate(vocabularies):
        #Load BOWs generated earlier
        train_representation.append(np.load(SAVEPATH + 'bow_train_' + str(i) + '.npy'))
        test_representation.append(np.load(SAVEPATH + 'bow_test_' + str(i) + '.npy'))
        for k in [1, 3, 6]:
            start_time = time.time()
            predicted_labels = KNN_classifier(train_representation[i], train_labels, test_representation[i], k)
            end_time = time.time()
            knn_accuracies.append(reportAccuracy(test_labels, predicted_labels))
            knn_runtimes.append(end_time - start_time)
  
    np.save(SAVEPATH+'knn_accuracies.npy', np.asarray(knn_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'knn_runtimes.npy', np.asarray(knn_runtimes)) # Save the runtimes in the Results/ directory
    print("Classified BOW using KNN")
    # Use BOW features to classify the images with 15 Linear SVM classifiers
    lin_accuracies = []
    lin_runtimes = []
    print("Linear SVM Running")
    # Your code below
    for cost in [1, 0.1, 10]:
        for i in range(0, len(train_representation)):
            start_time = time.time()
            predicted_labels = SVM_classifier(train_representation[i], train_labels, test_representation[i], True, cost)
            end_time = time.time()
            lin_accuracies.append(reportAccuracy(test_labels, predicted_labels))
            lin_runtimes.append(end_time - start_time)

    np.save(SAVEPATH+'lin_accuracies.npy', np.asarray(lin_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH+'lin_runtimes.npy', np.asarray(lin_runtimes)) # Save the runtimes in the Results/ directory
    print("Linear SVM Done.")
    # Use BOW features to classify the images with 15 Kernel SVM classifiers
    rbf_accuracies = []
    rbf_runtimes = []
    print("RBF SVM Starting")
    # Your code below
    for cost in [1, 0.1, 10]:
        for i in range(0, len(train_representation)):
            start_time = time.time()
            predicted_labels = SVM_classifier(train_representation[i], train_labels, test_representation[i], False, cost)
            end_time = time.time()
            rbf_accuracies.append(reportAccuracy(test_labels, predicted_labels))
            rbf_runtimes.append(end_time - start_time)
    
    np.save(SAVEPATH +'rbf_accuracies.npy', np.asarray(rbf_accuracies)) # Save the accuracies in the Results/ directory
    np.save(SAVEPATH +'rbf_runtimes.npy', np.asarray(rbf_runtimes)) # Save the runtimes in the Results/ directory
    print("RBF SVM Done")
    print("All tests successfully passed.")

    
