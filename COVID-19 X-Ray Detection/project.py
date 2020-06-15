#Libraries
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix


import matplotlib.pyplot as plt
from sklearn.metrics import classification_report


# For preprocessing
from imutils import paths
import pandas as pd
import numpy as np
import random
import shutil
import os
import cv2

from google.colab import drive
drive.mount('/content/drive')

locations = list(paths.list_images("/content/drive/My Drive/UCLA/Spring2020/CS 168/Final Project/dataset"))

#Model
#Initialization
X=[] #X_Ray
Y=[] #Cateogry


# X = images
# Y = categories

label = LabelEncoder()
batch_size=10
epochs=50
learning_rate=0.001
decay=learning_rate/epochs
aug = ImageDataGenerator()

i=0
while i < len(locations):
    X.append(cv2.resize(cv2.imread(locations[i]), (224,224)))
    Y.append(locations[i].split(os.path.sep)[-2])
    i += 1

#Normalize intensity & Train:Test ==> 75:25
X_train, X_test, Y_train, Y_test = train_test_split(np.array(X) / 255.0, np_utils.to_categorical(label.fit_transform(np.array(Y))),
test_size=0.25, stratify=np_utils.to_categorical(label.fit_transform(np.array(Y))), random_state=42)
train_steps = len(X_train)//batch_size
test_steps = len(X_test)//batch_size
validation =(X_test,Y_test)

VGGNet = VGG16(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))
CNN = Model(inputs=VGGNet.input, outputs=Dense(2, activation="softmax")(Dropout(0.5)(Dense(64, activation="relu")(Flatten(name="flatten")(AveragePooling2D(pool_size=(4, 4))(VGGNet.output))))))

i=0
while i<len(VGGNet.layers):
    VGGNet.layers[i].trainable = False
    i += 1

optimizer = Adam(lr=learning_rate, decay=decay)
CNN.compile(loss="binary_crossentropy", optimizer=optimizer,metrics=["accuracy"])

train_steps = len(X_train)//batch_size
test_steps = len(X_test)//batch_size
validation =(X_test,Y_test)
top = CNN.fit_generator(aug.flow(X_train, Y_train, batch_size=batch_size),steps_per_epoch=train_steps,validation_data=validation,validation_steps=test_steps,epochs=epochs)

# Plot accuracy
plt.figure(figsize=(14, 10), dpi= 80, facecolor='w', edgecolor='k')
plt.plot(np.arange(0, epochs), top.history["accuracy"], label="Training Accuracy")
plt.plot(np.arange(0, epochs), top.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Accuracy", fontsize=16)
plt.legend(loc="lower left", prop={'size': 16})
plt.title("Model Accuracy", fontsize=16)
plt.show()

# Plot loss
plt.figure(figsize=(14, 10), dpi= 80, facecolor='w')
plt.plot(np.arange(0, epochs), top.history["loss"], label="Training Loss")
plt.plot(np.arange(0, epochs), top.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(loc="lower left", prop={'size': 16})
plt.title("Model Loss", fontsize=16)
plt.show()

# Check testing set on the mode
predictions = CNN.predict(X_test, batch_size=batch_size)
predicted_classes = np.argmax(predictions, axis=1)
print(classification_report(Y_test.argmax(axis=1), predicted_classes,
    target_names=label.classes_))
    
# save the model
CNN.save("drive/My Drive/model.h5")
!ls "drive/My Drive"

# load model
model = load_model("drive/My Drive/model.h5")
model.summary()


# cm = confusion_matrix(testY.argmax(axis=1), predIdxs)
# total = sum(sum(cm))
# acc = (cm[0, 0] + cm[1, 1]) / total
# sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
# specificity = cm[1, 1] / (cm[1, 0] + cm[1, 1])
# # show the confusion matrix, accuracy, sensitivity, and specificity
# print(cm)
# print("acc: {:.4f}".format(acc))
# print("sensitivity: {:.4f}".format(sensitivity))
# print("specificity: {:.4f}".format(specificity))

# calculate specificity and sensitivity
confusion = confusion_matrix(Y_test.argmax(axis=1), predicted_classes)
tn = float(confusion[0,0])
fp = float(confusion[0,1])
fn = float(confusion[1,0])
tp = float(confusion[1,1])

sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
accuracy = (tp + tn) / (tn + fp + fn + tp)
print("Sensitivity: {:.3f}".format(sensitivity))
print("Specificity: {:.3f}".format(specificity))
print("Accuracy: {:.3f}".format(accuracy))

precision = tp / (tp + fp)
print("Precision: {:.3f}".format(accuracy))
print(confusion)

print(tn, fp, fn, tp)
tn, fp, fn, tp = confusion_matrix(Y_test.argmax(axis=1), predicted_classes).ravel()
print(tn, fp, fn, tp)
