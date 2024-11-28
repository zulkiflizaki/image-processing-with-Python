"""Advanced Artificial Intelligence - Use an object recognition technique to analyse images and deduce the
object within"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix

print("This is a testing program using OpenCV library")
print(cv2.__version__)

# Model Building
# 1. Dataset
# Start by using the supplied dataset, it is then advisable to add representative images to your own
# importing flower dataset
image = cv2.imread("shiba-imu.jpg")
if image is None:
    print("Image is not loaded correctly")
else:
    print("OpenCV is installed and working correctly!")


# 2. Data Preprocessing
# Perform data preprocessing. Methods such as grey scaling and histogram equalization
# should be considered.
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
resized_image = cv2.resize(gray_image, (28, 28))
normalized_image = resized_image / 255.0


# 3. Building and testing the object recognition systems
# Build a tool that parses an image and attempts to identify which objects are present.
# Split up our data into training and test sets and train the object recognizer/recognition
# tool/system on the training set. Evaluate its performance on the test set.

# feature extraction
# Edge Detection: Detect edges using Canny edge detection.
edges = cv2.Canny(image, 100, 200)

# Contours: Find contours in the image.
contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

# Building the Recognition Model
# Haar Cascades: Use pre-trained Haar cascades for face detection.
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(gray_image, 1.1, 4)

# Drawing Bounding Boxes: Draw boxes around detected features.
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

# Model training
clf = svm.SVC(gamma='scale')
clf.fit(train_data, train_labels)



# 4. Prediction of Unseen Images
# Evaluate the performance of your model on a reserved validation dataset of images.
predicted_labels = clf.predict(test_data)


accuracy = accuracy_score(test_labels, predicted_labels)
matrix = confusion_matrix(test_labels, predicted_labels)


# plt.hist(img.ravel(), bins=256, range=(0.0, 1.0), fc='k', ec='k')
# histr = cv2.calcHist([image],[0],None,[256],[0,256]) 

#RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#GRAYSCALE_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Edge detection
#edges = cv2.Canny(image = RGB_img, threshold1 = 100, threshold2 = 100)

# Create subplots
#fig, axs = plt.subplots(1, 2, figsize = (7, 4))

# Plot the original image
#axs[0].imshow(RGB_img)
#axs[0].set_title('Original Image')

# Plot the blurred image
#axs[1].imshow(edges)
#axs[1].set_title('Image edges')

# Remove ticks from the subplots
#for ax in axs:
#    ax.set_xticks([])
#    ax.set_yticks([])



# Display the subplots
#plt.tight_layout()
plt.imshow(image)
plt.show()
