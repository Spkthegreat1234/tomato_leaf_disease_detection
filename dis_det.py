import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn import preprocessing
import os

def extract_features(image):
    # Use simple color histogram features as an example
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def load_images(folder, label):
    images = []
    labels = []
    for filename in os.listdir(folder):
        path = os.path.join(folder, filename)
        image = cv2.imread(path)
        features = extract_features(image)
        images.append(features)
        labels.append(label)
    return images, labels

# Example: Assume you have two folders "healthy" and "infected" with plant images
healthy_folder = "images/healthy_images_tomato"
infected_folder = "images/unhealthy_images_tomato"
# Load images and labels for training
healthy_images, healthy_labels = load_images(healthy_folder, 0)
infected_images, infected_labels = load_images(infected_folder, 1)
# Combine healthy and infected data
all_images = healthy_images + infected_images
all_labels = healthy_labels + infected_labels
# Convert lists to numpy arrays
X_train = np.array(all_images)
y_train = np.array(all_labels)
# Normalize features
scaler = preprocessing.StandardScaler().fit(X_train)
X_train = scaler.transform(X_train)
# Train a Support Vector Machine (SVM) classifier
model = SVC(kernel='linear')
model.fit(X_train, y_train)
# Now, let's use the trained model to predict on new input images
# Replace 'path/to/your/input/image.jpg' with the path to your input image
input_image_path = 'images/unhealthy_images_tomato/3cb6ee9b-8550-4216-bde5-814807add4e5___PSU_CG 2395.JPG'
# Load the input image
input_image = cv2.imread(input_image_path)
# Extract features from the input image
input_features = extract_features(input_image)
# Normalize the features
input_features = scaler.transform([input_features])
# Make a prediction
prediction = model.predict(input_features)
# Display the prediction
if prediction[0] == 0:
    print("The plant leaf is healthy.")
else:
    print("The plant leaf is infected.")