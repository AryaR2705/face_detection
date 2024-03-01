import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
import joblib

# Function to load and preprocess color images
def load_images(directory):
    images = []
    labels = []
    for filename in os.listdir(directory):
        if filename.startswith("c") and filename.endswith(".png"):  # Adjust based on your image format
            image_path = os.path.join(directory, filename)
            label = int(filename[1])  # Assuming labels are encoded as t1, t2, ..., t9
            images.append(cv2.imread(image_path, cv2.IMREAD_COLOR))
            labels.append(label)
    return images, labels

# Specify the path to your thresholded images directory
thresholded_images_directory = "/Users/aryaramteke/Desktop/Work/webD/face/out"  # Update this path

# Load color images and labels
images, labels = load_images(thresholded_images_directory)

# Convert color images and labels to NumPy arrays
images = [cv2.resize(image, (50, 50)) for image in images]  # Resize images if needed
X = np.array([image.flatten() for image in images])
y = np.array(labels)

# Debugging: Print the shapes of the input data
print("X shape:", X.shape)
print("y shape:", y.shape)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Debugging: Print the shapes of training and testing sets
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Create a pipeline with PCA and SVM
clf = make_pipeline(PCA(n_components=3, whiten=True, random_state=42),
                    SVC(C=1.0, kernel='rbf', gamma=0.01))

# Train the model
clf.fit(X_train, y_train)

# Debugging: Print some predicted labels
y_pred = clf.predict(X_test)
print("Some predicted labels:", y_pred[:10])

# Calculate the accuracy score
accuracy = accuracy_score(y_test, y_pred)

# Display the accuracy score
print(f"Accuracy Score: {accuracy}")

# Save the trained model to a file (e.g., model.pkl)
joblib.dump(clf, 'facemodel.pkl')
