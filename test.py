import cv2
import numpy as np
import joblib

# Load the trained model
clf_colab = joblib.load('facemodel.pkl')

# Assume the label associated with the person during training is 1
trained_person_label = 1

# Function to preprocess a single image
def preprocess_image(image):
    resized_image = cv2.resize(image, (50, 50)).flatten()
    return resized_image

# Function to check if a person is correct or incorrect
def check_person(frame):
    # Convert the frame to grayscale if your model was trained with grayscale images
    # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
        frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Iterate through detected faces
    for (x, y, w, h) in faces:
        # Extract the face region
        face_roi = frame[y:y + h, x:x + w]

        # Preprocess the face image
        processed_face = preprocess_image(face_roi)

        # Ensure the number of features matches the model's expectations
        # Update the value (e.g., 7500) to match the number of features expected by PCA
        processed_face = np.array(processed_face).reshape(1, -1)[:, :7500]
        # Update with your feature count

        # Make a prediction using the trained model
        prediction = clf_colab.predict(processed_face)

        # Check if the predicted label matches the person it was trained on
        # Adjust the condition based on your needs
        if prediction[0] == trained_person_label:
            label_text = "Correct Person"
        else:
            label_text = "Incorrect Person"

        # Draw a rectangle around the face and display the label
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame along with label information
    cv2.imshow('Face Recognition', frame)
    cv2.waitKey(1)

# Open a video capture object (you may need to adjust the index based on your camera setup)
cap = cv2.VideoCapture(0)

while True:
    # Capture a frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Check if the person is correct or incorrect
    check_person(frame)

# Release the capture object
cap.release()
cv2.destroyAllWindows()
