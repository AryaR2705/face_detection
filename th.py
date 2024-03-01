import os
try:
    import cv2
except ImportError:
    # Install OpenCV if not already installed
    os.system('pip install opencv-python')
    import cv2

# Function to threshold and save images
def threshold_and_save(image_path, output_path, threshold_value=128):
    # Check if the image file exists
    if not os.path.isfile(image_path):
        print(f"Error: Image file not found - {image_path}")
        return

    # Read the image
    image = cv2.imread(image_path)

    # Check if the image is successfully loaded
    if image is None:
        print(f"Error: Unable to read image - {image_path}")
        return

    # Convert the image to grayscale (assuming color images)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply binary thresholding
    _, thresholded_image = cv2.threshold(gray_image, threshold_value, 255, cv2.THRESH_BINARY)

    # Save the thresholded image
    filename = os.path.basename(image_path)
    output_filepath = os.path.join(output_path, f"t{filename}")
    cv2.imwrite(output_filepath, thresholded_image)
    print(f"Thresholded image saved: {output_filepath}")

# Specify the path to your color images directory
input_images_directory = "/Users/aryaramteke/Desktop/Work/webD/face/img"  # Update this path

# Specify the output directory for the thresholded images
output_images_directory = "/Users/aryaramteke/Desktop/Work/webD/face/out"  # Update this path

# Set the threshold value (adjust as needed)
threshold_value = 80

# Iterate through each image in the input directory
for filename in os.listdir(input_images_directory):
    if filename.endswith(".png"):  # Adjust based on your image format
        image_path = os.path.join(input_images_directory, filename)
        threshold_and_save(image_path, output_images_directory, threshold_value)
