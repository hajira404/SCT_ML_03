import numpy as np
import matplotlib.pyplot as plt
import joblib  # Use joblib to load the scikit-learn pipeline
from PIL import Image  # Import PIL for image handling

# Load your trained model
model = joblib.load('cat_dog_classifier_pipeline.pkl')  # Adjust the filename as needed

# Function to load and preprocess the image
def load_and_preprocess_image(img_path, target_size):
    img = Image.open(img_path)  # Load the image using PIL
    img_resized = img.resize(target_size)  # Resize the image for model input
    img_array = np.array(img_resized)  # Convert resized image to numpy array
    img_array = img_array / 255.0  # Normalize if needed
    return img_array, img  # Return both processed array and original image

# Path to your new image
img_path = '2.png'  # Use the uploaded image path
image_height, image_width = 64, 64  # Update with the actual dimensions used during training
img_array, original_img = load_and_preprocess_image(img_path, (image_width, image_height))

# Flatten the image array for the model input
img_array = img_array.flatten().reshape(1, -1)  # Reshape to (1, n_features)

# Make a prediction
predictions = model.predict(img_array)

# For binary classification, threshold the predictions to get class indices
predicted_class = int(predictions[0] > 0.5)  # Convert probabilities to class label

# Map predicted class to label
labels = ['cat', 'dog']  # Adjust this based on your classes
predicted_label = labels[predicted_class]  # Get the predicted class

# Display the original image and prediction
plt.figure(figsize=(8, 8))  # Set a figure size for better display
plt.imshow(original_img)  # Display the original image
plt.title(f'Predicted: {predicted_label}', fontsize=20)  # Make title larger for visibility
plt.axis('off')  # Hide axes
plt.show()