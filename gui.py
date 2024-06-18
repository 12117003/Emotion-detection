import tkinter as tk
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import img_to_array

# Load the model architecture from JSON file
model_json_path = 'C:/Users/AISU/Downloads/model_a.json'
model_weights_path = 'C:/Users/AISU/Downloads/model_weights.weights.h5'

# Load model
def load_model(model_json_path, model_weights_path):
    if not tf.io.gfile.exists(model_json_path):
        print(f"Model JSON file not found: {model_json_path}")
        exit(1)
        
    if not tf.io.gfile.exists(model_weights_path):
        print(f"Model weights file not found: {model_weights_path}")
        exit(1)
    
    with open(model_json_path, 'r') as json_file:
        loaded_model_json = json_file.read()

    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)

    return loaded_model

# Load the emotion detection model
model = load_model(model_json_path, model_weights_path)

# Define emotions for mapping
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Create Tkinter GUI
root = tk.Tk()
root.title("Emotion Detector")

# Function to upload image
# Function to upload image
# Function to upload image
def upload_image():
    global selected_image_path
    selected_image_path = filedialog.askopenfilename()
    if selected_image_path:
        try:
            # Load image and resize for display
            image = Image.open(selected_image_path)
            image = image.resize((250, 250), Image.LANCZOS)  # Use Image.LANCZOS for resizing
            photo = ImageTk.PhotoImage(image)
            image_label.config(image=photo)
            image_label.image = photo
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")



# Function to predict emotion
# Function to predict emotion
def predict_emotion():
    global selected_image_path
    if not selected_image_path:
        messagebox.showerror("Error", "Please upload an image first.")
        return
    
    try:
        # Load and preprocess the image
        image = Image.open(selected_image_path).convert('L')  # Convert to grayscale
        image = image.resize((48, 48), Image.LANCZOS)  # Resize image
        image_array = img_to_array(image)
        image_array = np.expand_dims(image_array, axis=0)
        image_array = image_array.astype("float32") / 255.0
        
        # Predict emotion
        predictions = model.predict(image_array)
        emotion_index = np.argmax(predictions[0])
        emotion_label = emotion_labels[emotion_index]
        
        # Display predicted emotion
        messagebox.showinfo("Prediction", f"The detected emotion is: {emotion_label}")
    except Exception as e:
        messagebox.showerror("Error", f"Failed to predict emotion: {e}")


# Create widgets
upload_button = tk.Button(root, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

predict_button = tk.Button(root, text="Predict Emotion", command=predict_emotion)
predict_button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=20)

# Run the Tkinter main loop
root.mainloop()






