import tensorflow as tf
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
from PIL import Image, ImageTk

# Load the trained model
model_path = "my_model.keras"  # Update with your actual model path
model = tf.keras.models.load_model(model_path)

# Define class labels
class_labels = [
    "Normal (N)", "Diabetes (D)", "Glaucoma (G)", "Cataract (C)", 
    "Age-related Macular Degeneration (A)", "Hypertension (H)", 
    "Pathological Myopia (M)", "Other diseases/abnormalities (O)"
]

# Function to preprocess the image
def preprocess_image(image_path, img_size=(224, 224)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
    image = cv2.resize(image, img_size)  # Resize to match model input
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Function to make a prediction
def predict_image():
    global image_path
    if not image_path:
        result_label.config(text="Please select an image first!", fg="red")
        return

    result_label.config(text="Processing...", fg="blue")
    root.update_idletasks()

    image = preprocess_image(image_path)
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = np.max(prediction)

    result_label.config(
        text=f"Prediction: {class_labels[predicted_class]}\nConfidence: {confidence:.2f}",
        fg="green"
    )

# Function to open file dialog and display the image
def open_image():
    global image_path
    image_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.jpeg;*.png")])
    
    if image_path:
        img = Image.open(image_path)
        img = img.resize((300, 300))  # Resize for display
        img = ImageTk.PhotoImage(img)
        image_label.config(image=img)
        image_label.image = img
        result_label.config(text="")  # Clear previous result

# Create the Tkinter window
root = tk.Tk()
root.title("Eye Disease Classifier")
root.geometry("450x600")
root.configure(bg="#f4f4f4")

# Frame for Image Display
frame = Frame(root, bg="#f4f4f4")
frame.pack(pady=20)

image_label = Label(frame)
image_label.pack()

# Buttons
btn_frame = Frame(root, bg="#f4f4f4")
btn_frame.pack(pady=10)

open_button = Button(btn_frame, text="Open Image", command=open_image, font=("Arial", 12), bg="#3498db", fg="white", padx=10, pady=5)
open_button.grid(row=0, column=0, padx=10)

predict_button = Button(btn_frame, text="Predict", command=predict_image, font=("Arial", 12), bg="#2ecc71", fg="white", padx=10, pady=5)
predict_button.grid(row=0, column=1, padx=10)

# Result Label
result_label = Label(root, text="", font=("Arial", 14, "bold"), bg="#f4f4f4")
result_label.pack(pady=20)

# Run the Tkinter app
root.mainloop()
