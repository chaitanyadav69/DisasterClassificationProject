import tkinter as tk

from tkinter import filedialog
from tkinter import messagebox

from PIL import Image

import numpy as np
import tensorflow as tf

IMAGE_SIZE = (32,32)

classes = [
    'Earthquake Damage',
    'Infrastructure Damage',
    'Urban Fire',
    'Wildfire',
    'Human Damage',
    'Drought',
    'Landslide',
    'Non Damage Human',
    'Non Damage Buildings',
    'Non Damage Forest',
    'Non Damage Sea',
    'Water Disaster'
]

model = None

root = tk.Tk()

root.title("Disaster Classification System")

root.geometry("700x600")


def load_model():

    global model

    file_path = filedialog.askopenfilename(
        filetypes=[("H5 Model", "*.h5")]
    )

    model = tf.keras.models.load_model(file_path)

    messagebox.showinfo(
        "Success",
        "Model Loaded Successfully"
    )


def predict_image():

    global model

    if model is None:

        messagebox.showwarning(
            "Warning",
            "Please load model first"
        )

        return

    file_path = filedialog.askopenfilename(
        filetypes=[("Image Files", "*.jpg *.png *.jpeg")]
    )

    image = Image.open(file_path)

    image = image.resize(IMAGE_SIZE)

    image_array = np.array(image) / 255.0

    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)

    predicted_class = classes[np.argmax(prediction)]

    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}%"
    )


load_button = tk.Button(
    root,
    text="Load Model",
    command=load_model,
    width=20,
    height=2
)

load_button.pack(pady=20)

predict_button = tk.Button(
    root,
    text="Predict Image",
    command=predict_image,
    width=20,
    height=2
)

predict_button.pack(pady=20)

result_label = tk.Label(
    root,
    text="",
    font=("Arial", 14)
)

result_label.pack(pady=30)

root.mainloop()
