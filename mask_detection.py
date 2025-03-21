import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

model_path = '/content/drive/MyDrive/Mask_Detection/model/mask_detection_model.h5'
IMG_SIZE = 224

model = tf.keras.models.load_model(model_path)
print("Model loaded successfully!")

# Class labels
class_labels = {0: "With Mask", 1: "Without Mask"}

def predict_image(img_path):
    """
    Loads an image, preprocesses it, and makes a prediction.
    """
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Expand dimensions for batch input
    img_array = img_array / 255.0  # Normalize pixel values

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    result = class_labels[predicted_class]
    print(f"Prediction: {result} (Confidence: {confidence:.2f})")
    return result, confidence


for file in os.listdir("/Mask_Detection/Datasets/Test"):
  result, confidence = predict_image(file)
  plt.figure(figsize=(5, 5))
  plt.imshow(file)
  plt.title(f"Prediction: {result} (Confidence: {confidence:.2f})%")
  print('==================================================')