
# Load the trained model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load model
model = tf.keras.models.load_model('path/to/best_model.keras')

# Load and preprocess image
img_path = 'path/to/image.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0  # Normalize
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make prediction
prediction = model.predict(img_array)
predicted_class_idx = np.argmax(prediction[0])
confidence = prediction[0][predicted_class_idx]

# Class names
class_names = ['cocount', 'not_cocount']
predicted_class = class_names[predicted_class_idx]

print(f"Prediction: {predicted_class}")
print(f"Confidence: {confidence*100:.2f}%")
