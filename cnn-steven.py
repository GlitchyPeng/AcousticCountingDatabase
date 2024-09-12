import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os
from PIL import Image
import numpy as np

# Define the path to your images folder
image_folder = 'spectrograms'  # Update with the correct path
image_size = (128, 128)  # Resize images to 128*128

# Create lists to hold images and labels
images = []
labels = []

# Load images and labels based on filename pattern
for filename in os.listdir(image_folder):
    if filename.endswith('.png'):
        # Extract label from filename (first number before the dash)
        label = int(filename.split('-')[0])
        labels.append(label)
        
        # Load image and convert to RGB
        img_path = os.path.join(image_folder, filename)
        img = Image.open(img_path).resize(image_size).convert('RGB')  # Convert to RGB
        img_array = np.array(img) / 255.0  # Normalize pixel values
        images.append(img_array)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model with correct input shape
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),  # Corrected input shape
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(set(labels)))  # Adjust output layer to match the number of classes
])

# Compile the model with the correct optimizer
model.compile(optimizer=tf.keras.optimizers.Adam(amsgrad=True),  # Corrected AMSGrad usage
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# Train the model
history = model.fit(train_images, train_labels, epochs=300,
                    validation_data=(test_images, test_labels))

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Test accuracy: {test_acc}")
