import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import os

tf.random.set_seed(42)

# Configuration
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.001
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
OUTPUT_CLASSES = 2

MODEL_URL = "https://tfhub.dev/google/imagenet/mobilenet_v2_130_224/classification/4"

def build_model(model_url=MODEL_URL):
    print("Building model with:", model_url)

    # Load model
    feature_extractor_layer = hub.KerasLayer(
        model_url,
        trainable=False,
        name='feature_extractor'
    )

    inputs = tf.keras.Input(shape=INPUT_SHAPE)
    outputs = feature_extractor_layer(inputs)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()
    return model

def prepare_data_generators(train_dir, test_dir, validation_split=0.2):
    """
    Create data generators for training and testing
    Since we only have training and testing data, we'll use a validation split
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=validation_split 
    )

    test_datagen = ImageDataGenerator(rescale=1./255)
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical', 
        subset='training'
    )

    # Test generator
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, validation_generator, test_generator


def train_model(model, train_generator, validation_generator, epochs=EPOCHS):
    """
    Train the model
    """

    steps_per_epoch = train_generator.samples // train_generator.batch_size
    validation_steps = validation_generator.samples // validation_generator.batch_size

    # Training model
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps
    )

    return history

def plot_training_history(history):
    """
    Plot training and validation accuracy and loss
    """
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.show()

def evaluate_model(model, test_generator):
    """
    Evaluate the model on test data
    """
    
    class_indices = test_generator.class_indices
    class_names = {v: k for k, v in class_indices.items()}

    predictions = model.predict(test_generator, steps=np.ceil(test_generator.samples/test_generator.batch_size))
    predicted_classes = np.argmax(predictions, axis=1)
    true_classes = test_generator.classes[:len(predicted_classes)]

    from sklearn.metrics import classification_report, confusion_matrix
    print("\nClassification Report:")
    print(classification_report(true_classes, predicted_classes,
                              target_names=class_names.values()))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(true_classes, predicted_classes)
    print(cm)

    test_loss, test_acc, precision, recall = model.evaluate(test_generator)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test precision: {precision:.4f}")
    print(f"Test recall: {recall:.4f}")
    return test_loss, test_acc

def predict_image(model, image_path):
    """
    Make a prediction on a single image
    """
    from tensorflow.keras.preprocessing import image

    img = image.load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class]

    # Assuming class 0 is "with_mask" and class 1 is "without_mask"
    class_labels = {0: "With Mask", 1: "Without Mask"}
    result = class_labels[predicted_class]

    return result, confidence

def save_model(model, model_filename="mask_detection_model"):
    """
    Save the trained model
    """
    model.save(model_filename)
    print(f"Model saved as {model_filename}")



train_dir = '/content/drive/MyDrive/Mask_Detection/Train'
test_dir = '/content/drive/MyDrive/Mask_Detection/Test'

model = build_model()
print("Model built successfully")

train_generator, validation_generator, test_generator = prepare_data_generators(train_dir, test_dir)
print("Class mappings:", train_generator.class_indices)

# Train model
print("Starting model training...")
history = train_model(model, train_generator, validation_generator)
plot_training_history(history)
print("Evaluating model...")
evaluate_model(model, test_generator)

# Save model
save_model(model)
print("Training and evaluation completed!")