import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers ,models


data_train_path = "data/Bone_Fracture_Binary_Classification/train"
data_test_path = "data/Bone_Fracture_Binary_Classification/test"
data_val_path = "data/Bone_Fracture_Binary_Classification/val"

img_width = 256
img_height = 256

# Function to load dataset with error handling
def load_dataset(path):
    dataset = tf.keras.utils.image_dataset_from_directory(
        path,
        shuffle=True,
        image_size=(img_width, img_height),
        batch_size=32,
        validation_split=False
    )
    return dataset.apply(tf.data.experimental.ignore_errors())



# Load datasets with error handling
data_train = load_dataset(data_train_path)
data_val = load_dataset(data_val_path)
data_test = load_dataset(data_test_path)


# Build CNN model 
model = models.Sequential([

    layers.Conv2D(32, kernel_size=(3,3), padding='valid',activation='relu',input_shape=(256,256,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'),

    layers.Conv2D(32, kernel_size=(3,3), padding='valid',activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'),

    layers.Conv2D(32, kernel_size=(3,3), padding='valid',activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2,2),strides=2,padding='valid'),

    layers.Flatten(),

    layers.Dense(128,activation='relu'),
    layers.Dropout(0.1),

    layers.Dense(64,activation='relu'),
    layers.Dropout(0.1),

    layers.Dense(2, activation='sigmoid')
])


# Compile model
model.compile(optimizer='adam', 
              loss= 'sparse_categorical_crossentropy', 
              metrics=['accuracy'])


# Train model
history = model.fit(data_train,validation_data=data_val, batch_size=32, epochs=10)


# Plot training & validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()


# Evaluate model on test data
test_loss, test_acc = model.evaluate(data_test, verbose=2)
print(f"Test accuracy: {test_acc* 100:.2f}%")


# save the model
model.save("model/recognition_model.keras")




