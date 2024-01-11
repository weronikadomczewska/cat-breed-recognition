import tensorflow as tf
import os
import numpy as np

train_dataset_path = 'train_data'
validation_dataset_path = 'validation_data'
class_names = os.listdir(train_dataset_path)
num_classes = len(class_names)
checkpoint_path = "cat_breed_classification_model/cp.ckpt"
size = 256

model = tf.keras.models.Sequential([
    tf.keras.layers.Dropout(.2),
    tf.keras.layers.Rescaling(1. / 255, input_shape=(size, size, 3)),
    tf.keras.layers.Conv2D(32, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(128, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(256, 3, padding='same', activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.load_weights(checkpoint_path).expect_partial()

validation_folder = os.listdir(validation_dataset_path)
for breed in validation_folder:
    for i in os.listdir(f'{validation_dataset_path}/{breed}'):
        img = tf.keras.utils.load_img(f'{validation_dataset_path}/{breed}/{i}', target_size=(size, size))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(breed, i,
              "This image most likely belongs to {} with a {:.2f} percent confidence."
              .format(class_names[np.argmax(score)], 100 * np.max(score)))
