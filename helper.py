import numpy as np
import os
import tensorflow as tf

train_dataset_path = 'train_data'
test_dataset_path = 'test_data'
validation_dataset_path = 'validation_data'
size = 256
epoch = 3
class_names = os.listdir(validation_dataset_path)
num_classes = len(class_names)

train_dataset = tf.keras.utils.image_dataset_from_directory(train_dataset_path, labels='inferred',
                                                            label_mode='int',
                                                            color_mode='rgb',
                                                            image_size=(size, size),
                                                            shuffle=True,
                                                            seed=None,
                                                            validation_split=None,
                                                            subset=None,
                                                            interpolation='bilinear',
                                                            follow_links=False,
                                                            crop_to_aspect_ratio=False, )

test_dataset = tf.keras.utils.image_dataset_from_directory(test_dataset_path, labels='inferred',
                                                           label_mode='int',
                                                           color_mode='rgb',
                                                           image_size=(size, size),
                                                           shuffle=True,
                                                           seed=None,
                                                           validation_split=None,
                                                           subset=None,
                                                           interpolation='bilinear',
                                                           follow_links=False,
                                                           crop_to_aspect_ratio=False, )

AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = tf.keras.layers.Rescaling(1. / size)
normalized_ds = train_dataset.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]

checkpoint_path = "cat_breed_classification_model/cp.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

base_model = tf.keras.applications.ResNet152(weights='imagenet', include_top=False, input_shape=(size, size, 3))
for layer in base_model.layers:
  layer.trainable = False

x = tf.keras.layers.Flatten()(base_model.output)
x = tf.keras.layers.Dense(512, activation='relu')(x)
predictions = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

head_model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
head_model.compile(optimizer='adam', loss=tf.keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'])

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


history = head_model.fit(train_dataset, validation_data=test_dataset, batch_size=64, epochs=epoch, callbacks=[cp_callback])


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

os.listdir(checkpoint_dir)

epochs_range = range(epoch)

validation_folder = os.listdir(validation_dataset_path)
for breed in validation_folder:
    for i in os.listdir(f'{validation_dataset_path}/{breed}'):
        img = tf.keras.utils.load_img(f'{validation_dataset_path}/{breed}/{i}', target_size=(size, size))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        predictions = head_model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print(breed, i,
              "This image most likely belongs to {} with a {:.2f} percent confidence."
              .format(class_names[np.argmax(score)], 100 * np.max(score)))

