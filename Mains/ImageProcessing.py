import numpy as np
import os
import PIL
import PIL.Image
import random
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from AIs.TensorFlow.TFCNN import SupervisedNeuralNetwork as CNN
import shutil
import time

# grab local data set - will switch to sql database
myFile = 'C:/Users/angel/OneDrive/Documents/GitHub/Q-Learning-Practice/TrainingDataSets/Robots'
dataset_url = os.path.abspath("./" + myFile)
archive = tf.keras.utils.get_file(myFile, 'file://' + dataset_url)
data_dir = pathlib.Path(archive).with_suffix('')

# print # of images
image_count = len(list(data_dir.glob('*/*.jpg')))
print('# of images: ' + str(image_count))

batch_size = 32
img_height = 180
img_width = 180

tf.device('/cpu:0')

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
)

class_names = train_ds.class_names
print('class names: ' + str(class_names))

for image_batch, labels_batch in train_ds:
    print(image_batch.shape)
    print(labels_batch.shape)
    break

# standardize data
normalization_layer = tf.keras.layers.Rescaling(1. / 255)
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

nn = CNN(train_ds, train_ds, val_ds)
nn.compile_model()
nn.train_model()
nn.test_model()

'''plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    ax = plt.subplot(3, 3, 1)
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.title(class_names[labels[0]])
    plt.axis("off")

plt.show()'''


def test_nn():
    test_team = input('Pass team dataset to test: ')
    origin = myFile + '/' + test_team
    target = 'C:/Users/angel/OneDrive/Documents/GitHub/Q-Learning-Practice/NotRAM/' + test_team
    shutil.copytree(origin, target)

    time.sleep(2)

    # testing img intialization - should change later cuz this is def not right
    test_myFile = 'C:/Users/angel/OneDrive/Documents/GitHub/Q-Learning-Practice/NotRAM'
    test_dataset_url = os.path.abspath("./" + test_myFile)
    test_archive = tf.keras.utils.get_file(test_myFile, 'file://' + test_dataset_url)
    test_data_dir = pathlib.Path(test_archive).with_suffix('')

    # again test stuff
    test_batch_size = 32
    test_img_height = 180
    test_img_width = 180

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_data_dir,
        validation_split=0.9,
        subset="validation",
        seed=123,
        image_size=(test_img_height, test_img_width),
        batch_size=test_batch_size
    )

    test_class_names = test_ds.class_names

    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

    nn.single_test_model(test_ds)

    plt.figure(figsize=(10, 10))

    for images, labels in test_ds.take(9):
        rnd = random.randint(0, len(images) - 1)
        ax = plt.subplot(1, 1, 1)
        plt.imshow(images[rnd].numpy().astype("uint8"))
        plt.title(test_class_names[labels[rnd]])
        plt.axis("off")

    plt.show()

    shutil.rmtree(target, ignore_errors=True)

    test_nn()

test_nn()
