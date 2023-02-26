import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import pathlib
import matplotlib.pyplot as plt
from  AIs.TensorFlow.TFCNN import SupervisedNeuralNetwork as CNN

# grab local data set - will switch to sql database
myFile = 'C:/Users/robot/Documents/GitHub/AI-Projects/TrainingDataSets/Flowers'
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
normalization_layer = tf.keras.layers.Rescaling(1./255)
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

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    ax = plt.subplot(3, 3, 1)
    plt.imshow(images[0].numpy().astype("uint8"))
    plt.title(class_names[labels[0]])
    plt.axis("off")

plt.show()
