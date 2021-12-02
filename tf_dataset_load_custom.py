### MODIFIED FROM https://stackoverflow.com/questions/62166588/how-to-obtain-filenames-during-prediction-while-using-tf-keras-preprocessing-ima ###

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.keras.layers.preprocessing import image_preprocessing
from tensorflow.python.keras.preprocessing import dataset_utils
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
from tensorflow.python.util.tf_export import keras_export

def path_to_image(path, image_size, interpolation):
    img = io_ops.read_file(path)
    img = image_ops.decode_image(
        img, channels=3, expand_animations=False)
    img = image_ops.resize_images_v2(img, image_size, method=interpolation)
    img.set_shape((image_size[0], image_size[1], 3))
    return img

def path_to_bbox(path, image_size):

    as_string = tf.io.read_file(path)
    vals = tf.gather(tf.strings.split(as_string, '\n'), [7, 8, 18, 19, 20, 21])
    vals = tf.strings.regex_replace(vals, '[^0123456789]', '')
    vals = tf.strings.to_number(vals, tf.int32)

    return (
        tf.round(image_size[0] * vals[2] / vals[0]),
        tf.round(image_size[0] * vals[4] / vals[0]),
        tf.round(image_size[1] * vals[3] / vals[1]),
        tf.round(image_size[1] * vals[5] / vals[1]),
    )

def paths_and_labels_to_dataset(
    xml_paths,
    image_paths,
    image_size,
    labels,
    label_mode,
    num_classes,
    interpolation
):
    xmlp_ds = dataset_ops.Dataset.from_tensor_slices(xml_paths)
    bbox_ds = xmlp_ds.map(lambda x: path_to_bbox(x, image_size))
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    img_ds = path_ds.map(lambda x: path_to_image(x, image_size, interpolation))
    if label_mode:
        label_ds = dataset_utils.labels_to_dataset(labels, label_mode, num_classes)
    img_ds = dataset_ops.Dataset.zip((img_ds, label_ds, bbox_ds))
    return img_ds

def image_dataset_from_directory(
    directory_xml,
    directory_jpg,
    labels='inferred',
    label_mode='int',
    color_mode='rgb',
    batch_size=32,
    image_size=(299, 299),
    shuffle=True,
):
    num_channels = 3
    interpolation = image_preprocessing.get_interpolation('bilinear')

    seed = np.random.randint(1e6)

    xml_paths, bboxes, _ = dataset_utils.index_directory(
        directory_xml,
        labels,
        formats='xml',
        shuffle=shuffle,
        seed=seed,)

    image_paths, labels, class_names = dataset_utils.index_directory(
        directory_jpg,
        labels,
        formats='jpg',
        shuffle=shuffle,
        seed=seed,)

    dataset = paths_and_labels_to_dataset(
        xml_paths = xml_paths,
        image_paths=image_paths,
        image_size=image_size,
        labels=labels,
        label_mode=label_mode,
        num_classes=len(class_names),
        interpolation='bilinear')
    if shuffle:
        # Shuffle locally at each iteration
        dataset = dataset.shuffle(buffer_size=batch_size * 8, seed=seed)
    dataset = dataset.batch(batch_size)
    # Users may need to reference `class_names`.
    dataset.class_names = class_names
    return dataset