# -*- coding: utf-8 -*-
"""Ptyxiaki.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/12oRI4GwiwUErWqkKOSuRTpEBcPJlifEn

## Road Damage Detection
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import sys

sys.path.append("../ATOM/utils")

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from utils import label_map_util

from utils import visualization_utils as vis_util

# opencv stuff
print(tf.__version__)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


def detect(test_image, plot_show=False):
    # Path to frozen detection graph. This is the actual model that is used for the object detection.
    PATH_TO_CKPT = 'ssd_mobilenet_RoadDamageDetector.pb'

    # List of the strings that is used to add correct label for each box.
    PATH_TO_LABELS = 'crack_label_map.pbtxt'

    NUM_CLASSES = 8
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                                use_display_name=True)
    category_index = label_map_util.create_category_index(categories)

    IMAGE_SIZE = (12, 8)
    final_images, boxes_collect, scores_collect, classes_collect, num_collect = [], [], [], [], []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')

            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.

            image_np = test_image
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            print(image_np_expanded.shape)
            # Actual detection.
            (boxes, scores, classes, num) = sess.run(
                [detection_boxes, detection_scores, detection_classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(boxes),
                np.squeeze(classes).astype(np.int32),
                np.squeeze(scores),
                category_index,
                min_score_thresh=0.4,
                use_normalized_coordinates=True,
                line_thickness=8)
            if plot_show == True:
                plt.figure(figsize=IMAGE_SIZE)
                plt.imshow(image_np)
            final_images.append(image_np)
            boxes_collect.append(boxes)
            scores_collect.append(scores)
            classes_collect.append(classes)
            num_collect.append(num)

    return image_np, boxes_collect, scores_collect, classes_collect, num_collect

