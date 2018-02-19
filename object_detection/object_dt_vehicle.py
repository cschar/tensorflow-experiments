
# coding: utf-8

# # Object Detection Demo
# Welcome to the object detection inference walkthrough!  This notebook will walk you step by step through the process of using a pre-trained model to detect objects in an image. Make sure to follow the [installation instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md) before you start.

# # Imports

# In[2]:

print("Hey")
# exit()

import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
print("importing tf")
import tensorflow as tf
import zipfile


from collections import defaultdict
from io import StringIO

# from matplotlib import pyplot as plt
from PIL import Image
from object_detection.utils import ops as utils_ops

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")

try:
    from utils import label_map_util
    from utils import visualization_utils as vis_util
except UserWarning as e:
    print("user warning")



MODEL_NAME = 'ssd_mobilenet_v1_coco_2017_11_17'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
#PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 90



#Load a frozen tensorflow model into memory
print("Opening MODEL FILE: ", PATH_TO_CKPT)
detection_graph = tf.Graph()
#
#
#
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')


# ## Loading label map
# Label maps map indices to category names, so that when our convolution network predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility functions, but anything that returns a dictionary mapping integers to appropriate string labels would be fine

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


# ## Helper code
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)


from grabscreen import grab_screen
import cv2

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    while True:
        # ret, image_np = cap.read()
        screen = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,450))
        image_np = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)

        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
         # Each box represents a part of the image where a particular object was detected.
        boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = detection_graph.get_tensor_by_name('detection_scores:0')
        classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        (boxes, scores, classes, num_detections) = sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
          image_np,
          np.squeeze(boxes),
          np.squeeze(classes).astype(np.int32),
          np.squeeze(scores),
          category_index,
          use_normalized_coordinates=True,
          line_thickness=8)
        # output_dict = run_inference_for_single_image(sess, image_np, detection_graph)

        # # Visualization of the results of a detection.
        # vis_util.visualize_boxes_and_labels_on_image_array(
        # image_np,
        # output_dict['detection_boxes'],
        # output_dict['detection_classes'],
        # output_dict['detection_scores'],
        # category_index,
        # instance_masks=output_dict.get('detection_masks'),
        # use_normalized_coordinates=True,
        # line_thickness=8)
        #cv2.imshow('object detection', cv2.resize(image_np, (800,600)))

        cv2.imshow('window', image_np)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break