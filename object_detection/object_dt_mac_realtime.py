
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

print("resetting default graph")
tf.reset_default_graph()



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



MODEL_NAME = 'mac_n_cheese_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
# PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')
PATH_TO_LABELS = os.path.join('training', 'object-detection.pbtxt')

NUM_CLASSES = 1



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

# with detection_graph.as_default():
#   with tf.Session(graph=detection_graph) as sess:
#     while True:
#         # ret, image_np = cap.read()
#         image_np = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,450))
#         image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
#         # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
#         image_np_expanded = np.expand_dims(image_np, axis=0)

#         image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
#          # Each box represents a part of the image where a particular object was detected.
#         boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
#         # Each score represent how level of confidence for each of the objects.
#         # Score is shown on the result image, together with the class label.
#         scores = detection_graph.get_tensor_by_name('detection_scores:0')
#         classes = detection_graph.get_tensor_by_name('detection_classes:0')
#         num_detections = detection_graph.get_tensor_by_name('num_detections:0')
#         # Actual detection.

#         (boxes, scores, classes, num_detections) = sess.run(
#             [boxes, scores, classes, num_detections],
#             feed_dict={image_tensor: image_np_expanded})

#         print("num detections", num_detections)
#         print("classes", classes)


#         # print(boxes, scores, len(classes), num_detections)

#         # Visualization of the results of a detection.
#         vis_util.visualize_boxes_and_labels_on_image_array(
#           image_np,
#           np.squeeze(boxes),
#           np.squeeze(classes).astype(np.int32),
#           np.squeeze(scores),
#           category_index,
#           use_normalized_coordinates=True,
#           line_thickness=8)

#         cv2.imshow('window', image_np)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break

def run_inference_for_single_image(image, graph, sess):
      # Get handles to input and output tensors
  if True:
    if True:
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict    


PATH_TO_TEST_IMAGES_DIR = 'test_images'
TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR,
                                  'image{}.jpg'.format(i)) for i in range(13, 16) ]

#https://www.youtube.com/watch?v=VcnEYI7wNh0
# mac n cheese real time screen capture

with detection_graph.as_default():
  with tf.Session(graph=detection_graph) as sess:
    i = 13
    while True:

        image_np = cv2.resize(grab_screen(region=(0,40,1280,745)), (800,450))
        

        # image = Image.open(TEST_IMAGE_PATHS[i - 13])
        # image_np = load_image_into_numpy_array(image)
        print(image_np.shape)
        # image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        i += 1
        if i % 16 == 0:
            i = 13
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        # image_np_expanded = np.expand_dims(image_np, axis=0)

        x = 1
        # x = input("read in input for {}".format(i))
        if x == 'q':
            cv2.destroyAllWindows()
            break
        # image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        #  # Each box represents a part of the image where a particular object was detected.
        # boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        # # Each score represent how level of confidence for each of the objects.
        # # Score is shown on the result image, together with the class label.
        # scores = detection_graph.get_tensor_by_name('detection_scores:0')
        # classes = detection_graph.get_tensor_by_name('detection_classes:0')
        # num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.

        output_dict = run_inference_for_single_image(image_np, detection_graph, sess)
        # print(output_dict.keys())
        print(len(output_dict['detection_boxes']),
            len(output_dict['detection_classes']),
            len(output_dict['detection_scores']))
        print(category_index)
  # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(
            image_np,        
            output_dict['detection_boxes'],
            output_dict['detection_classes'],
            output_dict['detection_scores'],
            category_index,
            instance_masks=output_dict.get('detection_masks'),
            use_normalized_coordinates=True,
            line_thickness=8)

        image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2RGB)
        cv2.imshow('window', image_np)
        # cv2.imshow('object detection', cv2.resize(image_np, (800,600)))
        if cv2.waitKey(25) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break