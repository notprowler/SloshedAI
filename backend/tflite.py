import tensorflow as tf
import tensorflow_hub as hub
from tensorflow_docs.vis import embed
import numpy as np
import cv2
import time

#@title Helper functions for visualization

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a single color (white).
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'w',
    (0, 2): 'w',
    (1, 3): 'w',
    (2, 4): 'w',
    (5, 7): 'w',
    (7, 9): 'w',
    (6, 8): 'w',
    (8, 10): 'w',
    (5, 6): 'w',
    (5, 11): 'w',
    (6, 12): 'w',
    (11, 12): 'w',
    (11, 13): 'w',
    (13, 15): 'w',
    (12, 14): 'w',
    (14, 16): 'w'
}

def _keypoints_and_edges_for_display(keypoints_with_scores,
                                     height,
                                     width,
                                     keypoint_threshold=0.11):
  """Returns high confidence keypoints and edges for visualization.

  Args:
    keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
      the keypoint coordinates and scores returned from the MoveNet model.
    height: height of the image in pixels.
    width: width of the image in pixels.
    keypoint_threshold: minimum confidence score for a keypoint to be
      visualized.

  Returns:
    A (keypoints_xy, edges_xy, edge_colors) containing:
      * the coordinates of all keypoints of all detected entities;
      * the coordinates of all skeleton edges of all detected entities;
      * the colors in which the edges should be plotted.
  """
  keypoints_all = []
  keypoint_edges_all = []
  edge_colors = []
  num_instances, _, _, _ = keypoints_with_scores.shape
  for idx in range(num_instances):
    kpts_x = keypoints_with_scores[0, idx, :, 1]
    kpts_y = keypoints_with_scores[0, idx, :, 0]
    kpts_scores = keypoints_with_scores[0, idx, :, 2]
    kpts_absolute_xy = np.stack(
        [width * np.array(kpts_x), height * np.array(kpts_y)], axis=-1)
    kpts_above_thresh_absolute = kpts_absolute_xy[
        kpts_scores > keypoint_threshold, :]
    keypoints_all.append(kpts_above_thresh_absolute)

    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
      if (kpts_scores[edge_pair[0]] > keypoint_threshold and
          kpts_scores[edge_pair[1]] > keypoint_threshold):
        x_start = kpts_absolute_xy[edge_pair[0], 0]
        y_start = kpts_absolute_xy[edge_pair[0], 1]
        x_end = kpts_absolute_xy[edge_pair[1], 0]
        y_end = kpts_absolute_xy[edge_pair[1], 1]
        line_seg = np.array([[x_start, y_start], [x_end, y_end]])
        keypoint_edges_all.append(line_seg)
        edge_colors.append(color)
  if keypoints_all:
    keypoints_xy = np.concatenate(keypoints_all, axis=0)
  else:
    keypoints_xy = np.zeros((0, 17, 2))

  if keypoint_edges_all:
    edges_xy = np.stack(keypoint_edges_all, axis=0)
  else:
    edges_xy = np.zeros((0, 2, 2))
  return keypoints_xy, edges_xy, edge_colors


def draw_prediction_on_image(
    image, keypoints_with_scores, crop_region=None, close_figure=False,
    output_image_height=None):
    """Draws the keypoint predictions on image using OpenCV.

    Args:
        image: A numpy array with shape [height, width, channel] representing the
          pixel values of the input image.
        keypoints_with_scores: A numpy array with shape [1, 1, 17, 3] representing
          the keypoint coordinates and scores returned from the MoveNet model.
        crop_region: A dictionary that defines the coordinates of the bounding box
          of the crop region in normalized coordinates (see the init_crop_region
          function below for more detail). If provided, this function will also
          draw the bounding box on the image.
        output_image_height: An integer indicating the height of the output image.
          Note that the image aspect ratio will be the same as the input image.

    Returns:
        A numpy array with shape [out_height, out_width, channel] representing the
        image overlaid with keypoint predictions.
    """
    height, width, _ = image.shape
    keypoint_locs, keypoint_edges, edge_colors = _keypoints_and_edges_for_display(
        keypoints_with_scores, height, width)

    # Draw keypoints
    for keypoint in keypoint_locs:
        cv2.circle(image, (int(keypoint[0]), int(keypoint[1])), 5, (255, 255, 255), -1)  

    # Draw edges
    for edge in keypoint_edges:
        cv2.line(image, (int(edge[0][0]), int(edge[0][1])), (int(edge[1][0]), int(edge[1][1])), (255, 255, 255), 2)  

    if crop_region is not None:
        xmin = max(crop_region['x_min'] * width, 0.0)
        ymin = max(crop_region['y_min'] * height, 0.0)
        rec_width = min(crop_region['x_max'], 0.99) * width - xmin
        rec_height = min(crop_region['y_max'], 0.99) * height - ymin
        cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmin + rec_width), int(ymin + rec_height)), (255, 0, 0), 1)

    if output_image_height is not None:
        output_image_width = int(output_image_height / height * width)
        image = cv2.resize(image, (output_image_width, output_image_height), interpolation=cv2.INTER_CUBIC)

    return image

model_name = "movenet_thunder_f16.tflite"
input_size = 256

interpreter = tf.lite.Interpreter(model_path="thunder.tflite")
interpreter.allocate_tensors()

def movenet(input_image):
    """Runs detection on an input image.

    Args:
      input_image: A [1, height, width, 3] tensor represents the input image
        pixels. Note that the height/width should already be resized and match the
        expected input resolution of the model before passing into this function.

    Returns:
      A [1, 1, 17, 3] float numpy array representing the predicted keypoint
      coordinates and scores.
    """
    # TF Lite format expects tensor type of uint8.
    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    
    start_time = time.time()  # Start time before inference
    interpreter.invoke()
    end_time = time.time()  # End time after inference
    
    inference_time = (end_time - start_time) * 1000  # Convert to milliseconds
    print(f"Inference time: {inference_time:.2f} ms")
    
    keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
    return keypoints_with_scores

