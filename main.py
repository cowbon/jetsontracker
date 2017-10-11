#!/usr/bin/env python

from __future__ import print_function
import os
import cv2
import threading
import tensorflow as tf
import argparse
import numpy as np

from queue import Queue
from threading import Thread
from object_detection.utils import label_map_util, visualization_utils as vis_util
from CameraManager import *
from utils import visualization as vis

# Constants
detection_model = 'ssd_inception_v2_coco_11_06_2017'
ckpt_path = os.path.join('models', detection_model, 'frozen_inference_graph.pb')
label_path = 'object_detection/data/mscoco_label_map.pbtxt'
category_index = None
recognition_model = ''
NUM_CLASSES = 90
nowindow = False


def get_local_path(model_path):
    return


def initModel(model):
    global category_index
    graph = tf.Graph()
    with graph.as_default():
        with tf.gfile.FastGFile(model, 'rb') as f:
            detection_graph = tf.GraphDef()
            detection_graph.ParseFromString(f.read())
            tf.import_graph_def(detection_graph, name='')

    label_map = label_map_util.load_labelmap(label_path)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
        use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(graph=graph, config=config)
    return (graph, sess)


def object_detection(image, graph, sess):
    image_tensor = graph.get_tensor_by_name('image_tensor:0')
    dim_expanded = np.expand_dims(image, axis=0)

    boxes = graph.get_tensor_by_name('detection_boxes:0')
    scores = graph.get_tensor_by_name('detection_scores:0')
    classes = graph.get_tensor_by_name('detection_classes:0')
    num_detections = graph.get_tensor_by_name('num_detections:0')

    # Execute the flow
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: dim_expanded}
    )

    # vis.drawRects(image, rects)
    if nowindow:
        rects = vis.get_boxes_and_labels(
	    boxes=np.squeeze(boxes),
	    classes=np.squeeze(classes).astype(np.int32),
	    scores=np.squeeze(scores),
	    category_index=category_index,
	    min_score_thresh=.5
        )

        return rects

    else:
        # Visualization of the results of a detection.
        vis_util.visualize_boxes_and_labels_on_image_array(image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=8)

        return image, dict(rects=boxes, classes=classes)


def detection_thread(graph, sess, in_queue, out_queue):
    print('Thread begins to run')
    while True:
	if in_queue.empty():
	    continue
        image = in_queue.get()
        if nowindow:
            outcome = object_detection(image, graph, sess)
            out_queue.put(outcome)
        else:
            outcome, _ = object_detection(image, graph, sess)
            out_queue.put(outcome)


def main(args):
    
    # Initialize model for detection
    if args.model is not None:
        graph, sess = initModel(args.model)
    else:
        graph, sess = initModel(ckpt_path)
    

    # Select image or real-time video
    if args.src:
        image = cv2.imread(args.src)
        outcome, _ = object_detection(image, graph, sess)
        cv2.imwrite('result_'+os.path.basename(args.src), outcome)
    else:
        # Run as a background thread
        # Will be refactored as a independent module to support multicamera
        cm = CameraManager(args.config)

        src = cm.get_device(0)
        cap = cv2.VideoCapture(src)

        if cap.isOpened() is False:
            # TODO: Write bad news to log
            print('Open camera failed!')
            exit()
        else:
            global nowindow
            if args.nowindow:
               nowindow = True

            in_queue = Queue(30)
            out_queue = Queue()
            t = Thread(target=detection_thread, args=[graph, sess, in_queue, out_queue])
	    t.daemon = True
            t.start()

            try:
                while True:
                    _, image = cap.read()
                    in_queue.put(image)
		    if out_queue.empty():
			pass
		    else:
                        outcome = out_queue.get()
		        #outcome, _ = object_detection(image, graph, sess)
                        if nowindow:
                            print('------------------------')
                            for out in outcome:
                                print(out[0], out[1])
                        else:
                            cv2.imshow('Object Detect Test', outcome)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            exit()
            except KeyboardInterrupt:
                exit()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--src',
        type=str,
        help='Image source',
        action='store'
    )

    parser.add_argument('--model',
        type=str,
        help='Path to the model',
        action='store'
    )

    parser.add_argument('--config',
        type=str,
        help='Path to configuration'
    )

    parser.add_argument('--no-display',
        help='Supress the resulr window',
        dest='nowindow',
        action='store_true'
    )

    args = parser.parse_args()
    main(args)
    '''
    # Initialization
    frame_queue = Queue.queue()
    detection_thread = Thread(target='detector', args=(frame_queue, running))
    detection_thread.start()
    while running is True:
        running = False
    '''
