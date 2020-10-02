#!/usr/bin/env python
import os
import threading
import numpy as np

import cv2
from cv_bridge import CvBridge
import rospy
from sensor_msgs.msg import Image
from sensor_msgs.msg import RegionOfInterest

#
import agrotec_weed_detection.config as config
import agrotec_weed_detection.model as modellib
import agrotec_weed_detection.visualize as visualize
from agrotec_weed_detection.msg import Result

class Configuration(config.Config):
    # give the configuration a recognizable name
    NAME = "Configuration"
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.9
    
    def __init__(self, num_classes):
        self.NUM_CLASSES = num_classes
        super().__init__()

class Node(object):
    def __init__(self):
        # Get parameters
        self._publish_rate = rospy.get_param('~publish_rate', 100)
        self._visualization = rospy.get_param('~visualization', True)
        model_path = rospy.get_param('~model_path', './')
        class_names = rospy.get_param('~class_names', 'BG')

        self._class_names = class_names.split(", ")

        # Create configuration instance
        config = Configuration(len(self._class_names))
        config.display()

        # Create model object in inference mode.
        self._model = modellib.MaskRCNN(mode="inference", model_dir="./",
                                        config=config)

        self._model.load_weights(model_path, by_name=True)

        self._last_msg = None
        self._msg_lock = threading.Lock()

        self._class_colors = visualize.random_colors(len(self._class_names))

        # Create ROS topics
        self._result_pub = rospy.Publisher('~result', Result, queue_size=1)
        self._visual_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        self._camera_input = rospy.Subscriber('~input', Image,
                                 self._image_callback, queue_size=1)

    def run(self):
        rate = rospy.Rate(self._publish_rate)
        cv_bridge = CvBridge()

        while not rospy.is_shutdown():
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            if msg is not None:
                np_image = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')

                # Run detection
                results = self._model.detect([np_image], verbose=0)
                result = results[0]
                result_msg = self._build_result_msg(msg, result)
                self._result_pub.publish(result_msg)

                # Visualize results
                if self._visualization:
                    cv_result = self._visualize_cv(result, np_image)
                    image_msg = cv_bridge.cv2_to_imgmsg(cv_result, 'bgr8')
                    self._visual_pub.publish(image_msg)

            rate.sleep()

    def _build_result_msg(self, msg, result):
        result_msg = Result()
        result_msg.header = msg.header

        for i, (y1, x1, y2, x2) in enumerate(result['rois']):
            box = RegionOfInterest()
            box.x_offset = np.asscalar(x1)
            box.y_offset = np.asscalar(y1)
            box.height = np.asscalar(y2 - y1)
            box.width = np.asscalar(x2 - x1)
            result_msg.boxes.append(box)

            class_id = result['class_ids'][i]
            result_msg.class_ids.append(class_id)

            class_name = self._class_names[class_id]
            result_msg.class_names.append(class_name)

            score = result['scores'][i]
            result_msg.scores.append(score)

            mask = Image()
            mask.header = msg.header
            mask.height = result['masks'].shape[0]
            mask.width = result['masks'].shape[1]
            mask.encoding = "mono8"
            mask.is_bigendian = False
            mask.step = mask.width
            mask.data = (result['masks'][:, :, i] * 255).tobytes()
            result_msg.masks.append(mask)

        return result_msg

    def _visualize(self, result, image):
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        fig = Figure()
        canvas = FigureCanvasAgg(fig)
        axes = fig.gca()
        visualize.display_instances(image, result['rois'], result['masks'],
                                    result['class_ids'], self._class_names,
                                    result['scores'], ax = axes,
                                    class_colors = self._class_colors)
        fig.tight_layout()
        canvas.draw()
        result = np.fromstring(canvas.tostring_rgb(), dtype='uint8')

        _, _, w, h = fig.bbox.bounds
        result = result.reshape((int(h), int(w), 3))

        return result

    def _visualize_cv(self, result, image):
        image = visualize.display_instances_cv(image, result['rois'], result['masks'],
                                               result['class_ids'], self._class_names,
                                               result['scores'],
                                               class_colors=self._class_colors)

        return image

    def _image_callback(self, msg):
        rospy.logdebug("Get an image")
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()

def main():
    rospy.init_node('agrotec_detection')

    node = Node()
    node.run()

if __name__ == '__main__':
    main()
