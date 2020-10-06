#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image

import cv2
from cv_bridge import CvBridge
import numpy as np
import threading

class Node():
    def __init__(self):
        # Get ROS parameters
        self._publish_rate = rospy.get_param('~publish_rate', 100)
            
        # Create ROS topics
        self._visual_pub = rospy.Publisher('~visualization', Image, queue_size=1)
        self._camera_input = rospy.Subscriber('~input', Image,
                                 self.imageCallback, queue_size=1)

        # Create multitreading locks
        self._last_msg = None
        self._msg_lock = threading.Lock()

        # Set range for green color
        self._lower = np.array([25, 52, 72], np.uint8) 
        self._upper = np.array([102, 255, 255], np.uint8) 


    def run(self):
        rate = rospy.Rate(self._publish_rate)   # Rate of the main loop
        cv_bridge = CvBridge()

        while not rospy.is_shutdown():
            # If there is no lock on the message (not being written to in the moment)
            if self._msg_lock.acquire(False):
                msg = self._last_msg
                self._last_msg = None
                self._msg_lock.release()
            else:
                rate.sleep()
                continue

            # If the message is not empty
            if msg is not None:
                # Convert the message to an OpenCV object
                np_image = cv_bridge.imgmsg_to_cv2(msg, 'bgr8')
                hsvFrame = cv2.cvtColor(np_image, cv2.COLOR_BGR2HSV)

                # Obtain masks for colored objects
                mask = cv2.inRange(hsvFrame, self._lower, self._upper)
                kernal = np.ones((5, 5), "uint8")
                mask = cv2.dilate(mask, kernal)

                # Creating contour to track green color 
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
      
                for pic, contour in enumerate(contours): 
                    area = cv2.contourArea(contour) 

                    if (area > 500): 
                        x, y, w, h = cv2.boundingRect(contour)
                        np_image = cv2.rectangle(np_image, (x, y),  
                                    (x + w, y + h),
                                    (0, 255, 0), 2)

                image_msg = cv_bridge.cv2_to_imgmsg(np_image, 'bgr8')                
                self._visual_pub.publish(image_msg)
        
    def imageCallback(self, msg):
        if self._msg_lock.acquire(False):
            self._last_msg = msg
            self._msg_lock.release()


def main():
    #
    rospy.init_node('agrotec_weed_filtering')

    #
    node = Node()
    node.run()

if __name__ == '__main__':
    main()
