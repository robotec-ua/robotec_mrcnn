# The ROS Package of Mask R-CNN for Weeds Detection and Segmentation
This is a ROS package of [Mask R-CNN](https://arxiv.org/abs/1703.06870) algorithm for object detection and segmentation. It can run object detection on different keras weights (*.h5) using simplicy of ROS to make the package part of a bigger system.

The purpose of this package is to provide an CNN ROS package for detection of weeds (useful for high-end agricultural robots and )

## Credits
* Original project by Matterport Inc. : [Mask_RCNN](https://github.com/matterport/Mask_RCNN)
* Original ROS package by qixuxiang : [mask_rcnn_ros](https://github.com/qixuxiang/mask_rcnn_ros)

## Requirements
* ROS Melodic (Python 3 version) and higher
* TensorFlow 1.3+
* Keras 2.0.8+
* Numpy, skimage, scipy, Pillow, cython, h5py
* Python 3.6+
* see more dependencies and version details in [requirements.txt](https://github.com/qixuxiang/mask_rcnn_ros/blob/master/requirements.txt)

## ROS Interfaces
This part describes various ROS-related interfaces such as parameters and topics.

### Parameters

* `~model_path: string`

    Path to the HDF5 model file.
    If the model_path is default value and the file doesn't exist, the node automatically downloads the file.

    Default: `$ROS_HOME/mask_rcnn_coco.h5`

* `~visualization: bool`

    If true, the node publish visualized images to `~visualization` topic.
    Default: `true`

* `~class_names: string[]`

    Class names to be treated as detection targets.
    Default: `['BG']`.

### Topics Published

* `~result: agrotec_weed_detection/Result`

    Result of detection. See also `Result.msg` for detailed description.

* `~visualization: sensor_mgs/Image`

    Visualized result over an input image.


### Topics Subscribed

* `~input: sensor_msgs/Image`

    Input image to be proccessed

## Project structure
```
.
├── bags
├── doc
├── launch
│   └── detection.launch
├── msg
│   └── Result.msg
├── rviz
│   └── mask_rcnn_ros.rviz
├── scripts
│   ├── __init__.py
│   └── mrcnn.py
├── src
│   └── agrotec_mrcnn
│       ├── config.py
│       ├── __init__.py
│       ├── model.py
│       ├── parallel_model.py
│       ├── utils.py
│       └── visualize.py
├── CMakeLists.txt
├── LICENSE
├── package.xml
├── README.md
├── requirements.txt
└── setup.py
```

### Folders
*  `./bags`  - folder for data, recorded by Rosbag package
*  `./doc`  - folder contains some documentation an examples of work
*  `./msg`  - default folder for ROS messages
*  `./rviz`  - files for vizualization through RViz
*  `./scripts`  - main folder of the project (contains executables)
*  `./src/agrotec_weed_detection`  - local Mask RCNN implementation

### Files
The project uses following files

#### Launch
*  `weed_detection.launch`  - launching the example detection

#### Messages
*  `Result.msg`  - message type for delivering detection results 

#### RVIz
*  `mask_rcnn_ros.rviz`  - RViz visualization configuration

#### Package
*  `mrcnn.py`  - NN-based detection implementation

#### ROS
*  `CMakeLists.txt`  - package building instructions
*  `package.xml`  - data about the package
*  `setup.py`  - rospy instructions


## Getting Started
0. If you are using Ubuntu, you should consider installing some of the required dependencies through apt
```
$ sudo apt-get install python3-matplotlib python3-opencv python3-scipy pythin3-sklearn python3-skimage python3-ipython python3-numpy python3-h5py
```

1. Clone this repository to your catkin workspace, build workspace and source devel environment 
```
$ cd ~/.catkin_ws/src
$ git clone https://github.com/robotec-ua/agrotec_weed_detection.git
$ cd agrotec_mrcnn
$ python3 -m pip install --upgrade pip
$ python3 -m pip install -r requirements.txt
$ cd ../..
$ catkin_make
$ source devel/setup.bash

```
2. Set up your environment
        To do this, you should prepare your ROS package for camera and write the proper topics' names and parameters in the detection launchfile (you can see the example in the `launch` folder). For example, you should change `~class_names` and `~model_path` to use your own weights, and `~input` to the name of the topic where you are going to publish the messages.

3. Run mask_rcnn node
      ~~~bash
      $ roslaunch agrotec_mrcnn detection.launch
      ~~~

## Example of use
This part is dedicated for showing how to use the package.

Assuming that you are already in the project directory :
~~~bash
$ mkdir weights
$ cd weights
~~~

And download the weights you want to use inside the folder. 

Make sure that you've already changed the `~input` topic in `launch/detection.launch`. Then the last step is to start the detection:
~~~bash
$ roslaunch agrotec_mrcnn weed_detection.launch
~~~

If you want to visualize the detection, you should open a new tab/window in your terminal and run :
~~~bash
$ rosrun image_view image_view image:=<your topic>
~~~