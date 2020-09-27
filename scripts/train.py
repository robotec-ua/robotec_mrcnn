#
import agrotec_weed_detection.coco as coco
import agrotec_weed_detection.config as config
import agrotec_weed_detection.utils as utils

import os
from xml.etree import ElementTree
from numpy import zeros
from numpy import asarray
import pdb
import numpy as np
import skimage

# class that defines and loads the weed dataset
class WeedDataset(utils.Dataset):
	# load the dataset definitions
	def load_dataset(self, dataset_dir, is_train=True):
		# define classes
		self.add_class("dataset", 1, "crop")
		self.add_class("dataset", 2, "weed")
		# define data locations
		images_dir = dataset_dir + '/images/'
		annotations_dir = dataset_dir + '/annots/'
		# find all images
		for filename in os.listdir(images_dir):
			# extract image id
			image_id = filename[:-4]
			# skip bad images
			if (image_id == '.ipynb_checkpo'):
				continue
			# skip all images after 115 if we are building the train set
			if is_train and int(image_id) >= 115:
				continue
			# skip all images before 115 if we are building the test/val set
			if not is_train and int(image_id) < 115:
				continue
			img_path = images_dir + filename
			ann_path = annotations_dir + image_id + '.xml'
			# add to dataset
			self.add_image('dataset', image_id=image_id, path=img_path, annotation=ann_path)

	# extract bounding boxes from an annotation file
	def extract_boxes(self, filename):
		# load and parse the file
		tree = ElementTree.parse(filename)
		# get the root of the document
		root = tree.getroot()
		# extract each bounding box
		boxes = list()
		#for box in root.findall('.//bndbox'):
		for box in root.findall('.//object'):
			name = box.find('name').text
			xmin = int(box.find('./bndbox/xmin').text)
			ymin = int(box.find('./bndbox/ymin').text)
			xmax = int(box.find('./bndbox/xmax').text)
			ymax = int(box.find('./bndbox/ymax').text)
			#coors = [xmin, ymin, xmax, ymax, name]
			coors = [xmin, ymin, xmax, ymax, name]
			boxes.append(coors)
		# extract image dimensions
		width = int(root.find('.//size/width').text)
		height = int(root.find('.//size/height').text)
		return boxes, width, height

	# load the masks for an image
	def load_mask(self, image_id):
		#pdb.set_trace()
		# get details of image
		info = self.image_info[image_id]
		# define box file location
		path = info['annotation']
		# load XML
		boxes, w, h = self.extract_boxes(path)
		# create one array for all masks, each on a different channel
		masks = zeros([h, w, len(boxes)], dtype='uint8')
		# create masks
		class_ids = list()
		for i in range(len(boxes)):
			box = boxes[i]
			row_s, row_e = box[1], box[3]
			col_s, col_e = box[0], box[2]
			if (box[4] == 'crop'):
				masks[row_s:row_e, col_s:col_e, i] = 2
				class_ids.append(self.class_names.index('crop'))
			else:
				masks[row_s:row_e, col_s:col_e, i] = 1
				class_ids.append(self.class_names.index('weed'))
		return masks, asarray(class_ids, dtype='int32')

	# load an image reference
	def image_reference(self, image_id):
		info = self.image_info[image_id]
		return info['path']


class Configuration(config):
    # give the configuration a recognizable name
    NAME = "interference"
    # set the number of GPUs to use along with the number of images
    # per GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

def main():
    rospy.init_node('agrotec_network_training')

if __name__ == '__main__':
    main()
