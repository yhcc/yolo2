from PIL import Image
import numpy as np
import cv2

anchors_value = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]],dtype='float32')
anchors_length = len(anchors_value)

def read_data(filepath, nb_classes, target_image_size=(416, 416)):
	"""
	read data and return X,y. X will in the shape (None, image_size[0], image_size[1], 3).
		y1: (None, 5)
		y2: (None, image_size[0], image_size[1], 5, 1)
		y3: (None, image_size[0], image_size[1], 5, 5)
	para:
		filepath: ex:/images/imagelist.txt should have the form
			image_path, x, y, w, h, class
			image_path, x, y, w, h, class
			...
			ex:
			images/image1, 120, 30, 50, 20, 9
			...
	"""
	f = open(filepath)
	line = f.readline()
	X = []
	y1 = []
	y2 = []
	y3 = []
	while line:
		data = line.strip().split(',')
		X.append(read_image(data[0],target_image_size))
		image = Image.open(data[0])
		image_size = [image.width, image.height]
		object_mask, object_value = convert_ground_truth(float(data[1:]), image_size=image_size)
		y1.append(np.asarray(data[1:]))
		y2.append(object_mask)
		y3.append(object_value)

	return np.asarray(X, dtype='float32'), np.asarray(y1, dtype='float32'), \
		np.asarray(y2, dtype='float32'), np.asarray(y3,dtype='float32')


def data_generator(imageFile, batch_size=16, nb_classes=80, target_image_size=(416, 416)):
	"""
	used in fit_generator
	the yield data is ([X,y1,y2,y3],np.ones())
	"""
	f = open(imageFile)
	count = 0
	while True:
		line = f.readline()
		if count==0:
			X = []
			y1 = []
			y2 = []
			y3 = []
		while line:
			data = line.strip().split(',')
			X.append(read_image(data[0], target_image_size))
			image = Image.open(data[0])
			image_size = [image.width, image.height]
			array = []
			for num in data[1:]:
				array.append(float(num))
			array = np.asarray(array)
			object_mask, object_value = convert_ground_truth(array, image_size=image_size)
			y1.append(array)
			y2.append(object_mask)
			y3.append(object_value)
			count += 1
			if count==batch_size:
				break
			line = f.readline()
		if count==batch_size and line=='':
			yield [np.asarray(X, dtype='float32'), np.asarray(y1, dtype='float32'), \
				np.asarray(y2, dtype='float32'), np.asarray(y3,dtype='float32')], \
				np.ones((batch_size,),dtype='float32')
			f.seek(0)
			count = 0
		elif count==batch_size:
			yield [np.asarray(X, dtype='float32'), np.asarray(y1, dtype='float32'), \
				np.asarray(y2, dtype='float32'), np.asarray(y3,dtype='float32')], \
				np.ones((batch_size,),dtype='float32')
			count = 0
		elif line=='':
			f.seek(0)		


def read_image(path, image_size=(416,416)):
	"""
	read a single image and return shape as (416, 416, 3).
	data in the range (0,1)
	"""
	image = Image.open(path)
	if image_size is not None:
		resized_image = image.resize(
			tuple(reversed(image_size)), Image.BICUBIC)
		image_data = np.array(resized_image, dtype='float32')
	else:
		# Due to skip connection + max pooling in YOLO_v2, inputs must have
		# width and height as multiples of 32.
		new_image_size = (image.width - (image.width % 32),
					image.height - (image.height % 32))
		resized_image = image.resize(new_image_size, Image.BICUBIC)
		image_data = np.array(resized_image, dtype='float32')
	    
	image_data /= 255.

	return image_data

def convert_ground_truth(box, image_size):
	"""
	This function is used to convert boxes drawn by human to help determine loss
	para:
		true_boxes: list of box info, include x,y,w,h,class. x,y,w,h should be absolute 
			pixel postion
			ex:
				130, 140, 30, 40, 1
		image_size: the size of an image. shoud in order w, h 
	return:
		object_mask: shape (13, 13, 5, 1), help to determine which anchor capture the right box
		object_value: shape (13, 13, 5, 5 ). values like x,y,w,h and class for 
			bounding box is saved in this variable. x, y corresponds to sigmoid(x), sigmoid(y),
			w, h correponds to bw, bh
	"""
	anchors_length = len(anchors_value)
	half = anchors_value / 2.
	half = np.asarray(half, dtype='float32')
	anchors_min = -half
	anchors_max = half
	anchors_areas = half[:,1]*half[:,0]*4
	width, height = image_size

	#TODO change here to make it suitable for other image size
	object_mask = np.zeros((13, 13, anchors_length, 1))
	object_value = np.zeros((13, 13, anchors_length, 5))
	#object_mask = []
	#object_value = []

	box_wh = box[2:4]/np.array([width/13., height/13.])#32 is for downsample factor 32, may change in other net structure
	half = box_wh / 2
	box_half = np.repeat(np.asarray(half, dtype='float32').reshape((1,2)), anchors_length, axis=0)
	box_min = -box_half
	box_max = box_half
	intersect_min = np.minimum(box_min, anchors_min)
	intersect_max = np.maximum(box_max, anchors_max)
	intersect_box = np.maximum(intersect_max-intersect_min, 0.)
	intersect_areas = intersect_box[:, 0]*intersect_box[:, 1]
	box_areas = box_half[:,0]*box_half[:,1]*4
	iou = intersect_areas/(box_areas+anchors_areas-intersect_areas)
	maximum_iou = np.max(iou)
	if maximum_iou>0:
		index = np.argmax(iou)
		x = (box[0]+box[2]/2)/float(width)
		y = (box[1]+box[3]/2)/float(height)
		#not sure which is right. use bw, bh right now
		#w = np.log(box[2]/float(image_size[0])/anchors[index][0])
		#h = np.log(box[3]/float(image_size[1])/anchors[index][1])
		w = box[2]/float(width)
		h = box[3]/float(height)
		object_mask[np.int((box[0]+box[2]/2)/(width/13.)), \
				np.int((box[1]+box[3]/2)/(height/13.)), index, 0] = 1
		#object_mask.append([np.floor(box[0]/32.), np.floor(box[1]/32.), index])
		object_value[np.int((box[0]+box[2]/2)/(width/13.)), \
				np.int((box[1]+box[3]/2)/(height/13.)), index] = [x,y,w,h,box[4]]
		#object_value.append([np.floor(box[0]/32.), np.floor(box[1]/32.), index, x,y,w,h,box[4]])

	return object_mask, object_value