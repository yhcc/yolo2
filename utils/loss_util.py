import tensorflow as tf
import numpy as np
import keras.backend as K
from convert_result import convert_result

anchors_value = np.array(
    [[1.08, 1.19], [3.42, 4.41], [6.63, 11.38], [9.42, 5.11], [16.62, 10.52]],dtype='float32')
nb_classes = 80
anchors_length = len(anchors_value)

#used for final result

def loss_function(args):
	#should work fun
	"""
	y_true (batch, 13, 13, 425) tensor
	y1 (batch, 5) tensor
	y2 (batch, 13,13,5,1)
	y3 (batch, 13,13,5,5)
	"""
	y_pred, y1, y2, y3 = args
	#converted_result = convert_result(y_pred, anchors, nb_classes)
	return loss_calculator(y_pred, y2, y3)


def loss_calculator(output, object_mask, object_value):
	"""
	calculate loss on the basis of a batch
	para:
		output: output by the net. (13, 13, 485)
		anchors: list of anchor info. value is correspoding length. (that's value*32 convert
			back to absolute pixel values)
			ex:
				0.7684, 0.9980
				1.3340, 3.1890
				.....
		object_mask: shape(batch_size, 13, 13, 5, 1), with entry equals 1 means this anchor is the 
			right one. 1obj in loss equation
		object_value: shape(batch_size, 13, 13, 5, 5), indicates the x, y, w, h and class for the 
			right box
	"""

	#use convert_result to convert output. bxy is bx, by. 
	bxy, bwh, to, classes = convert_result(output, anchors_value, nb_classes)

	#leave the ratio unassigned right now
	alpha1 = 5.0
	alpha2 = 5.0
	alpha3 = 1.0
	alpha4 = 0.5
	alpha5 = 1.0

	#first term coordinate_loss
	bxy_sigmoid = bxy - tf.floor(bxy)
	bxy_loss = K.sum(K.square(bxy_sigmoid - object_value[...,0:2])*object_mask)

	#second term
	bwh_loss = K.sum(K.square(K.sqrt(bwh)-K.sqrt(object_value[...,2:4]))*object_mask)

	#third term
	to_obj_loss = K.sum(K.square(1-to)*object_mask)

	#forth term. TODO, need to multiply another factor.  (1 - object_detection)
	to_noobj_loss = K.sum(K.square(0-to)*(1-object_mask))

	#fifth term
	onehot_class = K.one_hot(tf.to_int32(object_value[...,4]), nb_classes)
	class_loss = K.sum(K.square(onehot_class-classes)*object_mask)

	#total loss
	result = alpha1*bxy_loss + alpha2*bwh_loss + alpha3*to_obj_loss + \
			alpha4*to_noobj_loss + alpha5*class_loss

	return result

