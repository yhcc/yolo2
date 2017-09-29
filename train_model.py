
import argparse
from keras.models import load_model
from keras.layers import Lambda, Input
from keras.models import Model
from keras.optimizers import SGD
from utils.loss_util import loss_function
from utils.data_util import data_generator

def model_to_train(model_path, optimizer=SGD(0.0001), weight_path=None, anchor_length=5):
	"""
	helper function to prepare to train
	para:
		model_path: path to model
		optimizer: keras Optimizer instance. Ex. SGD(0.0001), RMsprop(0.001) etc
		weight_path: where to load weight, default None, use model_path's weight
	return:
		model: use to fit. However input should be [y_pred,y1_, y2_, y3_]
	"""
	darknet = load_model(model_path)
	if weight_path!=None:
		darknet.load_weights(weight_path)
	image_size = darknet.layers[0].input_shape[1:3]
	y1_ = Input(shape=(5,))#x,y,w,h
	y2_ = Input(shape=(image_size[0]//32, image_size[1]//32, anchor_length, 1))#object_mask
	y3_ = Input(shape=(image_size[0]//32, image_size[1]//32, anchor_length, 5))#object_value
	image_input = Input(shape=(image_size[0],image_size[1],3))
	y_pred = darknet(image_input)

	loss_out = Lambda(loss_function, output_shape=(1,))([y_pred,y1_, y2_, y3_])

	model = Model(inputs=[image_input, y1_, y2_, y3_],outputs=[loss_out])

	# TODO: change to other optimizer
	model.compile(optimizer=optimizer,loss=lambda x,y: x*y)

	return model, darknet

def train_model(model_path, imageFile, samples_per_epoch=160, save_path='new_model.h5',
	nb_epoch=20, nb_classes=80,weight_path=None, batch_size=16, optimizer=SGD(0.0001), 
	val_imageFile=None, nb_val_samples=None, anchor_length=5, image_size=(416, 416)):
	"""
	Use to train model. use fit_generator by default
	para:
		imageFile: ex:/images/imagelist.txt should have the form. origin in top left
			image_path, x, y, w, h, class
			image_path, x, y, w, h, class
			...
			ex:
			images/image1, 120, 30, 50, 20, 9
		model_path: path to model
		optimizer: keras Optimizer instance. Ex. SGD(0.0001), RMsprop(0.001) etc
		weight_path: where to load weight, default None, use model_path's weight		
		val_imageFile: same as imageFile but is used for validation
	return:
		the original model. that's output is (13, 13, 425)
	"""
	if isinstance(model_path, str):
		model = model_to_train(model_path, optimizer, weight_path, anchor_length)
	else:
		model = model_path[0]
	if val_imageFile!=None:
		val = data_generator(val_imageFile, batch_size, nb_classes, image_size)
	else:
		val = None
	try:
		"""
		In case of exception, save the model
		"""
		model.fit_generator(
			data_generator(imageFile, batch_size, nb_classes, image_size),
			samples_per_epoch=samples_per_epoch,
			nb_epoch=nb_epoch,
			verbose=1,
			validation_data=val,
			nb_val_samples=nb_val_samples)
	except Exception, e:
		print e
		#model.save_weights('exception.h5')
	if save_path!=None:
		model_path[1].save_weights(save_path)
	return model_path[1]

if __name__=='__main__':
	pass

#maybe for later use
parser = argparse.ArgumentParser(
	description='Used to train model')
parser.add_argument(
	'model_path',
	help='Where to load model')
parser.add_argument(
	'imageFile',
	help='Where to find txt file contain training image information')
parser.add_argument(
	'-spe',
	'--samples_per_epoch',
	help='samples per epoch, used in fit_generator')
parser.add_argument(
	'-sp',
	'--save_path',
	help='where to save the trained weights',
	default=None)
parser.add_argument(
	'-ne',
	'--nb_epoch',
	help='number of epoch to train',
	default=20)
parser.add_argument(
	'-nc',
	'--nb_classes',
	help='number of classes',
	default=80)
parser.add_argument(
	'-wp',
	'--weight_path',
	help='where to load weight',
	default=None)
parser.add_argument(
	'-bs',
	'--batch_size',
	help='batch size',
	default=16)
parser.add_argument(
	'-o',
	'--optimizer',
	help='what optimizer to use.',
	default='SGD')
parser.add_argument(
	'-lr',
	'--learning_rate',
	help='learning_rate',
	type=float,
	default=0.00001)
parser.add_argument(
	'-vi',
	'--val_imageFile',
	help='validation imageFile.txt',
	default=None)
parser.add_argument(
	'-nbs',
	'--nb_val_samples',
	help='number of validation image',
	default=None)
