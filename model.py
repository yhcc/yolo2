import numpy as np
from keras import backend as K
from keras.layers import (Convolution2D, GlobalAveragePooling2D, Input, Lambda,InputLayer,
                          MaxPooling2D, merge, Merge)
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.regularizers import l2

import argparse

#usage: 
#python model.py darknet -l True -w yolo.weights -smodel_data/test.h5 -i 416,416
#python model.py customized_model -l True -w yolo.weights -smodel_data/test.h5 -i 416,416 -f 1024

parser = argparse.ArgumentParser(
    description='Generate keras yolo2 model file')
parser.add_argument(
    'net',
    help='which function to use darknet or cutomized_model')
parser.add_argument(
    '-l',
    '--include_last',
    help='For darknet whether to include last convolution',
    type=bool,
    default=True)
parser.add_argument(
    '-w',
    '--weight_path',
    help='Where to read pretrained model weight',
    default='yolo.weights')
parser.add_argument(
    '-s',
    '--save_path',
    help='where to save model',
    default='model_data/model.h5')
parser.add_argument(
    '-i',
    '--image_shape',
    help='Input image shape',
    type=str,
    default='416,416')
parser.add_argument(
    '-f',
    '--filters',
    help='When use cutomized_model, the number of filters in last Convolution',
    default=425)

"""
Used to generate a model
"""
def darknet(include_last=True, weight_path='yolo.weights', image_shape=(416, 416), 
       save_path=None):
       """
       generate yolo2 model. 
       para:
              include_last: whether to include the last default convolution layer, which 
                     output (None, 13,13,425) by default. If this is False, output shape will
                     be (None, 13,13, 1024), you may need to add layer to it yourself.
              weight_path: where to find the pretrained yolo weights
              image_shape: the input image shape
              save_path: if None, only return model. Otherwise it will save model to this path 
                     as well. ex. 'model_data/model.h5'
       return:
              model: keras model

       """
       if save_path!=None:
              assert save_path.endswith('.h5'), 'Keras model must be saved as .h5 file.'

       image_input = Input(shape=(image_shape[0],image_shape[1],3),name='input_1')

       #read pretrained weights
       weights_file = open('yolo.weights','rb')
       count = 16

       #drop first 16 bytes
       weights_file.read(16)

       """
       convolution_0
       batch_normalize=1
       filters=32
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(32,),dtype='float32',buffer=weights_file.read(32*4))
       bn_list = np.ndarray(shape=(3, 32),dtype='float32',buffer=weights_file.read(32*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(32,3,3,3),dtype='float32',buffer=weights_file.read(32*3*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(32,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_1')(image_input)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_1')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_1')(tmp)

       count += (32 + 32*3 + 3*3*3*32)*4
       print "file read to",count


        

       """
       maxpool_0
       size=2
       stride=2
       """
       tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_1')(tmp)


        

       """
       convolutional_1
       batch_normalize=1
       filters=64
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(64,),dtype='float32',buffer=weights_file.read(64*4))
       bn_list = np.ndarray(shape=(3, 64),dtype='float32',buffer=weights_file.read(64*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(64,32,3,3),dtype='float32',buffer=weights_file.read(64*32*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(64,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_2')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_2')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_2')(tmp)

       #help to go back.   
       count += (64 + 64*3 + 3*3*64*32)*4
       print "file read to",count

       """
       maxpool_1
       size=2
       stride=2
       """
       tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_2')(tmp)


        

       """
       convolutional_2
       batch_normalize=1
       filters=128
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(128,),dtype='float32',buffer=weights_file.read(128*4))
       bn_list = np.ndarray(shape=(3, 128),dtype='float32',buffer=weights_file.read(128*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(128,64,3,3),dtype='float32',buffer=weights_file.read(128*64*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(128,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_3')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_3')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_3')(tmp)

       #help to go back.   
       count += (128 + 128*3 + 3*3*128*64)*4
       print "file read to",count


        

       """
       convolutional_3
       batch_normalize=1
       filters=64
       size=1
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(64,),dtype='float32',buffer=weights_file.read(64*4))
       bn_list = np.ndarray(shape=(3, 64),dtype='float32',buffer=weights_file.read(64*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(64,128,1,1),dtype='float32',buffer=weights_file.read(64*128*1*1*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(64,1,1,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_4')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_4')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_4')(tmp)

       #help to go back.   
       count += (64 + 64*3 + 1*1*64*128)*4
       print "file read to",count


        

       """
       convolutional_4
       batch_normalize=1
       filters=128
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(128,),dtype='float32',buffer=weights_file.read(128*4))
       bn_list = np.ndarray(shape=(3, 128),dtype='float32',buffer=weights_file.read(128*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(128,64,3,3),dtype='float32',buffer=weights_file.read(128*64*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(128,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_5')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_5')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_5')(tmp)

       #help to go back.   
       count += (128 + 128*3 + 3*3*128*64)*4
       print "file read to",count


        

       """
       maxpool_2
       size=2
       stride=2
       """
       tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_3')(tmp)


        

       """
       convolutional_5
       batch_normalize=1
       filters=256
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
       bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(256,128,3,3),dtype='float32',buffer=weights_file.read(256*128*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(256,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_6')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_6')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_6')(tmp)

       #help to go back.   
       count += (256 + 256*3 + 3*3*256*128)*4
       print "file read to",count


        

       """
       convolutional_6
       batch_normalize=1
       filters=128
       size=1
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(128,),dtype='float32',buffer=weights_file.read(128*4))
       bn_list = np.ndarray(shape=(3, 128),dtype='float32',buffer=weights_file.read(128*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(128,256,1,1),dtype='float32',buffer=weights_file.read(128*256*1*1*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(128,1,1,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_7')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_7')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_7')(tmp)

       #help to go back.   
       count += (128 + 128*3 + 1*1*128*256)*4
       print "file read to",count


        

       """
       convolutional_7
       batch_normalize=1
       filters=256
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
       bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(256,128,3,3),dtype='float32',buffer=weights_file.read(256*128*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(256,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_8')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_8')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_8')(tmp)

       #help to go back.   
       count += (256 + 256*3 + 3*3*256*128)*4
       print "file read to",count


        

       """
       maxpool_3
       size=2
       stride=2
       """
       tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_4')(tmp)


        

       """
       convolutional_8
       batch_normalize=1
       filters=512
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
       bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(512,256,3,3),dtype='float32',buffer=weights_file.read(512*256*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(512,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_9')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_9')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_9')(tmp)

       #help to go back.   
       count += (512 + 512*3 + 3*3*512*256)*4
       print "file read to",count


        

       """
       convolutional_9
       batch_normalize=1
       filters=256
       size=1
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
       bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(256,512,1,1),dtype='float32',buffer=weights_file.read(256*512*1*1*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(256,1,1,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_10')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_10')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_10')(tmp)

       #help to go back.   
       count += (256 + 256*3 + 1*1*256*512)*4
       print "file read to",count


        

       """
       convolutional_10
       batch_normalize=1
       filters=512
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
       bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(512,256,3,3),dtype='float32',buffer=weights_file.read(512*256*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(512,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_11')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_11')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_11')(tmp)

       #help to go back.   
       count += (512 + 512*3 + 3*3*512*256)*4
       print "file read to",count


        

       """
       convolutional_11
       batch_normalize=1
       filters=256
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(256,),dtype='float32',buffer=weights_file.read(256*4))
       bn_list = np.ndarray(shape=(3, 256),dtype='float32',buffer=weights_file.read(256*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(256,512,1,1),dtype='float32',buffer=weights_file.read(256*512*1*1*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(256,1,1,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_12')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_12')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_12')(tmp)

       #help to go back.   
       count += (256 + 256*3 + 1*1*256*512)*4
       print "file read to",count


        

       """
       convolutional_12
       batch_normalize=1
       filters=512
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
       bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(512,256,3,3),dtype='float32',buffer=weights_file.read(512*256*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(512,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_13')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_13')(tmp)
                            
       #activation
       image_tmp_output = LeakyReLU(alpha=0.1,name='leakyrelu_13')(tmp)

       #help to go back.   
       count += (512 + 512*3 + 3*3*512*256)*4
       print "file read to",count


        

       """
       maxpool_4
       size=2
       stride=2
       """
       tmp = MaxPooling2D(pool_size=(2,2),strides=(2,2),border_mode='same',name='maxpooling2d_5')(image_tmp_output)


        

       """
       convolutional_13
       batch_normalize=1
       filters=1024
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
       bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(1024,512,3,3),dtype='float32',buffer=weights_file.read(1024*512*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(1024,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_14')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_14')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_14')(tmp)
       #help to go back.   
       count += (1024 + 1024*3 + 3*3*1024*512)*4
       print "file read to",count


        

       """
       convolutional_14
       batch_normalize=1
       filters=512
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
       bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(512,1024,1,1),dtype='float32',buffer=weights_file.read(512*1024*1*1*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(512,1,1,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_15')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_15')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_15')(tmp)

       #help to go back.   
       count += (512 + 512*3 + 1*1*512*1024)*4
       print "file read to",count


        

       """
       convolutional_15
       batch_normalize=1
       filters=1024
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
       bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(1024,512,3,3),dtype='float32',buffer=weights_file.read(1024*512*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(1024,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_16')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_16')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_16')(tmp)

       #help to go back.   
       count += (1024 + 1024*3 + 3*3*1024*512)*4
       print "file read to",count


        

       """
       convolutional_16
       batch_normalize=1
       filters=512
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(512,),dtype='float32',buffer=weights_file.read(512*4))
       bn_list = np.ndarray(shape=(3, 512),dtype='float32',buffer=weights_file.read(512*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(512,1024,1,1),dtype='float32',buffer=weights_file.read(512*1024*1*1*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(512,1,1,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_17')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_17')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_17')(tmp)

       #help to go back.   
       count += (512 + 512*3 + 1*1*512*1024)*4
       print "file read to",count


        

       """
       convolutional_17
       batch_normalize=1
       filters=1024
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
       bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(1024,512,3,3),dtype='float32',buffer=weights_file.read(1024*512*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(1024,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_18')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_18')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_18')(tmp)

       #help to go back.   
       count += (1024 + 1024*3 + 3*3*1024*512)*4
       print "file read to",count


        

       """
       convolutional_18
       batch_normalize=1
       filters=1024
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
       bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(1024,1024,3,3),dtype='float32',buffer=weights_file.read(1024*1024*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(1024,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_19')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_19')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_19')(tmp)

       #help to go back.   
       count += (1024 + 1024*3 + 3*3*1024*1024)*4
       print "file read to",count


        

       """
       convolutional_19
       batch_normalize=1
       filters=1024
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
       bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(1024,1024,3,3),dtype='float32',buffer=weights_file.read(1024*1024*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(1024,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_20')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_20')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_20')(tmp)

       #help to go back.   
       count += (1024 + 1024*3 + 3*3*1024*1024)*4
       print "file read to",count


        

       """
       convolutional_20
       batch_normalize=1
       filters=64
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(64,),dtype='float32',buffer=weights_file.read(64*4))
       bn_list = np.ndarray(shape=(3, 64),dtype='float32',buffer=weights_file.read(64*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(64,512,1,1),dtype='float32',buffer=weights_file.read(64*512*1*1*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp2 = Convolution2D(64,1,1,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_21')(image_tmp_output)

       #batchnormalization
       tmp2 = BatchNormalization(weights=bn_weights,name='batch_normalization_21')(tmp2)
                            
       #activation
       tmp2 = LeakyReLU(alpha=0.1,name='leakyrelu_21')(tmp2)

       #help to go back.   
       count += (64 + 64*3 + 1*1*64*512)*4
       print "file read to",count

       def fun(x):
           import tensorflow as tf
           return tf.space_to_depth(x, block_size=2)
       tmp2 = Lambda(fun,output_shape=(13,13,256),name='space_to_depth_2')(tmp2)
       tmp = Merge(name='merge_1',mode='concat')([tmp2,tmp])


       """
       convolutional_21
       batch_normalize=1
       filters=1024
       size=3
       stride=1
       pad=1
       activation=leaky
       """
       #read weights from yolo.weights
       #weights_file.seek(count)
       bias = np.ndarray(shape=(1024,),dtype='float32',buffer=weights_file.read(1024*4))
       bn_list = np.ndarray(shape=(3, 1024),dtype='float32',buffer=weights_file.read(1024*3*4))
       bn_weights = [bn_list[0], bias, bn_list[1], bn_list[2]]
       weights = np.ndarray(shape=(1024,1280,3,3),dtype='float32',buffer=weights_file.read(1024*1280*3*3*4))
       weights = np.transpose(weights,(2,3,1,0))

       #read for convolution
       tmp = Convolution2D(1024,3,3,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     weights=[weights], 
                     bias=False,
                     W_regularizer=l2(0.0005),
                     name='conv2d_22')(tmp)

       #batchnormalization
       tmp = BatchNormalization(weights=bn_weights,name='batch_normalization_22')(tmp)
                            
       #activation
       tmp = LeakyReLU(alpha=0.1,name='leakyrelu_22')(tmp)

       #help to go back.   
       count += (1024 + 1024*3 + 3*3*1024*1280)*4
       print "file read to",count


        
       if include_last:
              """
              convolutional_22
              batch_normalize=1
              filters=425
              size=1
              stride=1
              pad=1
              activation=leaky
              """
              #read weights from yolo.weights
              #weights_file.seek(count)
              bias = np.ndarray(shape=(425,),dtype='float32',buffer=weights_file.read(425*4))
              weights = np.ndarray(shape=(425,1024,1,1),dtype='float32',buffer=weights_file.read(425*1024*1*1*4))
              weights = np.transpose(weights,(2,3,1,0))

              #read for convolution
              tmp = Convolution2D(425,1,1,
                            subsample=(1,1),
                            border_mode='same',
                            activation=None, 
                            weights=[weights,bias], 
                            bias=True,
                            W_regularizer=l2(0.0005),
                            name='conv2d_23')(tmp)

              #help to go back.   
              count += (1*1*1024*425)*4
              print "file read to",count

       model = Model(inputs=image_input, outputs=tmp)
       if save_path!=None:
              model.save(save_path)
              print 'Model has been save'
       weights_file.close()

       return model

def customized_model(filters, weight_path='yolo.weights', image_shape=(416, 416), 
       save_path=None):
       """
       generate yolo2 customized model
       para:
              filters: int, the final output will be (None, 13, 13, filters)
              weight_path: where to find the pretrained yolo weights
              image_shape: the input image shape
              save_path: if None, only return model. Otherwise it will save model to this path 
                     as well. ex. 'model_data/model.h5'
       return:
              model: keras model

       """
       darknet_model = darknet(include_last=False, weight_path=weight_path, image_shape=image_shape)
       image_input = Input(shape=(image_shape[0],image_shape[1],3))
       tmp = darknet_model(image_input)
       tmp = Convolution2D(int(filters),1,1,
                     subsample=(1,1),
                     border_mode='same',
                     activation=None, 
                     bias=True,
                     W_regularizer=l2(0.0005),
                     name='conv2d_23',
                     input_shape=(image_shape[0]//32,image_shape[1]//32, 1024))(tmp)
       model = Model(inputs=image_input, outputs=tmp)
       if save_path!=None:
              model.save(save_path)
              print 'Model has been save'
       return model
 
#from keras.utils.vis_utils import plot_model
#plot_model(model,to_file='model_data/model.png',show_shapes=True)

if __name__=='__main__':
       args = parser.parse_args()
       tmp = args.image_shape.strip().split(',')
       image_shape = (int(tmp[0]), int(tmp[1]))
       if args.net == 'darknet':
              darknet(args.include_last, args.weight_path, image_shape, args.save_path)
       else:
              customized_model(args.filters, args.weight_path, image_shape, args.save_path)
       