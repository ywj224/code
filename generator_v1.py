#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow import keras
from tensorflow.keras import Model

encoder = keras.Sequential(
    [
        layers.Conv2D(64, (3,3), activation='relu', strides=(2,2), padding='same', input_shape=(128,128,3), name='conv0'), # (64,64,64)
        layers.Conv2D(32, (3,3), activation='relu', strides=(2,2), padding='same', name='conv1'), # (32,32,32)
        layers.Conv2D(16, (3,3), activation='relu', strides=(2,2), padding='same', name='conv2'), # (16,16,16)
        layers.Conv2D(8, (3,3), activation='relu', strides=(2,2), padding='same', name='conv3'), # (8,8,8)
        layers.Conv2D(4, (3,3), activation='relu', strides=(2,2), padding='same', name='conv4'), # (4,4,4)
        layers.Reshape((4*4*4,), name='reshape0'),
        layers.Flatten(name='flatten0'),
        layers.Dense(16, use_bias=False, name='dense0')
    ]
)

class generator(tf.keras.Model):
    def __init__(self, model_name):
        super(generator, self).__init__()
        self.encoder = keras.models.load_model(model_name)
        self.conv0 = self.encoder.get_layer(name='conv0') # (64,64,64)
        self.conv1 = self.encoder.get_layer(name='conv1') # (32,32,32)
        self.conv2 = self.encoder.get_layer(name='conv2') # (16,16,16)
        self.conv3 = self.encoder.get_layer(name='conv3') # (8,8,8)
        self.conv4 = self.encoder.get_layer(name='conv4') # (4,4,4)
        self.reshape0 = self.encoder.get_layer(name='reshape0')
        #self.flatten0 = self.encoder.get_layer(name='flatten0')
        self.dense0 = self.encoder.get_layer(name='dense0') # (16)
        
        # reshaping for prepare the features to be fed into the decoder
        self.mid_dense = layers.Dense(4*4*4, use_bias=False)
        self.mid_batchnorm = layers.BatchNormalization()
        self.mid_leaky = layers.LeakyReLU()
        self.mid_reshape = layers.Reshape((4,4,4))
        
        # decoder
        self.convT0 = layers.Conv2DTranspose(8,(5,5),strides=(2,2),padding='same',use_bias=False)
        self.batchnorm0 = layers.BatchNormalization(); self.leaky0 = layers.LeakyReLU()
        
        self.convT1 = layers.Conv2DTranspose(16,(5,5),strides=(2,2),padding='same',use_bias=False)
        self.batchnorm1 = layers.BatchNormalization(); self.leaky1 = layers.LeakyReLU()
        
        self.convT2 = layers.Conv2DTranspose(32,(5,5),strides=(2,2),padding='same',use_bias=False)
        self.batchnorm2 = layers.BatchNormalization(); self.leaky2 = layers.LeakyReLU()
        
        self.convT3 = layers.Conv2DTranspose(64,(5,5),strides=(2,2),padding='same',use_bias=False)
        self.batchnorm3 = layers.BatchNormalization(); self.leaky3 = layers.LeakyReLU()
        
        self.convT4 = layers.Conv2DTranspose(3,(5,5),strides=(2,2),padding='same',use_bias=False)
        
    def call(self, x, AL_diff, random_tensor, name, training=False):
        #encoder
        x0 = tf.stop_gradient(self.conv0(x)) # (64,64,64)
        x1 = tf.stop_gradient(self.conv1(x0)) # (32,32,32)
        x2 = tf.stop_gradient(self.conv2(x1)) # (16,16,16)
        x3 = tf.stop_gradient(self.conv3(x2)) # (8,8,8)
        x4 = tf.stop_gradient(self.conv4(x3)) # (4,4,4)
        x5 = tf.stop_gradient(self.reshape0(x4))
        #x5 = self.flatten0(x4)
        x6 = tf.stop_gradient(self.dense0(x5)) # (16)
        
        #middle
        mid_0, mid_1 = tf.split(x6, 2, axis=1)
        #mid_0, mid_1 = tf.split(x6, 2, axis=1)
        mid = mid_0 + tf.math.exp(mid_1)*random_tensor # size 1 + 4 = 5
        z1 = tf.zeros([batch_size,1], tf.float32) ; z2 = tf.zeros([batch_size,1], tf.float32)
        z3 = tf.zeros([batch_size,1], tf.float32) ; z4 = tf.zeros([batch_size,1], tf.float32)
        z5 = tf.zeros([batch_size,1], tf.float32) ; z6 = tf.zeros([batch_size,1], tf.float32)
        z7 = tf.zeros([batch_size,1], tf.float32) ; z8 = tf.zeros([batch_size,1], tf.float32)
        #print(mid.shape, mid_0.shape, mid_1.shape) # (batchsize, 8)
        if name=='275280':            
            mid = tf.concat([mid, mid, z1, z2, z3, z4, z5, z6, z7, z8 ],axis=1) #; print(1) 
        elif name=='280285':
            mid = tf.concat([mid, z1, mid, z2, z3, z4, z5, z6, z7, z8 ],axis=1) #; print(2)
        elif name=='285290':
            mid = tf.concat([mid, z1, z2, mid, z3, z4, z5, z6, z7, z8 ],axis=1) #; print(3)
        elif name=='290295':
            mid = tf.concat([mid, z1, z2, z3, mid, z4, z5, z6, z7, z8 ],axis=1) #; print(4)
        elif name=='295300':
            mid = tf.concat([mid, z1, z2, z3, z4, mid, z5, z6, z7, z8 ],axis=1) #; print(5)
        elif name=='300305':
            mid = tf.concat([mid, z1, z2, z3, z4, z5, mid, z6, z7, z8 ],axis=1) #; print(6)
        elif name=='305310':
            mid = tf.concat([mid, z1, z2, z3, z4, z5, z6, mid, z7, z8 ],axis=1) #; print(7)
        elif name=='310315':
            mid = tf.concat([mid, z1, z2, z3, z4, z5, z6, z7, mid, z8 ],axis=1) #; print(8)
        elif name=='315320':
            mid = tf.concat([mid, z1, z2, z3, z4, z5, z6, z7, z8, mid ],axis=1) #; print(9)
        elif name=='320325':
            mid = tf.concat([z1, mid, mid, z2, z3, z4, z5, z6, z7, z8 ],axis=1) #; print(10)
        elif name=='325330':
            mid = tf.concat([z1, mid, z2, mid, z3, z4, z5, z6, z7, z8 ],axis=1) #; print(11)
        elif name=='330335':
            mid = tf.concat([z1, mid, z2, z3, mid, z4, z5, z6, z7, z8 ],axis=1) #; print(12)
        elif name=='345350':
            mid = tf.concat([z1, mid, z2, z3, z4, mid, z5, z6, z7, z8 ],axis=1) #; print(13)
        elif name=='205210':
            mid = tf.concat([z1, mid, z2, z3, z4, z5, mid, z6, z7, z8 ],axis=1) #; print(14)
        elif name=='210215':
            mid = tf.concat([z1, mid, z2, z3, z4, z5, z6, mid, z7, z8 ],axis=1) #; print(15)
        else:
            mid = tf.concat([z1, mid, z2, z4, z5, z5, z6, z7, mid, z8 ],axis=1) #; print(16)

            #print(mid.shape) # (batchsize, 24)    
        mid = tf.concat([AL_diff,mid], axis=1) 
        #print(mid.shape) # (batchsize, 25) 
        mid = self.mid_reshape(self.mid_leaky(self.mid_batchnorm(self.mid_dense(mid)))) # (4,4,4)        
        
        #decoder
        x7 = tf.concat([x4,mid], axis=3)        
        x8 = self.leaky0(self.batchnorm0(self.convT0(x7))) # (8,8,8)
        x8 = tf.concat([x3,x8], axis=3)        
        x9 = self.leaky1(self.batchnorm1(self.convT1(x8))) # (16,16,16)
        x9 = tf.concat([x2,x9], axis=3)       
        x10 = self.leaky2(self.batchnorm2(self.convT2(x9))) # (32, 32, 32)
        x10 = tf.concat([x1,x10], axis=3)       
        x11 = self.leaky3(self.batchnorm3(self.convT3(x10))) # (64, 64, 64)
        x11 = tf.concat([x0,x11], axis=3)        
        x12 = self.convT4(x11) # (128,128,3)
        return x12 

