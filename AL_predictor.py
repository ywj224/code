#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import tensorflow.keras.layers as layers
from tensorflow.keras import Model


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu,True)
            #tf.config.per_process_gpu_memory_fraction = 0.4 # 메모리 사용률 제한
    except RuntimeError as e:
        print(e)



class ResidualUnit(tf.keras.Model):
    def __init__(self, filter_in, filter_out, kernel_size):
        super(ResidualUnit, self).__init__()
        self.bn1 = layers.BatchNormalization()
        self.conv1 = layers.Conv2D(filter_out, kernel_size, padding='same')
        
        self.bn2 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(filter_out, kernel_size, padding='same')
        
        if filter_in == filter_out:
            self.identity = lambda x:x
        else:
            self.identity = layers.Conv2D(filter_out, (1,1), padding='same')
            
    def call(self, x, training=False, mask=None):
        h = self.bn1(x, training=training)
        h = tf.nn.relu(h)
        h = self.conv1(h)
        
        h = self.bn2(h, training=training)
        h = tf.nn.relu(h)
        h = self.conv2(h)
        return self.identity(x) + h


# In[ ]:


class ResnetLayer(tf.keras.Model):
    def __init__(self, filter_in, filters, kernel_size):
        super(ResnetLayer, self).__init__()
        self.sequence = list()
        
        for f_in, f_out in zip([filter_in] + list(filters), filters):
            self.sequence.append(ResidualUnit(f_in, f_out, kernel_size))
            
    def call(self, x, training=False, mask=None):
        for unit in self.sequence:
            x = unit(x, training=training)
        return x


# In[ ]:


class AL_predictor(tf.keras.Model):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv1 = layers.Conv2D(64, (3,3), padding="same", activation="relu")
        self.res1 = ResnetLayer(64, (64,64), (3,3))
        self.res2 = ResnetLayer(64, (128,128), (3,3))
        #self.res3 = ResnetLayer(128, (256,256), (3,3))
        #self.res4 = ResnetLayer(256, (512,512), (3,3))
        
        self.flatten = layers.Flatten()
        #self.dense1 = layers.Dense(200)
        self.dense2 = layers.Dense(50)
        self.dense3 = layers.Dense(1)
        
    def call(self, x, training=False, mask='none'):
        x = self.conv1(x)
        
        x = self.res1(x, training=training)
        x = self.res2(x, training=training)
        #x = self.res3(x, training=training)
        #x = self.res4(x, training=training)
        
        x = self.flatten(x)
        #x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

