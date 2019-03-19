# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 22:23:11 2018
15 versions of DenseNet for  
@author: Qi Bi
"""
# coding: utf-8

# In[1]:


# -*- coding: utf-8 -*-
from tfdata import *
import numpy as np
import tensorflow as tf


# In[2]:


def weight_variable(shape, name):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable(name='weights',
                                  shape=shape,
                                  trainable=True,
                                  initializer=tf.truncated_normal_initializer(stddev=0.01))
        #REGULARIZATION_RATE=0.0001
        #regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
        #tf.add_to_collection('losses', regularizer(weights))
        return weights


# In[3]:


def bias_variable(shape, name):
    with tf.variable_scope(name) as scope:
        biases = tf.get_variable(name='biases',
                                 shape=shape,
                                 trainable=True,
                                 initializer=tf.constant_initializer(0.01))

        return biases


# In[4]:


def conv2d(input, in_feature_dim, out_feature_dim, kernel_size, stride, with_bias=True, name=None):
    W = weight_variable([kernel_size, kernel_size, in_feature_dim, out_feature_dim], name=name)
    conv = tf.nn.conv2d(input, W, [1, stride, stride, 1], padding='SAME')
    if with_bias:
        return conv + bias_variable([out_feature_dim], name=name)
    return conv

def lrn(x, radius, alpha, beta, name, bias=1.0):
    return tf.nn.local_response_normalization(x, depth_radius=radius, alpha=alpha,
                                              beta=beta, bias=bias, name=name)


# In[7]:


def avg_pool(input, s, stride):
    return tf.nn.avg_pool(input, [1, s, s, 1], [1, stride, stride, 1], 'SAME')


# In[8]:


def max_pool(input, s, stride):
    return tf.nn.max_pool(input, [1, s, s, 1], [1, stride, stride, 1], 'SAME')


# In[9]:


def loss(logits, targets):
    # Get rid of extra dimensions and cast targets into integers
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Calculate cross entropy from logits and targets
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=targets)
    # Take the average loss across batch size
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
    #+tf.add_n(tf.get_collection('losses'))
    return cross_entropy_mean


# In[10]:


# Train step
def train(loss_value, model_learning_rate):
    # Create optimizer
    # my_optimizer = tf.train.MomentumOptimizer(model_learning_rate, momentum=0.9)

    my_optimizer = tf.train.AdamOptimizer(model_learning_rate)
    # Initialize train step
    train_step = my_optimizer.minimize(loss_value)
    return train_step


# In[11]:


# Accuracy function
def accuracy_of_batch(logits, targets):
    # Make sure targets are integers and drop extra dimensions
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    # Get predicted values by finding which logit is the greatest
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    # Check if they are equal across the batch
    predicted_correctly = tf.equal(batch_predictions, targets)
    # Average the 1's and 0's (True's and False's) across the batch size
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


# In[12]:


def load_with_skip(data_path, session, skip_layer):
    data_dict = np.load(data_path, encoding="bytes").item()
    for key in data_dict:
        if key not in skip_layer:
            with tf.variable_scope(key, reuse=True):
                for subkey, data in zip(('weights', 'biases'), data_dict[key]):
                    get_var = tf.get_variable(subkey).assign(data)
                    session.run(get_var)


# In[13]:


def fc(x, num_in, num_out, name):
    with tf.variable_scope(name) as scope:
        Wfc = weight_variable([num_in, num_out], name=name)
        bfc = bias_variable([num_out], name=name)

        tf.summary.histogram(name + "/weights", Wfc)
        tf.summary.histogram(name + "/biases", bfc)

        act = tf.nn.xw_plus_b(x, Wfc, bfc, name=name + '/op')

        return act


### VGG16
def VGG16(xs,is_training,keep_prob):    
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### block1
    current = conv2d(current, 3, 64, 3, 1, name='block1_convlayer1')
    current = tf.nn.relu(current)
    
    current = conv2d(current, 64, 64, 3, 1, name='block1_convlayer2')
    current = tf.nn.relu(current)
    
    current = max_pool(current, 2, 2)
    
    ### block2
    current = conv2d(current, 64, 128, 3, 1, name='block2_convlayer1')
    current = tf.nn.relu(current)
    
    current = conv2d(current, 128, 128, 3, 1, name='block2_convlayer2')
    current = tf.nn.relu(current)
    
    current = max_pool(current, 2, 2)
    
    ### block3
    current = conv2d(current, 128, 256, 3, 1, name='block3_convlayer1')
    current = tf.nn.relu(current)
    
    current = conv2d(current, 256, 256, 3, 1, name='block3_convlayer2')
    current = tf.nn.relu(current)
    
    current = conv2d(current, 256, 256, 3, 1, name='block3_convlayer3')
    current = tf.nn.relu(current)
    
    current = max_pool(current, 2, 2)
    
    ### block4
    current = conv2d(current, 256, 512, 3, 1, name='block4_convlayer1')
    current = tf.nn.relu(current)
    
    current = conv2d(current, 512, 512, 3, 1, name='block4_convlayer2')
    current = tf.nn.relu(current)
    
    current = conv2d(current, 512, 512, 3, 1, name='block4_convlayer3')
    current = tf.nn.relu(current)
    
    current = max_pool(current, 2, 2)
    
    ### block5
    current = conv2d(current, 512, 512, 3, 1, name='block5_convlayer1')
    current = tf.nn.relu(current)
    
    current = conv2d(current, 512, 512, 3, 1, name='block5_convlayer2')
    current = tf.nn.relu(current)
    
    current = conv2d(current, 512, 512, 3, 1, name='block5_convlayer3')
    current = tf.nn.relu(current)
    
    current = max_pool(current, 2, 2)
    
    ### fc and softmax
    ## 7*7*512=25088
    final_dim = 25088
    current = tf.reshape(current, [-1, final_dim])
    
    current = fc(current, final_dim, 4096, name='fc1')
    current = tf.nn.relu(current)
    current = tf.nn.dropout(current, keep_prob=0.5)
    
    current = fc(current, 4096, 4096, name='fc2')
    current = tf.nn.relu(current)
    current = tf.nn.dropout(current, keep_prob=0.5)
    
    current = fc(current, 4096, 21, name='fc')
    
    return current

### GoogLeNet22
def GoogLeNet(xs,is_training,keep_prob):
    current = tf.reshape(xs, [-1, 224, 224, 3])

    ### conv1  output 112*112*64   
    current = conv2d(current, 3, 64, 7, 2, name='conv1_3x3')
    current = tf.nn.relu(current)
    
    ### maxpool  output 56*56
    current = max_pool(current, 3, 2)
    ### lrn1
    current = lrn(current, 2, 2e-05, 0.75, name='norm1')
       
    ### conv2  output 56*56*192 
    current = conv2d(current, 64, 64, 1, 1, name='conv2_1x1')
    current = conv2d(current, 64, 192, 3, 1, name='conv2_3x3')
    current = tf.nn.relu(current)  
    ### lrn2
    current = lrn(current, 2, 2e-05, 0.75, name='norm2')
    
    #####//////////  3a 3b maxpooling  ////////#####
    ### inception3a  output 28*28*256
    ### maxpool 28*28       
    current = max_pool(current, 3, 2)
    ##1  1*1+ReLU  number  64
    current1=conv2d(current, 192, 64, 1, 1, name='conv3a_b1_1x1')
    current1 = tf.nn.relu(current1) 
    
    ##2  1*1+3*3  number 128
    current2=conv2d(current, 192, 96, 1, 1, name='conv3a_b2_1x1')
    current2 = tf.nn.relu(current2) 
    current2 = conv2d(current2, 96, 128, 3, 1, name='conv3a_b2_3x3')
    current2 = tf.nn.relu(current2)
    
    ##3  1*1+5*5  number 32
    current3=conv2d(current, 192, 16, 1, 1, name='conv3a_b3_1x1')
    current3 = tf.nn.relu(current3) 
    current3 = conv2d(current3, 16, 32, 5, 1, name='conv3a_b3_5x5')
    current3 = tf.nn.relu(current3)
    
    ##4  maxpool+1*1  number 32
    current4 = max_pool(current, 3, 1)
    current4 = conv2d(current4, 192, 32, 1, 1, name='conv3a_b4_1x1')
    current4 = tf.nn.relu(current4)
    
    ## concate
    current2=tf.concat([current1,current2],axis=3)
    current3=tf.concat([current2,current3],axis=3)
    current=tf.concat([current3,current4],axis=3)
    
    ### inception3b  output 28*28*480
    ##1  1*1+ReLU  number  128
    current1=conv2d(current, 256, 128, 1, 1, name='conv3b_b1_1x1')
    current1 = tf.nn.relu(current1) 
    
    ##2  1*1+3*3  number 192
    current2=conv2d(current, 256, 128, 1, 1, name='conv3b_b2_1x1')
    current2 = tf.nn.relu(current2) 
    current2 = conv2d(current2, 128, 192, 3, 1, name='conv3b_b2_3x3')
    current2 = tf.nn.relu(current2)
    
    ##3  1*1+5*5  number 96
    current3=conv2d(current, 256, 32, 1, 1, name='conv3b_b3_1x1')
    current3 = tf.nn.relu(current3) 
    current3 = conv2d(current3, 32, 96, 5, 1, name='conv3b_b3_5x5')
    current3 = tf.nn.relu(current3)
    
    ##4  maxpool+1*1  number 64
    current4 = max_pool(current, 3, 1)
    current4 = conv2d(current4, 256, 64, 1, 1, name='conv3b_b4_1x1')
    current4 = tf.nn.relu(current4)
    
    ## concate
    current2=tf.concat([current1,current2],axis=3)
    current3=tf.concat([current2,current3],axis=3)
    current=tf.concat([current3,current4],axis=3)
       
    ### maxpool outputsize 14*14*480
    current = max_pool(current, 3, 2)
    
    #####//////////  4a 4b maxpooling  ////////#####
    ### inception4a  output 14*14*512
    ##1  1*1+ReLU  number  192
    current1=conv2d(current, 480, 192, 1, 1, name='conv4a_b1_1x1')
    current1 = tf.nn.relu(current1) 
    
    ##2  1*1+3*3  number 208
    current2=conv2d(current, 480, 96, 1, 1, name='conv4a_b2_1x1')
    current2 = tf.nn.relu(current2) 
    current2 = conv2d(current2, 96, 208, 3, 1, name='conv4a_b2_3x3')
    current2 = tf.nn.relu(current2)
    
    ##3  1*1+5*5  number 48
    current3=conv2d(current, 480, 16, 1, 1, name='conv4a_b3_1x1')
    current3 = tf.nn.relu(current3) 
    current3 = conv2d(current3, 16, 48, 5, 1, name='conv4a_b3_5x5')
    current3 = tf.nn.relu(current3)
    
    ##4  maxpool+1*1  number 64
    current4 = max_pool(current, 3, 1)
    current4 = conv2d(current4, 480, 64, 1, 1, name='conv4a_b4_1x1')
    current4 = tf.nn.relu(current4)
    
    ## concate
    current2=tf.concat([current1,current2],axis=3)
    current3=tf.concat([current2,current3],axis=3)
    current=tf.concat([current3,current4],axis=3)
    
    ### inception4b  output 14*14*512
    ##1  1*1+ReLU  number  160
    current1=conv2d(current, 512, 160, 1, 1, name='conv4b_b1_1x1')
    current1 = tf.nn.relu(current1) 
    
    ##2  1*1+3*3  number 224
    current2=conv2d(current, 512, 112, 1, 1, name='conv4b_b2_1x1')
    current2 = tf.nn.relu(current2) 
    current2 = conv2d(current2, 112, 224, 3, 1, name='conv4b_b2_3x3')
    current2 = tf.nn.relu(current2)
    
    ##3  1*1+5*5  number 64
    current3=conv2d(current, 512, 24, 1, 1, name='conv4b_b3_1x1')
    current3 = tf.nn.relu(current3) 
    current3 = conv2d(current3, 24, 64, 5, 1, name='conv4b_b3_5x5')
    current3 = tf.nn.relu(current3)
    
    ##4  maxpool+1*1  number 64
    current4 = max_pool(current, 3, 1)
    current4 = conv2d(current4, 512, 64, 1, 1, name='conv4b_b4_1x1')
    current4 = tf.nn.relu(current4)
    
    ## concate
    current2=tf.concat([current1,current2],axis=3)
    current3=tf.concat([current2,current3],axis=3)
    current=tf.concat([current3,current4],axis=3)
       
    ### aux classifier1
    ### size of ax1current is 4*4*512
    ax1current = avg_pool(current, 5, 3)
    ax1current = conv2d(ax1current, 512, 128, 1, 1, name='ax1_conv1_1x1')
    ax1current = tf.nn.relu(ax1current) 
    #### note in real GoogeLeNet here is 2048  not 3200  because input is 4*4 not 5*5
    ax1current = tf.reshape(ax1current, [-1, 3200])
    ax1current = fc(ax1current, 3200, 1024, name='ax1_fc1')
    ax1current = tf.nn.relu(ax1current) 
    ax1current = tf.nn.dropout(ax1current, 0.3)
    
    ax1current = fc(ax1current, 1024, 1024, name='ax1_fc2')
    ax1current = tf.nn.relu(ax1current) 
    ax1current = tf.nn.dropout(ax1current, 0.3)
    ### then you can feed into a softmax
    
        #####//////////  4c 4d 4e maxpooling  ////////#####
    ### inception4c  output 14*14*512
    ##1  1*1+ReLU  number  128
    current1=conv2d(current, 512, 128, 1, 1, name='conv4c_b1_1x1')
    current1 = tf.nn.relu(current1) 
    
    ##2  1*1+3*3  number 256
    current2=conv2d(current, 512, 128, 1, 1, name='conv4c_b2_1x1')
    current2 = tf.nn.relu(current2) 
    current2 = conv2d(current2, 128, 256, 3, 1, name='conv4c_b2_3x3')
    current2 = tf.nn.relu(current2)
    
    ##3  1*1+5*5  number 64
    current3=conv2d(current, 512, 24, 1, 1, name='conv4c_b3_1x1')
    current3 = tf.nn.relu(current3) 
    current3 = conv2d(current3, 24, 64, 5, 1, name='conv4c_b3_5x5')
    current3 = tf.nn.relu(current3)
    
    ##4  maxpool+1*1  number 64
    current4 = max_pool(current, 3, 1)
    current4 = conv2d(current4, 512, 64, 1, 1, name='conv4c_b4_1x1')
    current4 = tf.nn.relu(current4)
    
    ## concate
    current2=tf.concat([current1,current2],axis=3)
    current3=tf.concat([current2,current3],axis=3)
    current=tf.concat([current3,current4],axis=3)
    
    ### inception4b  output 14*14*528
    ##1  1*1+ReLU  number  112
    current1=conv2d(current, 512, 112, 1, 1, name='conv4d_b1_1x1')
    current1 = tf.nn.relu(current1) 
    
    ##2  1*1+3*3  number 288
    current2=conv2d(current, 512, 144, 1, 1, name='conv4d_b2_1x1')
    current2 = tf.nn.relu(current2) 
    current2 = conv2d(current2, 144, 288, 3, 1, name='conv4d_b2_3x3')
    current2 = tf.nn.relu(current2)
    
    ##3  1*1+5*5  number 64
    current3=conv2d(current, 512, 32, 1, 1, name='conv4d_b3_1x1')
    current3 = tf.nn.relu(current3) 
    current3 = conv2d(current3, 32, 64, 5, 1, name='conv4d_b3_5x5')
    current3 = tf.nn.relu(current3)
    
    ##4  maxpool+1*1  number 64
    current4 = max_pool(current, 3, 1)
    current4 = conv2d(current4, 512, 64, 1, 1, name='conv4d_b4_1x1')
    current4 = tf.nn.relu(current4)
    
    ## concate
    current2=tf.concat([current1,current2],axis=3)
    current3=tf.concat([current2,current3],axis=3)
    current=tf.concat([current3,current4],axis=3)
    
    ### aux classifier2
    ### size of ax1current is 4*4*528
    ax2current = avg_pool(current, 5, 3)
    ax2current = conv2d(ax2current, 528, 128, 1, 1, name='ax2_conv1_1x1')
    ax2current = tf.nn.relu(ax2current) 
    #4x4x128=2048
    ax2current = tf.reshape(ax2current, [-1, 3200])
    ax2current = fc(ax2current, 3200, 1024, name='ax2_fc1')
    ax2current = tf.nn.relu(ax2current) 
    ax2current = tf.nn.dropout(ax2current, 0.3)
    
    ax2current = fc(ax2current, 1024, 1024, name='ax2_fc2')
    ax2current = tf.nn.relu(ax2current) 
    ax2current = tf.nn.dropout(ax2current, 0.3)
    ### then you can feed into a softmax
  
    ### inception4e  output 14*14*832
    ##1  1*1+ReLU  number  256
    current1=conv2d(current, 528, 256, 1, 1, name='conv4e_b1_1x1')
    current1 = tf.nn.relu(current1) 
    
    ##2  1*1+3*3  number 320
    current2=conv2d(current, 528, 160, 1, 1, name='conv4e_b2_1x1')
    current2 = tf.nn.relu(current2) 
    current2 = conv2d(current2, 160, 320, 3, 1, name='conv4e_b2_3x3')
    current2 = tf.nn.relu(current2)
    
    ##3  1*1+5*5  number 128
    current3=conv2d(current, 528, 32, 1, 1, name='conv4e_b3_1x1')
    current3 = tf.nn.relu(current3) 
    current3 = conv2d(current3, 32, 128, 5, 1, name='conv4e_b3_5x5')
    current3 = tf.nn.relu(current3)
    
    ##4  maxpool+1*1  number 128
    current4 = max_pool(current, 3, 1)
    current4 = conv2d(current4, 528, 128, 1, 1, name='conv4e_b4_1x1')
    current4 = tf.nn.relu(current4)
    
    ## concate
    current2=tf.concat([current1,current2],axis=3)
    current3=tf.concat([current2,current3],axis=3)
    current=tf.concat([current3,current4],axis=3)
    
    #####//////////  5a 5b maxpooling  ////////#####
    ### inception5a  output 7*7*832
    current = max_pool(current, 3, 2)
    
    ##1  1*1+ReLU  number  256
    current1=conv2d(current, 832, 256, 1, 1, name='conv5a_b1_1x1')
    current1 = tf.nn.relu(current1) 
    
    ##2  1*1+3*3  number 320
    current2=conv2d(current, 832, 160, 1, 1, name='conv5a_b2_1x1')
    current2 = tf.nn.relu(current2) 
    current2 = conv2d(current2, 160, 320, 3, 1, name='conv5a_b2_3x3')
    current2 = tf.nn.relu(current2)
    
    ##3  1*1+5*5  number 128
    current3=conv2d(current, 832, 32, 1, 1, name='conv5a_b3_1x1')
    current3 = tf.nn.relu(current3) 
    current3 = conv2d(current3, 32, 128, 5, 1, name='conv5a_b3_5x5')
    current3 = tf.nn.relu(current3)
    
    ##4  maxpool+1*1  number 128
    current4 = max_pool(current, 3, 1)
    current4 = conv2d(current4, 832, 128, 1, 1, name='conv5a_b4_1x1')
    current4 = tf.nn.relu(current4)
    
    ## concate
    current2=tf.concat([current1,current2],axis=3)
    current3=tf.concat([current2,current3],axis=3)
    current=tf.concat([current3,current4],axis=3)
    
    ### inception5b  output 7*7*1024
    ##1  1*1+ReLU  number  384
    current1=conv2d(current, 832, 384, 1, 1, name='conv5b_b1_1x1')
    current1 = tf.nn.relu(current1) 
    
    ##2  1*1+3*3  number 384
    current2=conv2d(current, 832, 192, 1, 1, name='conv5b_b2_1x1')
    current2 = tf.nn.relu(current2) 
    current2 = conv2d(current2, 192, 384, 3, 1, name='conv5b_b2_3x3')
    current2 = tf.nn.relu(current2)
    
    ##3  1*1+5*5  number 128
    current3=conv2d(current, 832, 48, 1, 1, name='conv5b_b3_1x1')
    current3 = tf.nn.relu(current3) 
    current3 = conv2d(current3, 48, 128, 5, 1, name='conv5b_b3_5x5')
    current3 = tf.nn.relu(current3)
    
    ##4  maxpool+1*1  number 128
    current4 = max_pool(current, 3, 1)
    current4 = conv2d(current4, 832, 128, 1, 1, name='conv5b_b4_1x1')
    current4 = tf.nn.relu(current4)
    
    ## concate
    current2=tf.concat([current1,current2],axis=3)
    current3=tf.concat([current2,current3],axis=3)
    current=tf.concat([current3,current4],axis=3)
    
    ### fc
    ### size of current is 7*7*1024
    current = avg_pool(current, 7, 7)
    ## 1*1*1024
    current = tf.reshape(current, [-1, 1024])
    current = fc(current, 1024, 1024, name='fc_fc1')
    current = tf.nn.relu(current) 
    current = tf.nn.dropout(current, keep_prob)
    
    current = fc(current, 1024, 1024, name='fc_fc2')
    current = tf.nn.relu(current) 
    current = tf.nn.dropout(current, keep_prob)
    ### then you can feed into a softmax
    
    if is_training==True:
        current=current+ax2current+ax1current
    else:
        current=current
    
    current = fc(current, 1024, 21, name='final_fc2')
    return current



def main():
    # Dataset path
    train_tfrecords = 'train.tfrecords'
    test_tfrecords = 'test.tfrecords'

    # Learning param
    learning_rate = 0.001
    training_iters = 33600  
    batch_size = 20

    # Load batch
    train_img, train_label = input_pipeline(train_tfrecords, batch_size)
    test_img, test_label = input_pipeline(test_tfrecords, batch_size)

    # Model
    with tf.variable_scope('model_definition') as scope:
        train_output = VGG16(train_img, is_training=True, keep_prob=0.8)
        scope.reuse_variables()
        test_output = VGG16(test_img, is_training=False, keep_prob=1)

    # Loss and optimizer
    loss_op = loss(train_output, train_label)
    tf.summary.scalar('loss', loss_op)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = train(loss_op, learning_rate)
        test_loss_op = loss(test_output, test_label)
        # train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)

    # Evaluation
    train_accuracy = accuracy_of_batch(train_output, train_label)
    tf.summary.scalar("train_accuracy", train_accuracy)

    test_accuracy = accuracy_of_batch(test_output, test_label)
    tf.summary.scalar("test_accuracy", test_accuracy)

    # Init
    init = tf.global_variables_initializer()

    # Summary
    merged_summary_op = tf.summary.merge_all()

    # Create Saver
    # saver = tf.train.Saver(tf.trainable_variables())
    ### the default saver is tf.train.Saver() However use this leads to mistakes
    # saver = tf.train.Saver()

    ### new solution
    var_list = tf.trainable_variables()
    g_list = tf.global_variables()
    bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
    bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
    var_list += bn_moving_vars
    saver = tf.train.Saver(var_list=var_list)

    # Launch the graph
    with tf.Session() as sess:
        print('Init variable')
        sess.run(init)
        # with tf.variable_scope('model_definition'):
        #     load_with_skip('bvlc_alexnet.npy', sess, ['fc'])

        # load_ckpt_path = 'checkpoint/my-model.ckpt-21840'
        # saver.restore(sess, load_ckpt_path)

        summary_writer = tf.summary.FileWriter('logs', sess.graph)

        print('Start training')
        # coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess)
        for step in range(training_iters):
            step += 1
            # _, loss_value = sess.run([train_op, loss_op])
            # print('Generation {}: Loss = {:.5f}'.format(step, loss_value))
            # print(Wfc1value[1, 1], Wfc2value[1, 1])
            _, loss_value, test_loss_value = sess.run([train_op, loss_op, test_loss_op])
            print('Generation {}: Loss = {:.5f}     Test Loss={:.5f}'.format(step, loss_value, test_loss_value))

            # Display testing status
            if step % 40 == 0:
                acc1 = sess.run(train_accuracy)
                print(' --- Train Accuracy = {:.2f}%.'.format(100. * acc1))
                acc2 = sess.run(test_accuracy)
                print(' --- Test Accuracy = {:.2f}%.'.format(100. * acc2))

            if step % 40 == 0:
                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str, global_step=step)
            if step % 840 == 0:
                saver.save(sess, 'checkpoint/my-model.ckpt', global_step=step)

        print("Finish Training and validation!")

        # coord.request_stop()
        # coord.join(threads)


# In[19]:


if __name__ == '__main__':
    main()


