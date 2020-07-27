"""
Deep Convolutional Autoencoder with TensorFlow

"""
#   ---------------------------------
# import required packages
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
from imgaug import augmenters as ia
#   ---------------------------------
frame_path= '/home/kris_po/sequence/frames_SU/'

batch_size = 30

x = tf.placeholder(tf.float32, [None, 240, 320, 3], name = "data")

os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
# This is
logs_path = "/home/kris_po/sequence/log_conv_autoenc_1/"
#   ---------------------------------
"""
We start by creating the layers with name scopes so that the graph in
the tensorboard looks meaningful
"""
#   ---------------------------------
def conv2d(input, name, kshape, strides=[1, 1, 1, 1]):
    with tf.name_scope(name):
        W = tf.get_variable(name='w_'+name,
                            shape=kshape,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_' + name,
                            shape=[kshape[3]],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        out = tf.nn.conv2d(input,W,strides=strides, padding='SAME')
        out = tf.nn.bias_add(out, b)
        out = tf.nn.relu(out)
        return out
# ---------------------------------
def deconv2d(input, name, kshape, n_outputs, strides=[1, 1]):
    with tf.name_scope(name):
        out = tf.contrib.layers.conv2d_transpose(input,
                                                 num_outputs= n_outputs,
                                                 kernel_size=kshape,
                                                 stride=strides,
                                                 padding='SAME',
                                                 weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(uniform=False),
                                                 biases_initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                                 activation_fn=tf.nn.relu)
        return out
#   ---------------------------------
def maxpool2d(x,name,kshape=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    with tf.name_scope(name):
        out = tf.nn.max_pool(x,
                             ksize=kshape, #size of window
                             strides=strides,
                             padding='SAME')
        return out
#   ---------------------------------
def upsample(input, name, factor=[2,2]):
    size = [int(input.shape[1] * factor[0]), int(input.shape[2] * factor[1])]
    with tf.name_scope(name):
        out = tf.image.resize_bilinear(input, size=size, align_corners=None, name=None)
        return out
#   ---------------------------------
def fullyConnected(input, name, output_size):
    with tf.name_scope(name):
        input_size = input.shape[1:]
        input_size = int(np.prod(input_size))
        W = tf.get_variable(name='w_'+name,
                            shape=[input_size, output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        b = tf.get_variable(name='b_'+name,
                            shape=[output_size],
                            initializer=tf.contrib.layers.xavier_initializer(uniform=False))
        input = tf.reshape(input, [-1, input_size])
        out = tf.nn.relu(tf.add(tf.matmul(input, W), b))
        return out
#   ---------------------------------
def dropout(input, name, keep_rate):
    with tf.name_scope(name):
        out = tf.nn.dropout(input, keep_rate)
        return out
#   ---------------------------------
# Let us now design the autoencoder
def ConvAutoEncoder_1(x, name):
    with tf.name_scope(name):
        """
        We want to get dimensionality reduction of 784 to 196
        Layers:
            input --> 28, 28 (784)
            conv1 --> kernel size: (5,5), n_filters:25 ???make it small so that it runs fast
            pool1 --> 14, 14, 25
            dropout1 --> keeprate 0.8
            reshape --> 14*14*25
            FC1 --> 14*14*25, 14*14*5
            dropout2 --> keeprate 0.8
            FC2 --> 14*14*5, 196 --> output is the encoder vars
            FC3 --> 196, 14*14*5
            dropout3 --> keeprate 0.8
            FC4 --> 14*14*5,14*14*25
            dropout4 --> keeprate 0.8
            reshape --> 14, 14, 25
            deconv1 --> kernel size:(5,5,25), n_filters: 25
            upsample1 --> 28, 28, 25
            FullyConnected (outputlayer) -->  28* 28* 25, 28 * 28
            reshape --> 28*28
        """
        input = tf.reshape(x, shape=[-1, 240, 320, 3])

        # coding part
        c1 = conv2d(input, name='c1', kshape=[3, 3, 1, 12])
        p1 = maxpool2d(c1, name='p1')
        do1 = dropout(p1, name='do1', keep_rate=0.75)
        do1 = tf.reshape(do1, shape=[-1, 120*160*24])
        
        c2 = conv2d(input, name='c2', kshape=[3, 3, 1, 12])
        p2 = maxpool2d(c2, name='p2')
        do2 = dropout(p2, name='do2', keep_rate=0.75)
        do2 = tf.reshape(do2, shape=[-1, 60*80*24])
        
        c3 = conv2d(input, name='c3', kshape=[3, 3, 1, 12])
        p3 = maxpool2d(c3, name='p3')
        do3 = dropout(p3, name='do3', keep_rate=0.75)
        do3 = tf.reshape(do3, shape=[-1, 30*40*12])
        
        #fc1 = fullyConnected(do1, name='fc1', output_size=30*40*5)
        #do2 = dropout(fc1, name='do2', keep_rate=0.75)
        fc2 = fullyConnected(do3, name='fc2', output_size=30*40)
        
        # Decoding part
        #fc3 = fullyConnected(fc2, name='fc3', output_size=30 * 40 * 5)
        #do3 = dropout(fc3, name='do3', keep_rate=0.75)
        fc4 = fullyConnected(fc2, name='fc4', output_size=30 * 40 * 12)
        do4 = dropout(fc4, name='do3', keep_rate=0.75)
        do4 = tf.reshape(do4, shape=[-1, 30, 40, 24])
        
        dc1= deconv2d(do4, name='dc1', kshape=[5,5],n_outputs=12)
        up1 = upsample(dc1, name='up1', factor=[2, 2])
        
        dc2 = deconv2d(up1, name='dc2', kshape=[5,5],n_outputs=12)
        up2 = upsample(dc2, name='up2', factor=[2, 2])
        
        dc3 = deconv2d(up2, name='dc3', kshape=[5,5],n_outputs=12)
        up3 = upsample(dc3, name='up3', factor=[2, 2])
        
        output = fullyConnected(up3, name='output', output_size=240*320*3)
        output = tf.reshape(output, shape=[-1, 240, 320, 3])
        with tf.name_scope('cost'):
            cost = tf.reduce_mean(tf.square(tf.subtract(output, x)))
        return output, cost

def ConvAutoEncoder(x, name):
    with tf.name_scope(name):   
        conv1 = tf.layers.conv2d(x, 64, (2,2), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # 28x28x64
        # tuples for maxpool layer: pool_size and strides
        maxpool1 = tf.layers.max_pooling2d(conv1, (2,2), (2,2), padding='same') # 14x14x64
        # Conv 2 + Maxpool
        print(maxpool1.shape)
        conv2 = tf.layers.conv2d(maxpool1, 32, (2,2), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # 14x14x32
        maxpool2 = tf.layers.max_pooling2d(conv2, (2,2), (2,2), padding='same') # 7x7x32
        # Conv 3 + Maxpool 
        print(maxpool2.shape)
        conv3 = tf.layers.conv2d(maxpool2, 16, (1,1), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # 7x7x16
        maxpool3 = tf.layers.max_pooling2d(conv3, (2,2), (2,2), padding='same') # 4x4x16
        # Conv 4 + Maxpool
        print(maxpool3.shape)
        conv4 = tf.layers.conv2d(maxpool3, 8, (1,1), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # 4x4x8
        maxpool4 = tf.layers.max_pooling2d(conv4, (2,2), (2,2), padding='same') # 2x2x8
        # Conv
        print(maxpool4.shape)
        conv5 = tf.layers.conv2d(maxpool4, 4, (1,1), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # 2x2x4
        maxpool5 = tf.layers.max_pooling2d(conv5, (2,2), (1,1), padding='same') # 2x2x4
        print(maxpool5.shape) # (?, 2, 2, 4)
        ashape = tf.shape(maxpool5)
        flattened = tf.layers.Flatten()(maxpool5)
        encoded = tf.layers.dense(flattened,256,name='encode')
        print(encoded)
        
        Dense2 = tf.layers.dense(flattened,1200)
        reshape = tf.reshape(Dense2,ashape)
        upsample1 = tf.image.resize_nearest_neighbor(reshape, (30,40)) # 4x4x4 
        deconv1 = tf.layers.conv2d(upsample1, 8, (2,2), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # 4x4x8
        upsample2 = tf.image.resize_nearest_neighbor(deconv1, (60,80)) # 7x7x8 
        deconv2 = tf.layers.conv2d(upsample2, 16, (2,2), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # 7x7x16
        upsample3 = tf.image.resize_nearest_neighbor(deconv2, (120,160)) # 14x14x8
        deconv3 = tf.layers.conv2d(upsample3, 32, (2,2), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # 14x14x32
        upsample4 = tf.image.resize_nearest_neighbor(deconv3, (240,320)) # 28x28x8
        deconv4 = tf.layers.conv2d(upsample4, 64, (3,3), padding='same', activation=tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) # 28x28x16
        #print(deconv4.shape)
        output = tf.layers.conv2d(deconv4, 3, (2,2), padding='same', activation=None,kernel_initializer=tf.contrib.layers.xavier_initializer(uniform=False)) #logits 
        print(output)
        decoded = tf.nn.sigmoid(output, name='decoded')
        #loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=x, logits=decoded)
        loss = tf.square(x-decoded)
            
        return output, loss

#   ---------------------------------
        # to get the array of augmented imgs
def create_batchSample(training_images):
  imgs = []
  for f in training_images:
    np_image = cv2.imread(frame_path+f)
    shape = np_image.shape
    if shape[0]==480:
      images = ia.Scale(0.5).augment_image(np_image)
    else:
      images = np_image
    #images = images.astype(np.float32)
    images=images/255.0
    imgs.append(images)   
  return imgs
#--------------------------------------------------
  
def train_network(x):
    list_train_files=[]
    df_labels = pd.read_csv("/home/kris_po/label_final.csv")
    file_list = df_labels[df_labels['filename'].str.contains('Suturing_')]['filename']
    for file in file_list:
        list_train_files.append(file.replace('test','_capture1test'))
        list_train_files.append(file.replace('test','_capture2test'))    
        
    
    '''
    prediction, cost = ConvAutoEncoder(x, 'ConvAutoEnc')
    with tf.name_scope('opt'):
        optimizer = tf.train.AdamOptimizer(0.1).minimize(cost)

    # Create a summary to monitor cost tensor
    tf.summary.scalar("cost", cost)

    # Merge all summaries into a single op
    merged_summary_op = tf.summary.merge_all()

    n_epochs = 20
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        # create log writer object
        writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

        for epoch in range(n_epochs):
            avg_cost = 0
            n_batches = int(len(list_train_files)/ batch_size)
            # Loop over all batches
            for frame in range(0,n_batches,batch_size):
                batch_x = create_batchSample(list_train_files[frame:frame+batch_size])
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c, summary = sess.run([optimizer, cost, merged_summary_op], feed_dict={x: batch_x})
                # Compute average loss
                avg_cost += c / n_batches
             # write log
            writer.add_summary(summary, epoch)

            # Display logs per epoch step
            print('Epoch', epoch+1, ' / ', n_epochs, 'cost:', avg_cost)
          
            if ((epoch+1)%n_epochs)==0:
                save_path = saver.save(sess, str(logs_path)+"model_new.ckpt")
        print('Optimization Finished')
     '''
    output,loss=ConvAutoEncoder(x,'ConvAutoEnc')
    cost = tf.reduce_mean(loss)
    opt = tf.train.AdamOptimizer(1e-2).minimize(cost)
    fetches = {
    'optimizer': opt,
    'loss': cost
    }
    sess = tf.Session()
    saver = tf.train.Saver()
    epochs = 50
    n_batches = int(len(list_train_files)/ batch_size)
    sess.run(tf.global_variables_initializer())
    for e in range(epochs):
        avg_cost=0.0
        for frame in range(0,n_batches,batch_size):
            batch_x = create_batchSample(list_train_files[frame:frame+batch_size])
            res = sess.run(fetches, feed_dict={x: batch_x})
            avg_cost+=res['loss']
                  
            if ((e+1)%epochs)==0:
                save_path = saver.save(sess, str(logs_path)+"model_new_1.ckpt")
                
        print("Epoch: {}/{}...".format(e+1, epochs),
                  "Training loss: {:.4f}".format(avg_cost/n_batches))
        
         
     
train_network(x)




