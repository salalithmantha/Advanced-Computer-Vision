import pickle
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('agg')
import pylab as plt

count=1
epc=1
loss_graph_list=[]


#Reading the data
def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train=unpickle("cifar-100-python/train")
test=unpickle("cifar-100-python/test")

print(train.keys())
print(test.keys())

#Train data
Xtrain=train[b'data'].reshape((len(train[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
ytrain=np.array(train[b'fine_labels'])

#Test data
Xtest=test[b'data'].reshape((len(test[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
ytest=np.array(test[b'fine_labels'])

Xtrain=Xtrain.astype(np.float32)
Xtest=Xtest.astype(np.float32)

Xval=Xtrain[40000:,:,:,:]
Xtrain=Xtrain[0:40000]

yval=ytrain[40000:]
ytrain=ytrain[0:40000]


# Reset graph parameters
tf.reset_default_graph()

x = tf.placeholder(tf.float32, shape=(None, 32, 32, 3), name='input_x')
y = tf.placeholder(tf.int32, shape=(None,), name='output_y')



# model Architecture
def leNet(features):
    # layer 1 input
    # input_layer=tf.reshape(features['x'],[-1,32,32,3])

    # conv_layer #1
    conv1 = tf.layers.conv2d(inputs=features, filters=64,
                             kernel_size=[3,3], strides=1,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    # pooling_layer #1
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2],
                                    strides=2)

    batch1=tf.layers.batch_normalization(inputs=pool1)




    # conv_layer #2
    conv2 = tf.layers.conv2d(inputs=batch1, filters=128,
                             kernel_size=[3,3], strides=1,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    # pooling_layer #2
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2],
                                    strides=2)

    batch2=tf.layers.batch_normalization(inputs=pool2)



    # Dense_layer #1
    pool2_flat = tf.reshape(batch2, [-1, 4608])
    dense1 = tf.layers.dense(inputs=pool2_flat, units=256,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    batch3=tf.layers.batch_normalization(inputs=dense1)


    # Dense_layer #2
    dense2 = tf.layers.dense(inputs=batch3, units=512,
                             activation=tf.nn.relu,
                             kernel_initializer=tf.contrib.layers.xavier_initializer())

    batch4=tf.layers.batch_normalization(inputs=dense2)

    # logits_final_layer
    logits = tf.layers.dense(inputs=batch4, units=100)

    return logits


#epochs, batchSize,Learning rate
epochs = 10
batch_size = 128
learning_rate = 0.001

# Output of model
logits = leNet(x)
model = tf.identity(logits, name='logits')

#Loss function & Optimization Algorithm
cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)


#Prediction and Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1),tf.cast(y,tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')


# Shuffling Data
def batch_features_labels(features, labels, batch_size):

    rand_index=np.random.choice(len(features),size=len(features))
    for start in range(0,len(features),batch_size):
        end=min(start+batch_size,len(features))
        tmp=np.array(rand_index[start:end])
        yield features[tmp],labels[tmp]


#saving model
model_path = './image_classification'


#Creating Session
print('Training...')
with tf.Session() as sess:
    #inistalizing Global_variables
    sess.run(tf.global_variables_initializer())
    ###########

    # Data Augmentation
    data_tf=tf.convert_to_tensor(Xtrain,np.float32)
    big=tf.image.resize_images(Xtrain,(36,36))
    top_r=tf.image.crop_to_bounding_box(big,0,0,32,32)
    top_l=tf.image.crop_to_bounding_box(big,0,4,32,32)
    bot_r=tf.image.crop_to_bounding_box(big,4,0,32,32)
    bot_l=tf.image.crop_to_bounding_box(big, 4, 4, 32, 32)
    cen=tf.image.crop_to_bounding_box(big,2,2,32,32)
    flip=tf.image.flip_left_right(data_tf)

    fbig = tf.image.resize_images(flip, (36, 36))
    ftop_r = tf.image.crop_to_bounding_box(fbig, 0, 0, 32, 32)
    ftop_l = tf.image.crop_to_bounding_box(fbig, 0, 4, 32, 32)
    fbot_r = tf.image.crop_to_bounding_box(fbig, 4, 0, 32, 32)
    fbot_l = tf.image.crop_to_bounding_box(fbig, 4, 4, 32, 32)
    fcen = tf.image.crop_to_bounding_box(fbig, 2, 2, 32, 32)


    Xtrain1 = tf.concat([data_tf, top_r, top_l, bot_l, bot_r, cen, flip], axis=0)
    Xtrain2=tf.concat([ftop_l,ftop_r,fbot_l,fbot_r,fcen],axis=0)

    # Xtrain=tf.concat([Xtrain1,Xtrain2],axis=0)

    sess.run(Xtrain1)
    sess.run(Xtrain2)
    Xtrain1=Xtrain1.eval()
    Xtrain2=Xtrain2.eval()
    Xtrain=np.concatenate((Xtrain1,Xtrain2))
    ytrain=np.concatenate((ytrain,ytrain,ytrain,ytrain,ytrain,ytrain,ytrain,ytrain,ytrain,ytrain,ytrain,ytrain))
    # Xtrain=Xtrain.eval()
    print(type(Xtrain))
    print(Xtrain.shape)
    print(ytrain.shape)

    #Calculating mean of the dataSet
    sub = np.mean(Xtrain, axis=0)
    Xtrain = Xtrain - sub
    Xval=Xval-sub
    np.save("mean_vec",sub)

    # Traning and Validation
    for epoch in range(0,epochs):
        for batch_features, batch_labels in batch_features_labels(Xtrain, ytrain, batch_size):
            sess.run(optimizer,
                            feed_dict={
                                x: batch_features,
                                y: batch_labels
                            })

            loss = sess.run(cost,
                            feed_dict={
                                x: batch_features,
                                y: batch_labels
                            })
            loss_graph_list.append(loss)

            if(epoch==epc):
                epc+=1
                print('Epoch {:>2}\n'.format(epoch), end='')
                print("Loss= ",str(sum(loss_graph_list)/len(loss_graph_list)))

            if(epoch==count):
                count+=1
                valid_acc = sess.run(accuracy,
                                     feed_dict={
                                         x: Xval,
                                         y: yval})

                print('Loss: {:>10.4f} Validation Accuracy: {:.6f}'.format(sum(loss_graph_list)/len(loss_graph_list), valid_acc))


    # Saving model and plotting Loss Graph
    plt.plot(loss_graph_list)
    plt.savefig("myfig")
    model_saver = tf.train.Saver()
    save_path_model = model_saver.save(sess, model_path)