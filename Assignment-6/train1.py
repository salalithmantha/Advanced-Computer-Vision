import tensorflow as tf
import cv2
import pickle
import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split



train=sorted(os.listdir("HW6_Dataset/image/train"))
test=sorted(os.listdir("HW6_Dataset/image/test"))
print(train[0])

train_labels=sorted(os.listdir("HW6_Dataset/label/train"))
test_labels=sorted(os.listdir("HW6_Dataset/label/test"))
img=cv2.imread("HW6_Dataset/image/train/"+train[2])

X_train=[]
for i in range(len(train)):
    X_train.append(cv2.imread("HW6_Dataset/image/train/"+train[i]))

y_train=[]
def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

img_label=unpickle("HW6_Dataset/label/train/"+train_labels[2])

for i in range(len(train_labels)):
    y_train.append(unpickle("HW6_Dataset/label/train/"+train_labels[i]))

y_train=np.array(y_train)
y_train=np.where(y_train<0,0,y_train)


X_train,X_test,y_train,y_test=train_test_split(X_train,y_train,test_size=0.2)



x = tf.placeholder(tf.float32, shape=(None, 352, 1216, 3), name='input_x')
y = tf.placeholder(tf.float32, shape=(None,352,1216), name='output_y')


def calc(pred,label):
    pred=np.where(pred>0.5,1,0)
    tp = 0
    fp = 0
    fn = 0
    for i in range(352):
        for j in range(1216):
            if (label[i][j] == 1 and pred[i][j] ==1):
                tp += 1
            if (label[i][j] == 1 and pred[i][j] ==0):
                fn += 1
            if (label[i][j] == 0 and pred[i][j] ==1):
                fp += 1
    print("accuracy= " + str(tp / (tp + fp + fn)))


def vgg16(features):

    # features=tf.expand_dims(features,0)

    conv1=tf.layers.conv2d(inputs=features,filters=64,
                           kernel_size=[3,3],padding="same",
                           activation=tf.nn.relu)

    conv2=tf.layers.conv2d(inputs=conv1,filters=64,
                           kernel_size=[3,3],padding="same",
                           activation=tf.nn.relu)


    pool1=tf.layers.max_pooling2d(inputs=conv2,
                                  pool_size=[2,2],strides=2,
                                  padding="same")

    conv3=tf.layers.conv2d(inputs=pool1,filters=128,
                           kernel_size=[3,3],padding="same",
                           activation=tf.nn.relu)

    conv4=tf.layers.conv2d(inputs=conv3,filters=128,
                           kernel_size=[3,3],padding="same",
                           activation=tf.nn.relu)

    pool2 = tf.layers.max_pooling2d(inputs=conv4,
                                    pool_size=[2, 2], strides=2,
                                    padding="same")

    conv5 = tf.layers.conv2d(inputs=pool2, filters=256,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)

    conv6 = tf.layers.conv2d(inputs=conv5, filters=256,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)

    conv7 = tf.layers.conv2d(inputs=conv6, filters=256,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)

    pool3 = tf.layers.max_pooling2d(inputs=conv7,
                                    pool_size=[2, 2], strides=2,
                                    padding="same")

    conv8 = tf.layers.conv2d(inputs=pool3, filters=512,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)
    conv9 = tf.layers.conv2d(inputs=conv8, filters=512,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)
    conv10 = tf.layers.conv2d(inputs=conv9, filters=512,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)

    pool4 = tf.layers.max_pooling2d(inputs=conv10,
                                    pool_size=[2, 2], strides=2,
                                    padding="same")

    conv11 = tf.layers.conv2d(inputs=pool4, filters=512,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)
    conv12 = tf.layers.conv2d(inputs=conv11, filters=512,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)
    conv13 = tf.layers.conv2d(inputs=conv12, filters=512,
                             kernel_size=[3, 3], padding="same",
                             activation=tf.nn.relu)

    pool5 = tf.layers.max_pooling2d(inputs=conv13,
                                    pool_size=[2, 2], strides=2,
                                    padding="same")



    conv14=tf.layers.conv2d(inputs=pool5,filters=4096,
                            kernel_size=[7,7],padding="same",activation=tf.nn.relu)

    conv15=tf.layers.conv2d(inputs=conv14,filters=4096,
                            kernel_size=[1,1],padding="same",activation=tf.nn.relu)

    conv16 = tf.layers.conv2d(inputs=conv15, filters=4096,
                              kernel_size=[1, 1],padding="same")


    deconv=tf.layers.conv2d_transpose(inputs=conv16,filters=1,
                                      kernel_size=[64,64],strides=32,padding="same")


    return deconv



logits=vgg16(x)
loss=tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=tf.squeeze(logits,[3]),labels=y))
pred=tf.sigmoid(tf.squeeze(logits,[0,3]))

optimizer=tf.train.MomentumOptimizer(learning_rate=0.001,momentum=0.99).minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initializers.global_variables())
    for epochs in range(0,10):
        print(epochs)
        print(X_train[0].shape)
        # rand_index = np.random.choice(len(X_train), size=len(X_train))
        for pos in range(len(X_train)):
            sess.run(optimizer,feed_dict = {x: [X_train[pos]], y: [y_train[pos]]})
            cost=sess.run(loss, feed_dict={x: [X_train[pos]], y: [y_train[pos]]})
            print("image"+str(pos)+" loss= "+str(cost))


        for pos in range(len(X_test)):
            pred1=sess.run(pred,feed_dict = {x: [X_test[pos]], y: [y_test[pos]]})
            calc(pred1,y_test[pos])

