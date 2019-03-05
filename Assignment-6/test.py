import os
import pickle
import random
import sys
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
import pylab as plt
from PIL import Image

def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def calc(pred,label,lab):
    # label = y_train[pos]
    pred=np.where(pred>0.5,1,0)
    tp = 0
    fp = 0
    fn = 0
    img=[]
    for i in range(352):
        temp=[]
        for j in range(1216):
            if(pred[i][j]==1):
                temp.append([255,0,0])
            else:
                temp.append([255,0,255])
            if (label[i][j] == 1 and pred[i][j] ==1):
                tp += 1
            if (label[i][j] == 1 and pred[i][j] ==0):
                fn += 1
            if (label[i][j] == 0 and pred[i][j] ==1):
                fp += 1
        img.append(temp)

    # cv2.imwrite()
    img_i=Image.fromarray(np.array(img).astype(np.uint8))
    # print(lab)
    img_i.save("images/"+lab)
    print("accuracy= " + str(tp / (tp + fp + fn)))
    return (tp / (tp + fp + fn))



test=sorted(os.listdir("HW6_Dataset/image/test"))
test_labels=sorted(os.listdir("HW6_Dataset/label/test"))


X_test=[]
for i in range(1,len(test)):
    X_test.append(cv2.imread("HW6_Dataset/image/test/"+test[i]))
ytest=[]
for i in range(len(test_labels)):
    ytest.append(unpickle("HW6_Dataset/label/test/"+test_labels[i]))

y_test=np.array(ytest)
y_test=np.where(y_test<0,0,y_test)
print(np.array(X_test).shape)
# print(y_test[0].shape)


model_path = './image_classification'

loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(model_path + '.meta')
    loader.restore(sess, model_path)
    loaded_x = loaded_graph.get_tensor_by_name('input_x:0')
    loaded_y = loaded_graph.get_tensor_by_name('output_y:0')
    loaded_acc = loaded_graph.get_tensor_by_name('predi:0')

    accuracy=[]
    for i in range(len(X_test)):
        pred=sess.run(loaded_acc,feed_dict={loaded_x:[X_test[i]],loaded_y:[y_test[i]]})
        accuracy.append(calc(pred, y_test[i],test[i+1]))
    print("Total accuracy: "+str(sum(accuracy)/len(accuracy)))

