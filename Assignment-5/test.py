import tensorflow as tf
import pickle
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score
# np.set_printoptions(threshold=np.nan)


model_path = './image_classification'
batch_size = 64
n_samples = 10
top_n_predictions = 5


def unpickle(file):

    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

train=unpickle("cifar-100-python/train")
test=unpickle("cifar-100-python/test")

print(train.keys())
print(test.keys())

Xtrain=train[b'data'].reshape((len(train[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
ytrain=np.array(train[b'fine_labels'])

Xtest=test[b'data'].reshape((len(test[b'data']), 3, 32, 32)).transpose(0, 2, 3, 1)
ytest=np.array(test[b'fine_labels'])
Xtrain=Xtrain.astype(np.float32)
Xtest=Xtest.astype(np.float32)


sub=np.load("mean_vec.npy")
Xtest=Xtest-sub

ytestC=np.array(test[b'coarse_labels'])

lab={}
for i in range(0,20):
    lab[i]=set()
for i in range(0,len(ytest)):
    lab[ytestC[i]].add(ytest[i])
print(lab)




def batch_features_labels(features, labels, batch_size):
    for start in range(0, len(features), batch_size):
        end = min(start + batch_size, len(features))
        yield features[start:end], labels[start:end]

def test_model():
    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:

        loader = tf.train.import_meta_graph(model_path + '.meta')
        loader.restore(sess, model_path)
        loaded_x = loaded_graph.get_tensor_by_name('input_x:0')
        loaded_y = loaded_graph.get_tensor_by_name('output_y:0')
        loaded_logits = loaded_graph.get_tensor_by_name('logits:0')
        loaded_acc = loaded_graph.get_tensor_by_name('accuracy:0')


        label_predictor = sess.run(
                tf.nn.top_k(tf.nn.softmax(loaded_logits),1),
                feed_dict={loaded_x: Xtest, loaded_y: ytest})
        print(label_predictor[1])

        matFine=confusion_matrix(ytest,label_predictor[1])
        accFine=accuracy_score(ytest,label_predictor[1])

        print("Accuracy and confusion matrix for 100 labels")
        print(matFine)
        # print(accFine)

        np.savetxt("matFine.csv",np.array(matFine),delimiter=',',fmt='%d')
        # np.savetxt("accFine",np.array(accFine),delimiter=',')



        coarse_pred=[]
        pred=np.reshape(label_predictor[1],(len(label_predictor[1]))).tolist()
        for i in range(0,len(pred)):
            for pos in lab.keys():
                if(pred[i] in lab[pos]):
                    coarse_pred.append(pos)
        matCoa = confusion_matrix(ytestC,coarse_pred)
        accCoa = accuracy_score(ytestC, coarse_pred)

        print("Accuracy and confusion matrix for 20 labels")
        print(matCoa)
        print(accCoa)

        random_test_features, random_test_labels = tuple(zip(*random.sample(list(zip(Xtest, ytest)), n_samples)))
        random_test_predictions = sess.run(
            tf.nn.top_k(tf.nn.softmax(loaded_logits), top_n_predictions),
            feed_dict={loaded_x: random_test_features, loaded_y: random_test_labels})
        print(random_test_labels)
        print(random_test_predictions[1])




test_model()
