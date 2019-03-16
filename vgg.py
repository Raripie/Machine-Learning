#encoding:utf-8
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Activation,MaxPooling2D,Dropout
from keras.optimizers import SGD
from keras.utils import to_categorical


#class plot
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

#(x_train,y_train),(x_test,y_test)=mnist.load_data()
from tensorflow.examples.tutorials.mnist import input_data  
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)  

x_train, y_train = mnist.train.images, mnist.train.labels  
x_test, y_test = mnist.test.images, mnist.test.labels
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


x_train=x_train.reshape(55000,28,28,1)
x_test=x_test.reshape(10000,28,28,1)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

#y_train=to_categorical(y_train,num_classes=10)
#y_test=to_categorical(y_test,num_classes=10)
print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)
# create model
model = Sequential()
model.add(Conv2D(16, kernel_size=(3, 3),
                 padding='same',
                 input_shape=(28,28,1))) 
model.add(Activation('relu'))
model.add(Conv2D(16, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2))) 
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), padding='same',activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(50, activation='relu'))
#model.add(Dropout(0.2))
model.add(Dense(10))
#model.add(Dropout(0.2))
model.add(Activation('softmax'))
#compile model


model.compile(
    loss='categorical_crossentropy',
    optimizer=SGD(lr=0.01, decay=1e-3, momentum=0.9, nesterov=True),
    metrics=['accuracy']
)

#train model
history =LossHistory()
model.fit(x_train,y_train,batch_size=100,verbose=2,epochs=40,
          callbacks=[history],validation_data = (x_test, y_test))

#evaluate model

score=model.evaluate(x_test,y_test,verbose=0)
print("Total loss on Testing Set:", score[0])
print("Accuracy of Testing Set:", score[1])
history.loss_plot('epoch')
