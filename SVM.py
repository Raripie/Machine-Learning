import numpy as np
import stats as sts
import matplotlib.pyplot as plt
from numpy import *
import pandas as pd
import random as rd
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import time
from sklearn.metrics import hamming_loss
from sklearn.metrics import accuracy_score




dataset=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h6\Anuran Calls (MFCCs)\Frogs_MFCCs.csv","rb"),str,delimiter=",",skiprows=1)
label=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h6\Anuran Calls (MFCCs)\class.csv","r"),str,delimiter=",")
#print(label)

dataset1=np.zeros((7195,26))
for i in range(7195):
    dataset1[i,0:21]=dataset[i,0:21]
    dataset1[i,25]=dataset[i,25]
    for j in range(22):
        if dataset[i,22]==label[j,0]:
            dataset1[i,22]=label[j,1]
        if dataset[i,23]==label[j,0]:
            dataset1[i,23]=label[j,1]
        if dataset[i,24]==label[j,0]:
            dataset1[i,24]=label[j,1] 

#print(dataset1[4914,22:25])
list0 = [n for n in range(0, 7195)]
rd.shuffle(list0)

#print(list0[5000])
train_set=np.zeros((5000,26))
test_set=np.zeros((2195,26))
for i in range(5000):
    train_set[i,:]=dataset1[list0[i],:]
for i in range(2195):
    test_set[i,:]=dataset1[list0[i+5000],:]
#print(test_set)

## Training data and label & Testing data and label
X1_data=np.zeros((5000,22))
X2_data=np.zeros((5000,23))
X3_data=np.zeros((5000,24))
Y1_label=[0]*5000
Y2_label=[0]*5000
Y3_label=[0]*5000

for i in range(5000):
    X1_data[i,:]=train_set[i,0:22]
    X2_data[i,:]=train_set[i,0:23]
    X3_data[i,:]=train_set[i,0:24]
    Y1_label[i]=int(train_set[i,22]*100)
    Y2_label[i]=int(train_set[i,23]*100)
    Y3_label[i]=int(train_set[i,24]*100)
    

    
x1_data=np.zeros((2195,22))
x2_data=np.zeros((2195,23))
x3_data=np.zeros((2195,24))
y1_label=[0]*2195
y2_label=[0]*2195
y3_label=[0]*2195
y_label=np.zeros((2195,3))

for i in range(2195):
    x1_data[i,:]=test_set[i,0:22]
    x2_data[i,0:22]=test_set[i,0:22]
    x3_data[i,0:22]=test_set[i,0:22]
    y1_label[i]=int(test_set[i,22]*100)
    y2_label[i]=int(test_set[i,23]*100)
    y3_label[i]=int(test_set[i,24]*100)
    y_label[i,0]=int(test_set[i,22]*100)
    y_label[i,1]=int(test_set[i,23]*100)
    y_label[i,2]=int(test_set[i,24]*100)
    
## Tranin model 
parameters = { 'C':[0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000], 'gamma':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2]}  
svr = SVC(kernel='rbf')
clf = GridSearchCV(svr, parameters,cv=10)


## 123
print('label order 123:')
#classifier 1
time0=time.time()
print('start time:',time0)
clf.fit(X1_data, Y1_label)
print(clf.best_params_)
y1_pred=clf.predict(x1_data)
loss1=hamming_loss(y1_label, y1_pred)
print('loss1:',loss1)
accuracy1=accuracy_score(y1_label,y1_pred)
print('exact match score 1:',accuracy1)
for i in range(2195):
    x2_data[i,22]=y1_pred[i]/100
    x3_data[i,22]=y1_pred[i]/100

#classifier 2
time1=time.time()
print('classifier 1 time:',time1)
clf.fit(X2_data,Y2_label)
print(clf.best_params_)
y2_pred=clf.predict(x2_data)
loss2=hamming_loss(y2_label, y2_pred)
print('loss2:',loss2)
accuracy2=accuracy_score(y2_label,y2_pred)
print('exact match score 2:',accuracy2)
for i in range(2195):
    x3_data[i,23]=y2_pred[i]/100

#classifier 3
time2=time.time()
print('classifier 2 time:',time2)
clf.fit(X3_data,Y3_label)
print(clf.best_params_)
y3_pred=clf.predict(x3_data)
loss3=hamming_loss(y3_label,y3_pred)
print('loss3:',loss3)
accuracy3=accuracy_score(y3_label,y3_pred)
print('exact match score 3:',accuracy3)

#y_pred=np.zeros((2195,3))
#for i in range(2195):
#    y_pred[i,0]=y1_pred[i]
#    y_pred[i,1]=y2_pred[i]
#    y_pred[i,2]=y3_pred[i]

ave_loss=(loss1+loss2+loss3)/3
print('average hamming loss:',ave_loss)
ave_accuracy=(accuracy1+accuracy2+accuracy3)/3
print('averacy accuracy score:',ave_accuracy)
time3=time.time()
print('classifier 3 time:',time3)

print('123 totally cost:',time3-time0)



## 132
print('label order 132:')
#classifier 1
time0=time.time()
print('start time:',time0)
clf.fit(X1_data, Y1_label)
print(clf.best_params_)
y1_pred=clf.predict(x1_data)
loss1=hamming_loss(y1_label, y1_pred)
print('loss1:',loss1)
accuracy1=accuracy_score(y1_label,y1_pred)
print('exact match score 1:',accuracy1)
for i in range(2195):
    x2_data[i,22]=y1_pred[i]/100
    x3_data[i,22]=y1_pred[i]/100

#classifier 2
time1=time.time()
print('classifier 1 time:',time1)
clf.fit(X2_data,Y3_label)
print(clf.best_params_)
y2_pred=clf.predict(x2_data)
loss2=hamming_loss(y3_label, y2_pred)
print('loss2:',loss2)
accuracy2=accuracy_score(y3_label,y2_pred)
print('exact match score 2:',accuracy2)
for i in range(2195):
    x3_data[i,23]=y2_pred[i]/100

#classifier 3
time2=time.time()
print('classifier 2 time:',time2)
clf.fit(X3_data,Y2_label)
print(clf.best_params_)
y3_pred=clf.predict(x3_data)
loss3=hamming_loss(y2_label,y3_pred)
print('loss3:',loss3)
accuracy3=accuracy_score(y2_label,y3_pred)
print('exact match score 3:',accuracy3)

#y_pred=np.zeros((2195,3))
#for i in range(2195):
#    y_pred[i,0]=y1_pred[i]
#    y_pred[i,1]=y2_pred[i]
#    y_pred[i,2]=y3_pred[i]

ave_loss=(loss1+loss2+loss3)/3
print('average hamming loss:',ave_loss)
ave_accuracy=(accuracy1+accuracy2+accuracy3)/3
print('averacy accuracy score:',ave_accuracy)
time3=time.time()
print('classifier 3 time:',time3)

print('132 totally cost:',time3-time0)


## 213
print('label order 213:')
#classifier 1
time0=time.time()
print('start time:',time0)
clf.fit(X1_data, Y2_label)
print(clf.best_params_)
y1_pred=clf.predict(x1_data)
loss1=hamming_loss(y2_label, y1_pred)
print('loss1:',loss1)
accuracy1=accuracy_score(y2_label,y1_pred)
print('exact match score 1:',accuracy1)
for i in range(2195):
    x2_data[i,22]=y1_pred[i]/100
    x3_data[i,22]=y1_pred[i]/100

#classifier 2
time1=time.time()
print('classifier 1 time:',time1)
clf.fit(X2_data,Y1_label)
print(clf.best_params_)
y2_pred=clf.predict(x2_data)
loss2=hamming_loss(y1_label, y2_pred)
print('loss2:',loss2)
accuracy2=accuracy_score(y1_label,y2_pred)
print('exact match score 2:',accuracy2)
for i in range(2195):
    x3_data[i,23]=y2_pred[i]/100

#classifier 3
time2=time.time()
print('classifier 2 time:',time2)
clf.fit(X3_data,Y3_label)
print(clf.best_params_)
y3_pred=clf.predict(x3_data)
loss3=hamming_loss(y3_label,y3_pred)
print('loss3:',loss3)
accuracy3=accuracy_score(y3_label,y3_pred)
print('exact match score 3:',accuracy3)

#y_pred=np.zeros((2195,3))
#for i in range(2195):
#    y_pred[i,0]=y1_pred[i]
#    y_pred[i,1]=y2_pred[i]
#    y_pred[i,2]=y3_pred[i]

ave_loss=(loss1+loss2+loss3)/3
print('average hamming loss:',ave_loss)
ave_accuracy=(accuracy1+accuracy2+accuracy3)/3
print('averacy accuracy score:',ave_accuracy)
time3=time.time()
print('classifier 3 time:',time3)

print('213 totally cost:',time3-time0)


## 231
print('label order 231:')
#classifier 1
time0=time.time()
print('start time:',time0)
clf.fit(X1_data, Y2_label)
print(clf.best_params_)
y1_pred=clf.predict(x1_data)
loss1=hamming_loss(y2_label, y1_pred)
print('loss1:',loss1)
accuracy1=accuracy_score(y2_label,y1_pred)
print('exact match score 1:',accuracy1)
for i in range(2195):
    x2_data[i,22]=y1_pred[i]/100
    x3_data[i,22]=y1_pred[i]/100

#classifier 2
time1=time.time()
print('classifier 1 time:',time1)
clf.fit(X2_data,Y3_label)
print(clf.best_params_)
y2_pred=clf.predict(x2_data)
loss2=hamming_loss(y3_label, y2_pred)
print('loss2:',loss2)
accuracy2=accuracy_score(y3_label,y2_pred)
print('exact match score 2:',accuracy2)
for i in range(2195):
    x3_data[i,23]=y2_pred[i]/100

#classifier 3
time2=time.time()
print('classifier 2 time:',time2)
clf.fit(X3_data,Y1_label)
print(clf.best_params_)
y3_pred=clf.predict(x3_data)
loss3=hamming_loss(y1_label,y3_pred)
print('loss3:',loss3)
accuracy3=accuracy_score(y1_label,y3_pred)
print('exact match score 3:',accuracy3)

#y_pred=np.zeros((2195,3))
#for i in range(2195):
#    y_pred[i,0]=y1_pred[i]
#    y_pred[i,1]=y2_pred[i]
#    y_pred[i,2]=y3_pred[i]

ave_loss=(loss1+loss2+loss3)/3
print('average hamming loss:',ave_loss)
ave_accuracy=(accuracy1+accuracy2+accuracy3)/3
print('averacy accuracy score:',ave_accuracy)
time3=time.time()
print('classifier 3 time:',time3)

print('231 totally cost:',time3-time0)


## 312
print('label order 312')
#classifier 1
time0=time.time()
print('start time:',time0)
clf.fit(X1_data, Y3_label)
print(clf.best_params_)
y1_pred=clf.predict(x1_data)
loss1=hamming_loss(y3_label, y1_pred)
print('loss1:',loss1)
accuracy1=accuracy_score(y3_label,y1_pred)
print('exact match score 1:',accuracy1)
for i in range(2195):
    x2_data[i,22]=y1_pred[i]/100
    x3_data[i,22]=y1_pred[i]/100

#classifier 2
time1=time.time()
print('classifier 1 time:',time1)
clf.fit(X2_data,Y1_label)
print(clf.best_params_)
y2_pred=clf.predict(x2_data)
loss2=hamming_loss(y1_label, y2_pred)
print('loss2:',loss2)
accuracy2=accuracy_score(y1_label,y2_pred)
print('exact match score 2:',accuracy2)
for i in range(2195):
    x3_data[i,23]=y2_pred[i]/100

#classifier 3
time2=time.time()
print('classifier 2 time:',time2)
clf.fit(X3_data,Y2_label)
print(clf.best_params_)
y3_pred=clf.predict(x3_data)
loss3=hamming_loss(y2_label,y3_pred)
print('loss3:',loss3)
accuracy3=accuracy_score(y2_label,y3_pred)
print('exact match score 3:',accuracy3)

#y_pred=np.zeros((2195,3))
#for i in range(2195):
#    y_pred[i,0]=y1_pred[i]
#    y_pred[i,1]=y2_pred[i]
#    y_pred[i,2]=y3_pred[i]

ave_loss=(loss1+loss2+loss3)/3
print('average hamming loss:',ave_loss)
ave_accuracy=(accuracy1+accuracy2+accuracy3)/3
print('averacy accuracy score:',ave_accuracy)
time3=time.time()
print('classifier 3 time:',time3)

print('312 totally cost:',time3-time0)


## 321
print('label order 321:')
#classifier 1
time0=time.time()
print('start time:',time0)
clf.fit(X1_data, Y3_label)
print(clf.best_params_)
y1_pred=clf.predict(x1_data)
loss1=hamming_loss(y3_label, y1_pred)
print('loss1:',loss1)
accuracy1=accuracy_score(y3_label,y1_pred)
print('exact match score 1:',accuracy1)
for i in range(2195):
    x2_data[i,22]=y1_pred[i]/100
    x3_data[i,22]=y1_pred[i]/100

#classifier 2
time1=time.time()
print('classifier 1 time:',time1)
clf.fit(X2_data,Y2_label)
print(clf.best_params_)
y2_pred=clf.predict(x2_data)
loss2=hamming_loss(y2_label, y2_pred)
print('loss2:',loss2)
accuracy2=accuracy_score(y2_label,y2_pred)
print('exact match score 2:',accuracy2)
for i in range(2195):
    x3_data[i,23]=y2_pred[i]/100

#classifier 3
time2=time.time()
print('classifier 2 time:',time2)
clf.fit(X3_data,Y1_label)
print(clf.best_params_)
y3_pred=clf.predict(x3_data)
loss3=hamming_loss(y1_label,y3_pred)
print('loss3:',loss3)
accuracy3=accuracy_score(y1_label,y3_pred)
print('exact match score 3:',accuracy3)

#y_pred=np.zeros((2195,3))
#for i in range(2195):
#    y_pred[i,0]=y1_pred[i]
#    y_pred[i,1]=y2_pred[i]
#    y_pred[i,2]=y3_pred[i]

ave_loss=(loss1+loss2+loss3)/3
print('average hamming loss:',ave_loss)
ave_accuracy=(accuracy1+accuracy2+accuracy3)/3
print('averacy accuracy score:',ave_accuracy)
time3=time.time()
print('classifier 3 time:',time3)

print('321 totally cost:',time3-time0)
