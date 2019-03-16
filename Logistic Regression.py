import numpy as np
import stats as sts
import matplotlib.pyplot as plt
from numpy import *
import pandas as pd

bending1_1=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending1\dataset1.csv","rb"),delimiter=",",skiprows=5)
bending1_2=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending1\dataset2.csv","rb"),delimiter=",",skiprows=5)
bending1_3=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending1\dataset3.csv","rb"),delimiter=",",skiprows=5)
bending1_4=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending1\dataset4.csv","rb"),delimiter=",",skiprows=5)
bending1_5=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending1\dataset5.csv","rb"),delimiter=",",skiprows=5)
bending1_6=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending1\dataset6.csv","rb"),delimiter=",",skiprows=5)
bending1_7=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending1\dataset7.csv","rb"),delimiter=",",skiprows=5)

bending2_1=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending2\dataset1.csv","rb"),delimiter=",",skiprows=5)
bending2_2=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending2\dataset2.csv","rb"),delimiter=",",skiprows=5)
bending2_3=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending2\dataset3.csv","rb"),delimiter=",",skiprows=5)
bending2_4=pd.read_csv(r'C:\Users\samsung\Desktop\EE559\h3\AReM\bending2\dataset4.csv',sep="\s+",skiprows=5)
bending2_5=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending2\dataset5.csv","rb"),delimiter=",",skiprows=5)
bending2_6=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\bending2\dataset6.csv","rb"),delimiter=",",skiprows=5)

cycling_1=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset1.csv","rb"),delimiter=",",skiprows=5)
cycling_2=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset2.csv","rb"),delimiter=",",skiprows=5)
cycling_3=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset3.csv","rb"),delimiter=",",skiprows=5)
cycling_4=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset4.csv","rb"),delimiter=",",skiprows=5)
cycling_5=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset5.csv","rb"),delimiter=",",skiprows=5)
cycling_6=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset6.csv","rb"),delimiter=",",skiprows=5)
cycling_7=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset7.csv","rb"),delimiter=",",skiprows=5)
cycling_8=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset8.csv","rb"),delimiter=",",skiprows=5)
cycling_9=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset9.csv","rb"),delimiter=",",skiprows=5)
cycling_10=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset10.csv","rb"),delimiter=",",skiprows=5)
cycling_11=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset11.csv","rb"),delimiter=",",skiprows=5)
cycling_12=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset12.csv","rb"),delimiter=",",skiprows=5)
cycling_13=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset13.csv","rb"),delimiter=",",skiprows=5)
cycling_14=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset14.csv","rb"),delimiter=",",skiprows=5)
cycling_15=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\cycling\dataset15.csv","rb"),delimiter=",",skiprows=5)

lying_1=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset1.csv","rb"),delimiter=",",skiprows=5)
lying_2=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset2.csv","rb"),delimiter=",",skiprows=5)
lying_3=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset3.csv","rb"),delimiter=",",skiprows=5)
lying_4=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset4.csv","rb"),delimiter=",",skiprows=5)
lying_5=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset5.csv","rb"),delimiter=",",skiprows=5)
lying_6=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset6.csv","rb"),delimiter=",",skiprows=5)
lying_7=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset7.csv","rb"),delimiter=",",skiprows=5)
lying_8=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset8.csv","rb"),delimiter=",",skiprows=5)
lying_9=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset9.csv","rb"),delimiter=",",skiprows=5)
lying_10=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset10.csv","rb"),delimiter=",",skiprows=5)
lying_11=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset11.csv","rb"),delimiter=",",skiprows=5)
lying_12=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset12.csv","rb"),delimiter=",",skiprows=5)
lying_13=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset13.csv","rb"),delimiter=",",skiprows=5)
lying_14=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset14.csv","rb"),delimiter=",",skiprows=5)
lying_15=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\lying\dataset15.csv","rb"),delimiter=",",skiprows=5)

sitting_1=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset1.csv","rb"),delimiter=",",skiprows=5)
sitting_2=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset2.csv","rb"),delimiter=",",skiprows=5)
sitting_3=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset3.csv","rb"),delimiter=",",skiprows=5)
sitting_4=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset4.csv","rb"),delimiter=",",skiprows=5)
sitting_5=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset5.csv","rb"),delimiter=",",skiprows=5)
sitting_6=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset6.csv","rb"),delimiter=",",skiprows=5)
sitting_7=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset7.csv","rb"),delimiter=",",skiprows=5)
sitting_8=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset8.csv","rb"),delimiter=",",skiprows=5)
sitting_9=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset9.csv","rb"),delimiter=",",skiprows=5)
sitting_10=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset10.csv","rb"),delimiter=",",skiprows=5)
sitting_11=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset11.csv","rb"),delimiter=",",skiprows=5)
sitting_12=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset12.csv","rb"),delimiter=",",skiprows=5)
sitting_13=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset13.csv","rb"),delimiter=",",skiprows=5)
sitting_14=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset14.csv","rb"),delimiter=",",skiprows=5)
sitting_15=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\sitting\dataset15.csv","rb"),delimiter=",",skiprows=5)

standing_1=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset1.csv","rb"),delimiter=",",skiprows=5)
standing_2=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset2.csv","rb"),delimiter=",",skiprows=5)
standing_3=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset3.csv","rb"),delimiter=",",skiprows=5)
standing_4=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset4.csv","rb"),delimiter=",",skiprows=5)
standing_5=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset5.csv","rb"),delimiter=",",skiprows=5)
standing_6=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset6.csv","rb"),delimiter=",",skiprows=5)
standing_7=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset7.csv","rb"),delimiter=",",skiprows=5)
standing_8=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset8.csv","rb"),delimiter=",",skiprows=5)
standing_9=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset9.csv","rb"),delimiter=",",skiprows=5)
standing_10=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset10.csv","rb"),delimiter=",",skiprows=5)
standing_11=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset11.csv","rb"),delimiter=",",skiprows=5)
standing_12=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset12.csv","rb"),delimiter=",",skiprows=5)
standing_13=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset13.csv","rb"),delimiter=",",skiprows=5)
standing_14=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset14.csv","rb"),delimiter=",",skiprows=5)
standing_15=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\standing\dataset15.csv","rb"),delimiter=",",skiprows=5)

walking_1=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset1.csv","rb"),delimiter=",",skiprows=5)
walking_2=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset2.csv","rb"),delimiter=",",skiprows=5)
walking_3=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset3.csv","rb"),delimiter=",",skiprows=5)
walking_4=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset4.csv","rb"),delimiter=",",skiprows=5)
walking_5=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset5.csv","rb"),delimiter=",",skiprows=5)
walking_6=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset6.csv","rb"),delimiter=",",skiprows=5)
walking_7=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset7.csv","rb"),delimiter=",",skiprows=5)
walking_8=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset8.csv","rb"),delimiter=",",skiprows=5)
walking_9=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset9.csv","rb"),delimiter=",",skiprows=5)
walking_10=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset10.csv","rb"),delimiter=",",skiprows=5)
walking_11=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset11.csv","rb"),delimiter=",",skiprows=5)
walking_12=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset12.csv","rb"),delimiter=",",skiprows=5)
walking_13=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset13.csv","rb"),delimiter=",",skiprows=5)
walking_14=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset14.csv","rb"),delimiter=",",skiprows=5)
walking_15=np.loadtxt(open(r"C:\Users\samsung\Desktop\EE559\h3\AReM\walking\dataset15.csv","rb"),delimiter=",",skiprows=5)


data=np.zeros((480,6,69))

data[:,:,0]=bending1_3[0:480,1:7]
data[:,:,1]=bending1_4[0:480,1:7]
data[:,:,2]=bending1_5[0:480,1:7]
data[:,:,3]=bending1_6[0:480,1:7]
data[:,:,4]=bending1_7[0:480,1:7]

data[:,:,5]=bending2_3[0:480,1:7]
#data[:,:,6]=bending2_4[0:480,1:7]
data[:,:,7]=bending2_5[0:480,1:7]
data[:,:,8]=bending2_6[0:480,1:7]


data[:,:,9]=cycling_4[0:480,1:7]
data[:,:,10]=cycling_5[0:480,1:7]
data[:,:,11]=cycling_6[0:480,1:7]
data[:,:,12]=cycling_7[0:480,1:7]
data[:,:,13]=cycling_8[0:480,1:7]
data[:,:,14]=cycling_9[0:480,1:7]
data[:,:,15]=cycling_10[0:480,1:7]
data[:,:,16]=cycling_11[0:480,1:7]
data[:,:,17]=cycling_12[0:480,1:7]
data[:,:,18]=cycling_13[0:480,1:7]
data[:,:,19]=cycling_14[0:480,1:7]
data[:,:,20]=cycling_15[0:480,1:7]


data[:,:,21]=lying_4[0:480,1:7]
data[:,:,22]=lying_5[0:480,1:7]
data[:,:,23]=lying_6[0:480,1:7]
data[:,:,24]=lying_7[0:480,1:7]
data[:,:,25]=lying_8[0:480,1:7]
data[:,:,26]=lying_9[0:480,1:7]
data[:,:,27]=lying_10[0:480,1:7]
data[:,:,28]=lying_11[0:480,1:7]
data[:,:,29]=lying_12[0:480,1:7]
data[:,:,30]=lying_13[0:480,1:7]
data[:,:,31]=lying_14[0:480,1:7]
data[:,:,32]=lying_15[0:480,1:7]


data[:,:,33]=sitting_4[0:480,1:7]
data[:,:,34]=sitting_5[0:480,1:7]
data[:,:,35]=sitting_6[0:480,1:7]
data[:,:,36]=sitting_7[0:480,1:7]
data[0:479,:,37]=sitting_8[0:480,1:7]
data[:,:,38]=sitting_9[0:480,1:7]
data[:,:,39]=sitting_10[0:480,1:7]
data[:,:,40]=sitting_11[0:480,1:7]
data[:,:,41]=sitting_12[0:480,1:7]
data[:,:,42]=sitting_13[0:480,1:7]
data[:,:,43]=sitting_14[0:480,1:7]
data[:,:,44]=sitting_15[0:480,1:7]


data[:,:,45]=standing_4[0:480,1:7]
data[:,:,46]=standing_5[0:480,1:7]
data[:,:,47]=standing_6[0:480,1:7]
data[:,:,48]=standing_7[0:480,1:7]
data[:,:,49]=standing_8[0:480,1:7]
data[:,:,50]=standing_9[0:480,1:7]
data[:,:,51]=standing_10[0:480,1:7]
data[:,:,52]=standing_11[0:480,1:7]
data[:,:,53]=standing_12[0:480,1:7]
data[:,:,54]=standing_13[0:480,1:7]
data[:,:,55]=standing_14[0:480,1:7]
data[:,:,56]=standing_15[0:480,1:7]


data[:,:,57]=walking_4[0:480,1:7]
data[:,:,58]=walking_5[0:480,1:7]
data[:,:,59]=walking_6[0:480,1:7]
data[:,:,60]=walking_7[0:480,1:7]
data[:,:,61]=walking_8[0:480,1:7]
data[:,:,62]=walking_9[0:480,1:7]
data[:,:,63]=walking_10[0:480,1:7]
data[:,:,64]=walking_11[0:480,1:7]
data[:,:,65]=walking_12[0:480,1:7]
data[:,:,66]=walking_13[0:480,1:7]
data[:,:,67]=walking_14[0:480,1:7]
data[:,:,68]=walking_15[0:480,1:7]

y=[[0] for row in range(480)]
dataset=np.zeros((69,7,6))
for a in range(6):
    for i in range(69):
        dataset[i,0,a]=data[:,a,i].min()
        dataset[i,1,a]=data[:,a,i].max()
        dataset[i,2,a]=data[:,a,i].mean()
        dataset[i,3,a]=np.median(data[:,a,i])
        dataset[i,4,a]=data[:,a,i].std()
        y=data[:,a,i].tolist()
        dataset[i,5,a]=sts.quantile(y,p=0.25)
        dataset[i,6,a]=sts.quantile(y,p=0.75)

scatter_matrix=np.zeros((69,3,3))
scatter_matrix[:,:,0]=dataset[:,0:3,0]
scatter_matrix[:,:,1]=dataset[:,0:3,1]
scatter_matrix[:,:,2]=dataset[:,0:3,5]


fig=plt.figure()
plt.title('Scatter Plots')
#
ax1=fig.add_subplot(991)
ax1.scatter(scatter_matrix[0:9,0,0].tolist(),scatter_matrix[0:9,0,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,0].tolist(),scatter_matrix[9:69,0,0].tolist(),c='g',marker='.')
plt.ylabel('min1')

ax1=fig.add_subplot(992)
ax1.scatter(scatter_matrix[0:9,1,0].tolist(),scatter_matrix[0:9,0,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,0].tolist(),scatter_matrix[9:69,0,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(993)
ax1.scatter(scatter_matrix[0:9,2,0].tolist(),scatter_matrix[0:9,0,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,0].tolist(),scatter_matrix[9:69,0,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(994)
ax1.scatter(scatter_matrix[0:9,0,1].tolist(),scatter_matrix[0:9,0,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,1].tolist(),scatter_matrix[9:69,0,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(995)
ax1.scatter(scatter_matrix[0:9,1,1].tolist(),scatter_matrix[0:9,0,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,1].tolist(),scatter_matrix[9:69,0,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(996)
ax1.scatter(scatter_matrix[0:9,2,1].tolist(),scatter_matrix[0:9,0,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,1].tolist(),scatter_matrix[9:69,0,0].tolist(),c='g',marker='.')


ax1=fig.add_subplot(997)
ax1.scatter(scatter_matrix[0:9,0,2].tolist(),scatter_matrix[0:9,0,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,2].tolist(),scatter_matrix[9:69,0,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(998)
ax1.scatter(scatter_matrix[0:9,1,2].tolist(),scatter_matrix[0:9,0,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,2].tolist(),scatter_matrix[9:69,0,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(999)
ax1.scatter(scatter_matrix[0:9,2,2].tolist(),scatter_matrix[0:9,0,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,2].tolist(),scatter_matrix[9:69,0,0].tolist(),c='g',marker='.')


#
ax1=fig.add_subplot(9,9,10)
ax1.scatter(scatter_matrix[0:9,0,0].tolist(),scatter_matrix[0:9,1,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,0].tolist(),scatter_matrix[9:69,1,0].tolist(),c='g',marker='.')
plt.ylabel('max1')

ax1=fig.add_subplot(9,9,11)
ax1.scatter(scatter_matrix[0:9,1,0].tolist(),scatter_matrix[0:9,1,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,0].tolist(),scatter_matrix[9:69,1,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,12)
ax1.scatter(scatter_matrix[0:9,2,0].tolist(),scatter_matrix[0:9,1,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,0].tolist(),scatter_matrix[9:69,1,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,13)
ax1.scatter(scatter_matrix[0:9,0,1].tolist(),scatter_matrix[0:9,1,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,1].tolist(),scatter_matrix[9:69,1,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,14)
ax1.scatter(scatter_matrix[0:9,1,1].tolist(),scatter_matrix[0:9,1,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,1].tolist(),scatter_matrix[9:69,1,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,15)
ax1.scatter(scatter_matrix[0:9,2,1].tolist(),scatter_matrix[0:9,1,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,1].tolist(),scatter_matrix[9:69,1,0].tolist(),c='g',marker='.')


ax1=fig.add_subplot(9,9,16)
ax1.scatter(scatter_matrix[0:9,0,2].tolist(),scatter_matrix[0:9,1,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,2].tolist(),scatter_matrix[9:69,1,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,17)
ax1.scatter(scatter_matrix[0:9,1,2].tolist(),scatter_matrix[0:9,1,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,2].tolist(),scatter_matrix[9:69,1,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,18)
ax1.scatter(scatter_matrix[0:9,2,2].tolist(),scatter_matrix[0:9,1,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,2].tolist(),scatter_matrix[9:69,1,0].tolist(),c='g',marker='.')


#
ax1=fig.add_subplot(9,9,19)
ax1.scatter(scatter_matrix[0:9,0,0].tolist(),scatter_matrix[0:9,2,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,0].tolist(),scatter_matrix[9:69,2,0].tolist(),c='g',marker='.')
plt.ylabel('mean1')

ax1=fig.add_subplot(9,9,20)
ax1.scatter(scatter_matrix[0:9,1,0].tolist(),scatter_matrix[0:9,2,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,0].tolist(),scatter_matrix[9:69,2,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,21)
ax1.scatter(scatter_matrix[0:9,2,0].tolist(),scatter_matrix[0:9,2,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,0].tolist(),scatter_matrix[9:69,2,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,22)
ax1.scatter(scatter_matrix[0:9,0,1].tolist(),scatter_matrix[0:9,2,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,1].tolist(),scatter_matrix[9:69,2,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,23)
ax1.scatter(scatter_matrix[0:9,1,1].tolist(),scatter_matrix[0:9,2,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,1].tolist(),scatter_matrix[9:69,2,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,24)
ax1.scatter(scatter_matrix[0:9,2,1].tolist(),scatter_matrix[0:9,2,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,1].tolist(),scatter_matrix[9:69,2,0].tolist(),c='g',marker='.')


ax1=fig.add_subplot(9,9,25)
ax1.scatter(scatter_matrix[0:9,0,2].tolist(),scatter_matrix[0:9,2,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,2].tolist(),scatter_matrix[9:69,2,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,26)
ax1.scatter(scatter_matrix[0:9,1,2].tolist(),scatter_matrix[0:9,2,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,2].tolist(),scatter_matrix[9:69,2,0].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,27)
ax1.scatter(scatter_matrix[0:9,2,2].tolist(),scatter_matrix[0:9,2,0].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,2].tolist(),scatter_matrix[9:69,2,0].tolist(),c='g',marker='.')


#
ax1=fig.add_subplot(9,9,28)
ax1.scatter(scatter_matrix[0:9,0,0].tolist(),scatter_matrix[0:9,0,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,0].tolist(),scatter_matrix[9:69,0,1].tolist(),c='g',marker='.')
plt.ylabel('min2')

ax1=fig.add_subplot(9,9,29)
ax1.scatter(scatter_matrix[0:9,1,0].tolist(),scatter_matrix[0:9,0,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,0].tolist(),scatter_matrix[9:69,0,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,30)
ax1.scatter(scatter_matrix[0:9,2,0].tolist(),scatter_matrix[0:9,0,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,0].tolist(),scatter_matrix[9:69,0,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,31)
ax1.scatter(scatter_matrix[0:9,0,1].tolist(),scatter_matrix[0:9,0,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,1].tolist(),scatter_matrix[9:69,0,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,32)
ax1.scatter(scatter_matrix[0:9,1,1].tolist(),scatter_matrix[0:9,0,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,1].tolist(),scatter_matrix[9:69,0,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,33)
ax1.scatter(scatter_matrix[0:9,2,1].tolist(),scatter_matrix[0:9,0,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,1].tolist(),scatter_matrix[9:69,0,1].tolist(),c='g',marker='.')


ax1=fig.add_subplot(9,9,34)
ax1.scatter(scatter_matrix[0:9,0,2].tolist(),scatter_matrix[0:9,0,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,2].tolist(),scatter_matrix[9:69,0,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,35)
ax1.scatter(scatter_matrix[0:9,1,2].tolist(),scatter_matrix[0:9,0,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,2].tolist(),scatter_matrix[9:69,0,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,36)
ax1.scatter(scatter_matrix[0:9,2,2].tolist(),scatter_matrix[0:9,0,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,2].tolist(),scatter_matrix[9:69,0,1].tolist(),c='g',marker='.')


#
ax1=fig.add_subplot(9,9,37)
ax1.scatter(scatter_matrix[0:9,0,0].tolist(),scatter_matrix[0:9,1,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,0].tolist(),scatter_matrix[9:69,1,1].tolist(),c='g',marker='.')
plt.ylabel('max2')

ax1=fig.add_subplot(9,9,38)
ax1.scatter(scatter_matrix[0:9,1,0].tolist(),scatter_matrix[0:9,1,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,0].tolist(),scatter_matrix[9:69,1,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,39)
ax1.scatter(scatter_matrix[0:9,2,0].tolist(),scatter_matrix[0:9,1,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,0].tolist(),scatter_matrix[9:69,1,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,40)
ax1.scatter(scatter_matrix[0:9,0,1].tolist(),scatter_matrix[0:9,1,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,1].tolist(),scatter_matrix[9:69,1,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,41)
ax1.scatter(scatter_matrix[0:9,1,1].tolist(),scatter_matrix[0:9,1,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,1].tolist(),scatter_matrix[9:69,1,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,42)
ax1.scatter(scatter_matrix[0:9,2,1].tolist(),scatter_matrix[0:9,1,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,1].tolist(),scatter_matrix[9:69,1,1].tolist(),c='g',marker='.')


ax1=fig.add_subplot(9,9,43)
ax1.scatter(scatter_matrix[0:9,0,2].tolist(),scatter_matrix[0:9,1,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,2].tolist(),scatter_matrix[9:69,1,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,44)
ax1.scatter(scatter_matrix[0:9,1,2].tolist(),scatter_matrix[0:9,1,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,2].tolist(),scatter_matrix[9:69,1,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,45)
ax1.scatter(scatter_matrix[0:9,2,2].tolist(),scatter_matrix[0:9,1,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,2].tolist(),scatter_matrix[9:69,1,1].tolist(),c='g',marker='.')


#
ax1=fig.add_subplot(9,9,46)
ax1.scatter(scatter_matrix[0:9,0,0].tolist(),scatter_matrix[0:9,2,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,0].tolist(),scatter_matrix[9:69,2,1].tolist(),c='g',marker='.')
plt.ylabel('mean2')

ax1=fig.add_subplot(9,9,47)
ax1.scatter(scatter_matrix[0:9,1,0].tolist(),scatter_matrix[0:9,2,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,0].tolist(),scatter_matrix[9:69,2,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,48)
ax1.scatter(scatter_matrix[0:9,2,0].tolist(),scatter_matrix[0:9,2,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,0].tolist(),scatter_matrix[9:69,2,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,49)
ax1.scatter(scatter_matrix[0:9,0,1].tolist(),scatter_matrix[0:9,2,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,1].tolist(),scatter_matrix[9:69,2,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,50)
ax1.scatter(scatter_matrix[0:9,1,1].tolist(),scatter_matrix[0:9,2,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,1].tolist(),scatter_matrix[9:69,2,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,51)
ax1.scatter(scatter_matrix[0:9,2,1].tolist(),scatter_matrix[0:9,2,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,1].tolist(),scatter_matrix[9:69,2,1].tolist(),c='g',marker='.')


ax1=fig.add_subplot(9,9,52)
ax1.scatter(scatter_matrix[0:9,0,2].tolist(),scatter_matrix[0:9,2,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,2].tolist(),scatter_matrix[9:69,2,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,53)
ax1.scatter(scatter_matrix[0:9,1,2].tolist(),scatter_matrix[0:9,2,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,2].tolist(),scatter_matrix[9:69,2,1].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,54)
ax1.scatter(scatter_matrix[0:9,2,2].tolist(),scatter_matrix[0:9,2,1].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,2].tolist(),scatter_matrix[9:69,2,1].tolist(),c='g',marker='.')


#
ax1=fig.add_subplot(9,9,55)
ax1.scatter(scatter_matrix[0:9,0,0].tolist(),scatter_matrix[0:9,0,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,0].tolist(),scatter_matrix[9:69,0,2].tolist(),c='g',marker='.')
plt.ylabel('min3')

ax1=fig.add_subplot(9,9,56)
ax1.scatter(scatter_matrix[0:9,1,0].tolist(),scatter_matrix[0:9,0,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,0].tolist(),scatter_matrix[9:69,0,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,57)
ax1.scatter(scatter_matrix[0:9,2,0].tolist(),scatter_matrix[0:9,0,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,0].tolist(),scatter_matrix[9:69,0,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,58)
ax1.scatter(scatter_matrix[0:9,0,1].tolist(),scatter_matrix[0:9,0,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,1].tolist(),scatter_matrix[9:69,0,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,59)
ax1.scatter(scatter_matrix[0:9,1,1].tolist(),scatter_matrix[0:9,0,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,1].tolist(),scatter_matrix[9:69,0,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,60)
ax1.scatter(scatter_matrix[0:9,2,1].tolist(),scatter_matrix[0:9,0,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,1].tolist(),scatter_matrix[9:69,0,2].tolist(),c='g',marker='.')


ax1=fig.add_subplot(9,9,61)
ax1.scatter(scatter_matrix[0:9,0,2].tolist(),scatter_matrix[0:9,0,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,2].tolist(),scatter_matrix[9:69,0,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,62)
ax1.scatter(scatter_matrix[0:9,1,2].tolist(),scatter_matrix[0:9,0,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,2].tolist(),scatter_matrix[9:69,0,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,63)
ax1.scatter(scatter_matrix[0:9,2,2].tolist(),scatter_matrix[0:9,0,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,2].tolist(),scatter_matrix[9:69,0,2].tolist(),c='g',marker='.')


#
ax1=fig.add_subplot(9,9,64)
ax1.scatter(scatter_matrix[0:9,0,0].tolist(),scatter_matrix[0:9,1,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,0].tolist(),scatter_matrix[9:69,1,2].tolist(),c='g',marker='.')
plt.ylabel('max3')

ax1=fig.add_subplot(9,9,65)
ax1.scatter(scatter_matrix[0:9,1,0].tolist(),scatter_matrix[0:9,1,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,0].tolist(),scatter_matrix[9:69,1,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,66)
ax1.scatter(scatter_matrix[0:9,2,0].tolist(),scatter_matrix[0:9,1,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,0].tolist(),scatter_matrix[9:69,1,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,67)
ax1.scatter(scatter_matrix[0:9,0,1].tolist(),scatter_matrix[0:9,1,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,1].tolist(),scatter_matrix[9:69,1,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,68)
ax1.scatter(scatter_matrix[0:9,1,1].tolist(),scatter_matrix[0:9,1,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,1].tolist(),scatter_matrix[9:69,1,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,69)
ax1.scatter(scatter_matrix[0:9,2,1].tolist(),scatter_matrix[0:9,1,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,1].tolist(),scatter_matrix[9:69,1,2].tolist(),c='g',marker='.')


ax1=fig.add_subplot(9,9,70)
ax1.scatter(scatter_matrix[0:9,0,2].tolist(),scatter_matrix[0:9,1,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,2].tolist(),scatter_matrix[9:69,1,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,71)
ax1.scatter(scatter_matrix[0:9,1,2].tolist(),scatter_matrix[0:9,1,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,2].tolist(),scatter_matrix[9:69,1,2].tolist(),c='g',marker='.')

ax1=fig.add_subplot(9,9,72)
ax1.scatter(scatter_matrix[0:9,2,2].tolist(),scatter_matrix[0:9,1,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,2].tolist(),scatter_matrix[9:69,1,2].tolist(),c='g',marker='.')


#
ax1=fig.add_subplot(9,9,73)
ax1.scatter(scatter_matrix[0:9,0,0].tolist(),scatter_matrix[0:9,2,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,0].tolist(),scatter_matrix[9:69,2,2].tolist(),c='g',marker='.')
plt.ylabel('mean3')
plt.xlabel('min1')

ax1=fig.add_subplot(9,9,74)
ax1.scatter(scatter_matrix[0:9,1,0].tolist(),scatter_matrix[0:9,2,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,0].tolist(),scatter_matrix[9:69,2,2].tolist(),c='g',marker='.')
plt.xlabel('max1')

ax1=fig.add_subplot(9,9,75)
ax1.scatter(scatter_matrix[0:9,2,0].tolist(),scatter_matrix[0:9,2,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,0].tolist(),scatter_matrix[9:69,2,2].tolist(),c='g',marker='.')
plt.xlabel('mean1')

ax1=fig.add_subplot(9,9,76)
ax1.scatter(scatter_matrix[0:9,0,1].tolist(),scatter_matrix[0:9,2,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,1].tolist(),scatter_matrix[9:69,2,2].tolist(),c='g',marker='.')
plt.xlabel('min2')

ax1=fig.add_subplot(9,9,77)
ax1.scatter(scatter_matrix[0:9,1,1].tolist(),scatter_matrix[0:9,2,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,1].tolist(),scatter_matrix[9:69,2,2].tolist(),c='g',marker='.')
plt.xlabel('max2')

ax1=fig.add_subplot(9,9,78)
ax1.scatter(scatter_matrix[0:9,2,1].tolist(),scatter_matrix[0:9,2,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,1].tolist(),scatter_matrix[9:69,2,2].tolist(),c='g',marker='.')
plt.xlabel('mean2')


ax1=fig.add_subplot(9,9,79)
ax1.scatter(scatter_matrix[0:9,0,2].tolist(),scatter_matrix[0:9,2,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,0,2].tolist(),scatter_matrix[9:69,2,2].tolist(),c='g',marker='.')
plt.xlabel('min3')

ax1=fig.add_subplot(9,9,80)
ax1.scatter(scatter_matrix[0:9,1,2].tolist(),scatter_matrix[0:9,2,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,1,2].tolist(),scatter_matrix[9:69,2,2].tolist(),c='g',marker='.')
plt.xlabel('max3')

ax1=fig.add_subplot(9,9,81)
ax1.scatter(scatter_matrix[0:9,2,2].tolist(),scatter_matrix[0:9,2,2].tolist(),c='b',marker='.')
ax1.scatter(scatter_matrix[9:69,2,2].tolist(),scatter_matrix[9:69,2,2].tolist(),c='g',marker='.')
plt.xlabel('mean3')


plt.show()





