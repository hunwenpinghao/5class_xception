import numpy as np
import pickle
import sys
import h5py
import pickle
import h5py
import os
import numpy as np
#import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
#%%
def calc_vec_energy_vec(vec):
    isquared = np.power(vec[0],2.0)
    qsquared = np.power(vec[1],2.0)
    inst_energy = np.sqrt(isquared+qsquared)
    return inst_energy

def read(path,lenmax):
    label = path[-14:-4]
    data = np.load(path)
    I = data[0::2]
    Q = data[1::2]
    del data
    plt.figure()
    plt.subplot(211)
    x=np.arange(0,len(I),1)
    plt.plot(x,I)
    plt.subplot(212)
    x=np.arange(0,len(Q),1)
    plt.plot(x,Q)
#    savefig('/media/lab2/F/zbw/new/%s.jpg'%(label))
    P=abs(I)+abs(Q)
    i=0
    S= np.where(P>60)
    aryS = np.array(S) #aryS is the positon of uncompleted signal
    i=0
    W=[]
    for i in range(1,len(aryS[0,:])-2):
        if aryS[0,i]+50<aryS[0,i+1] and aryS[0,i-1]+20>aryS[0,i]:
            W.append(i) # W is the final position of each signal (the position of position)
    mydata = np.array([I, Q]) # , dtype=np.float32
    lengthw = np.shape(W)[0]
    lenths=[]
    lenths.append(aryS[0,W[0]]-aryS[0,0]+1) # lenths is each signal's length
    for i in range(0,lengthw-1):
        lenths.append(aryS[0,W[i+1]]-aryS[0,W[i]+1]+1)
    k=lenmax
    X = np.zeros([lengthw,2,k], dtype=np.float32)
    for i in range(0,lengthw-1):
        X[i,0,0:aryS[0,W[i+1]]-aryS[0,W[i]+1]] = mydata[0,aryS[0,W[i]+1]:aryS[0,W[i+1]]]
        X[i,1,0:aryS[0,W[i+1]]-aryS[0,W[i]+1]] = mydata[1,aryS[0,W[i]+1]:aryS[0,W[i+1]]]
        max0 = np.max(abs(X[i,0,:]))
        max1 = np.max(abs(X[i,1,:]))
        X[i,0,:] /= max0
        X[i,1,:] /= max1
    return X,label,lengthw
#%%
def readhuayin1(path):
    label = path[-39:-32]
    data = np.load(path)
#    data = data[10485760:50000000]
    I = data[0::2]
    Q = data[1::2]
    del data
    plt.figure()
    plt.subplot(211)
    x=np.arange(0,1000000,1)
    plt.plot(x,I[0:1000000])
    plt.subplot(212)
    x=np.arange(0,1000000,1)
    plt.plot(x,Q[0:1000000])
#    savefig('/media/lab2/F/zbw/new/%s.jpg'%(label))
    length=np.shape(I)[0]
    P=abs(I)+abs(Q)
    b=[0,0]
    P=np.hstack((b,P))
#    print(P)

    P=np.hstack((P,b))
    SUMP=[]
#    print(P)
    for i in range(2,length+1):
        temp_SUMP=P[i]+P[i-1]+P[i-2]+P[i+1]+P[i+2]
        SUMP.append(temp_SUMP)
    SUMP = np.array(SUMP)  
    aryS= np.where(SUMP>8000)#15000deshihou91839364 13000deshihou863091064
    aryS=np.array(aryS)
    i=0
    W=[]
    for i in range(1,len(aryS[0,:])-2):
        if aryS[0,i]+150<aryS[0,i+1] and aryS[0,i-1]+100>aryS[0,i]:
            W.append(i) # W is the final position of each signal (the position of position)
    mydata = np.array([I, Q]) # , dtype=np.float32
    lengthw = np.shape(W)[0]#signal numbers
    lenths=[]
    lenths.append(aryS[0,W[0]]-aryS[0,0]+1) # lenths is each signal's length

    for i in range(0,lengthw-1):
        lenths.append(aryS[0,W[i+1]]-aryS[0,W[i]+1]+1)
    k=max(lenths)#max of signals
    X = np.ones([10,2,k], dtype=np.float32)
    for i in range(0,10):     
        X[i,0,0:lenth-1] = mydata[0,aryS[0,W[i]+1]:aryS[0,W[i]+1]+lenth-1]
        X[i,1,0:lenth-1] = mydata[1,aryS[0,W[i]+1]:aryS[0,W[i]+1]+lenth-1]
    l=5401
    d=5401
    k = ((np.shape(X)[2]) - l + 1) // d
    b=k*np.shape(X)[0]
    A = np.zeros([b,2,l], dtype=np.float32)
    for i in range(0,np.shape(X)[0]):
        for j in range(0,k):
            s=99*i+j
            A[s,0,:] = X[i,0,j*d:j*d+l]
            A[s,1,:] = X[i,1,j*d:j*d+l]
            max0 = np.max(abs(A[s,0,:]))
            max1 = np.max(abs(A[s,1,:]))
            A[s,0,:] /= max0
            A[s,1,:] /= max1
    return A,label,b
#%%
def readhuayin2(path,lenth):
    label = path[-35:-28]
    data = np.load(path)
#    data = data[10485760:50000000]
    I = data[0::2]
    Q = data[1::2]
    del data
    plt.figure()
    plt.subplot(211)
    x=np.arange(0,1000000,1)
    plt.plot(x,I[0:1000000])
    plt.subplot(212)
    x=np.arange(0,1000000,1)
    plt.plot(x,Q[0:1000000])
#    savefig('/media/lab2/F/zbw/new/%s.jpg'%(label))
    P=abs(I)+abs(Q)
    S= np.where(P<300)
    aryS = np.array(S) #aryS is the positon of uncompleted signal
    W=[]
    for i in range(2,len(aryS[0,:])-2):
        if (aryS[0,i+1]-aryS[0,i])/(aryS[0,i]-aryS[0,i-1])>500 and aryS[0,i-1]+200>aryS[0,i] and aryS[0,i-2]+300>aryS[0,i] and aryS[0,i]+10000<aryS[0,i+1] :
            W.append(i)
    mydata = np.array([I, Q]) # , dtype=np.float32
    X = np.zeros([10,2,lenth], dtype=np.float32)
    for i in range(0,10):
        X[i,0,0:lenth-1] = mydata[0,aryS[0,W[i]+1]:aryS[0,W[i]+1]+lenth-1]
        X[i,1,0:lenth-1] = mydata[1,aryS[0,W[i]+1]:aryS[0,W[i]+1]+lenth-1]
    l=5401
    d=5401
    k = ((np.shape(X)[2]) - l + 1) // d
    b=k*np.shape(X)[0]
    A = np.zeros([b,2,l], dtype=np.float32)
    for i in range(0,np.shape(X)[0]):
        for j in range(0,k):
            s=99*i+j
            A[s,0,:] = X[i,0,j*d:j*d+l]
            A[s,1,:] = X[i,1,j*d:j*d+l]
            max0 = np.max(abs(A[s,0,:]))
            max1 = np.max(abs(A[s,1,:]))
            A[s,0,:] /= max0
            A[s,1,:] /= max1
    return A,label,b
#%%
if __name__ == '__main__':
    path1 = '/media/lab2/F/zbw/newdata4-10/data/20180228154608_600 MHz_10 MHz_1Y_diantai1-1.npy'
#    #%%
#    label = path1[-14:-4]
#    #%%
#    data2 = np.load(path1)
#    #%%
#    I = data2[0::2]
#    Q = data2[1::2]
#    del data2
#    #%%
#    import matplotlib.pyplot as plt
#    plt.subplot(211)
#    x=np.arange(0,len(I),1)
#    plt.plot(x,I)
#    plt.subplot(212)
#    x=np.arange(0,len(Q),1)
#    plt.plot(x,Q)
#    #%%
#    P=abs(I)+abs(Q)
#    i=0
#    S= np.where(P>60)
#    aryS = np.array(S) #aryS is the positon of uncompleted signal
#    i=0
#    W=[]
#    for i in range(1,len(aryS[0,:])-2):
#        if aryS[0,i]+50<aryS[0,i+1] and aryS[0,i-1]+20>aryS[0,i]:
#            W.append(i) # W is the final position of each signal (the position of position)
#            #%%
#    mydata = np.array([I, Q]) # , dtype=np.float32
#    lengthw = np.shape(W)[0]
#    lenths=[]
#    lenths.append(aryS[0,W[0]]-aryS[0,0]+1) # lenths is each signal's length
#    for i in range(0,lengthw-1):
#        lenths.append(aryS[0,W[i+1]]-aryS[0,W[i]+1]+1)
#        #%%
#    k=5401
#    X = np.zeros([lengthw,2,k], dtype=np.float32)
#    for i in range(0,lengthw-1):
#        X[i,0,0:aryS[0,W[i+1]]-aryS[0,W[i]+1]] = mydata[0,aryS[0,W[i]+1]:aryS[0,W[i+1]]]
#        X[i,1,0:aryS[0,W[i+1]]-aryS[0,W[i]+1]] = mydata[1,aryS[0,W[i]+1]:aryS[0,W[i+1]]]
#        max0 = np.max(abs(X[i,0,:]))
#        max1 = np.max(abs(X[i,1,:]))
#        X[i,0,:] /= max0
#        X[i,1,:] /= max1
#    #%%
    path2 = '/media/lab2/F/zbw/newdata4-10/data/20180228154608_600 MHz_10 MHz_1Y_diantai2-1.npy'
    path3 = '/media/lab2/F/zbw/newdata4-10/data/20180228154608_600 MHz_10 MHz_1Y_diantai3-1.npy'
    path4= '/media/lab2/F/zbw/newdata4-10/data/huayin1_1_60 MHz_60 MHz_1Y_NewTask_75M.npy'
    path5= '/media/lab2/F/zbw/newdata4-10/data/huayin2_1_60 MHz_60 MHz_1Y_NewTask.npy'
    start_index = 10485761
    version=3
    flag = 'max_normalize'
    lenmax=5401
    lenth=540100
    data1, label1, lenth1 = read(path1,lenmax)
    data2, label2, lenth2 = read(path2,lenmax)
    data3, label3, lenth3 = read(path3,lenmax)
    data4, label4, lenth4 = readhuayin1(path4)
    data5, label5, lenth5 = readhuayin2(path5,lenth)
#%%
#    label = path4[-35:-28]
#    data6 = np.fromfile(path4, dtype=np.int16)
#    data6 = data6[10485760:50000000]
#    I = data6[0::2]
#    Q = data6[1::2]
#    del data6
#    length=np.shape(I)[0]
#    P=abs(I)+abs(Q)
#    b=[0,0]
#    P=np.hstack((b,P))
#    #%%
#    P=np.hstack((P,b))
#    SUMP=[]
#    for i in range(2,length+1):
#        temp_SUMP=P[i]+P[i-1]+P[i-2]+P[i+1]+P[i+2]
#        SUMP.append(temp_SUMP)
#    SUMP = np.array(SUMP)  
#    aryS= np.where(SUMP>8000)#15000deshihou91839364 13000deshihou863091064
#    aryS=np.array(aryS)
#    #%%
#    i=0
#    W=[]
#    for i in range(1,len(aryS[0,:])-2):
#        if aryS[0,i]+150<aryS[0,i+1] and aryS[0,i-1]+100>aryS[0,i]:
#            W.append(i) # W is the final position of each signal (the position of position)
#    mydata = np.array([I, Q]) # , dtype=np.float32
#    lengthw = np.shape(W)[0]#signal numbers
#    lenths=[]
#    lenths.append(aryS[0,W[0]]-aryS[0,0]+1) # lenths is each signal's length
#
#    for i in range(0,lengthw-1):
#        lenths.append(aryS[0,W[i+1]]-aryS[0,W[i]+1]+1)
#    k=max(lenths)#max of signals
#      
#    #%%
#    X = np.ones([10,2,k], dtype=np.float32)
#    for i in range(0,10):     
#        X[i,0,0:lenth-1] = mydata[0,aryS[0,W[i]+1]:aryS[0,W[i]+1]+lenth-1]
#        X[i,1,0:lenth-1] = mydata[1,aryS[0,W[i]+1]:aryS[0,W[i]+1]+lenth-1]
#        #%%
#    l=5401
#    d=5401
#    k = ((np.shape(X)[2]) - l + 1) // d
#    b=k*np.shape(X)[0]
#    A = np.zeros([b,2,l], dtype=np.float32)
#        #%%
#    import matplotlib.pyplot as plt
#    plt.subplot(311)
#    x=np.arange(0,14535124-13421824,1)
#    plt.plot(x,I[13421824:14535124])
#    #%%
#    for i in range(0,np.shape(X)[0]):
#        for j in range(0,k):
#            s=99*i+j
#            A[s,0,:] = X[i,0,j*d:j*d+l]
#            A[s,1,:] = X[i,1,j*d:j*d+l]
#            max0 = np.max(abs(A[s,0,:]))
#            max1 = np.max(abs(A[s,1,:]))
#            A[s,0,:] /= max0
#            A[s,1,:] /= max1
#             
#             
##%%
#    for i in range(0,np.shape(X)[0]):
#        for j in range(0,105):
#            s=99*i+j
#            A[s,0,:] = X[i,0,j*d:j*d+l]
#            A[s,1,:] = X[i,1,j*d:j*d+l]
#            max0 = np.max(abs(A[s,0,:]))
#            max1 = np.max(abs(A[s,1,:]))
#            A[s,0,:] /= max0
#            A[s,1,:] /= max1
             #%%
#    data5 = np.fromfile(path5, dtype=np.int16)
#    data5 = data5[10485760:50000000]
#    I = data5[0::2]
#    Q = data5[1::2]
#    del data5
#    P=abs(I)+abs(Q)
#    S= np.where(P<300)
#    aryS = np.array(S) #aryS is the positon of uncompleted signal
#    W=[]
#    for i in range(2,len(aryS[0,:])-2):
#        if (aryS[0,i+1]-aryS[0,i])/(aryS[0,i]-aryS[0,i-1])>500 and aryS[0,i-1]+200>aryS[0,i] and aryS[0,i-2]+300>aryS[0,i] and aryS[0,i]+10000<aryS[0,i+1] :
#            W.append(i)
#    mydata = np.array([I, Q]) # , dtype=np.float32
#    X = np.zeros([10,2,lenth], dtype=np.float32)
#    for i in range(0,10):
#        X[i,0,0:lenth-1] = mydata[0,aryS[0,W[i]+1]:aryS[0,W[i]+1]+lenth-1]
#        X[i,1,0:lenth-1] = mydata[1,aryS[0,W[i]+1]:aryS[0,W[i]+1]+lenth-1]
#    l=5401
#    d=5401
#    #%%
#    k = ((np.shape(X)[2])) // d
#    b=k*np.shape(X)[0]
#    A = np.zeros([b,2,l], dtype=np.float32)
#    #%%
#    for i in range(0,np.shape(X)[0]-1):
#        for j in range(0,k-1):
#            s=99*i+j
#            A[s,0,:] = X[i,0,j*d:j*d+l]
#            A[s,1,:] = X[i,1,j*d:j*d+l]
#            max0 = np.max(abs(A[s,0,:]))
#            max1 = np.max(abs(A[s,1,:]))
#            A[s,0,:] /= max0
#            A[s,1,:] /= max1

#%%
    f = h5py.File('/media/lab2/F/zbw/new/radio_5classes_1Y_little_version%d.h5'%(version),'w') 
    f[label1] = data1
    f[label2] = data2
    f[label3] = data3
    f[label4] = data4
    f[label5] = data5
    f.close()  
    fr=open('/media/lab2/F/zbw/new/radio_5classes_1Y_little_version%d.txt'%(version),'w') 
    fr.write('%s\t%d\n%s\t%d\n%s\t%d\n%s\t%d\n%s\t%d'%(label1,lenth1,label2,lenth2,label3,lenth3,label4,lenth4,label5,lenth5))
    fr.close()
#    path = "cut_radio_start%d_l%d_d%d.h5"%(start_index,l,d)