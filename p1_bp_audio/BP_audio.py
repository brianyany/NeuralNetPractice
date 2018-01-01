#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 17:18:00 2017

@author: Brian
"""
import numpy as np
from scipy import special, optimize
import matplotlib.pyplot as plt
import scipy.io as spio
from sklearn.preprocessing import MinMaxScaler
import time

def main():
    start_time = time.time()
    C1 = spio.loadmat('data1.mat')
    C2 = spio.loadmat('data2.mat')
    C3 = spio.loadmat('data3.mat')
    C4 = spio.loadmat('data4.mat')
    
    c1=C1['c1']
    c2=C2['c2']
    c3=C3['c3']
    c4=C4['c4']
    
    data=np.concatenate([c1,c2,c3,c4])
    print data
    print len(data)
    
    inputX=data[:,1:25]
    output1=data[:,0]
    print output1[599]
    print len(inputX[1])
    print inputX[0][23]
    
    output = np.zeros((2000,4))
    for i in xrange(0,2000):
        if(output1[i]==1):
            output[i,:]=[1,0,0,0];
        if(output1[i]==2):
            output[i,:]=[0,1,0,0];
        if(output1[i]==3):
            output[i,:]=[0,0,1,0];
        if(output1[i]==4):
            output[i,:]=[0,0,0,1];   
    print output
    
    k=np.random.random((1,2000))
    print (k[0])
    m=np.sort(k[0])
    n=np.argsort(k[0])
    print m
    print n

    input_train=inputX[n[0:1500],:]
    output_train=output[n[0:1500],:]
    input_test=inputX[n[1500:2000],:]
    output_test=output[n[1500:2000],:]
    
    print len(input_train)
    print len(input_test)

    scaler=MinMaxScaler(copy=True, feature_range=(-1, 1))
    inputn=scaler.fit_transform(input_train)
    print len(inputn[0])
    
    innum=24
    hidnum=25
    outnum=4
    
    w1=2*np.random.random((hidnum,innum))-1
    b1=2*np.random.random((hidnum,1))-1
    w2=2*np.random.random((hidnum,outnum))-1
    b2=2*np.random.random((outnum,1))-1
    
    w2_1=w2;w2_2=w2_1;
    w1_1=w1;w1_2=w1_1;
    b1_1=b1;b1_2=b1_1;
    b2_1=b2;b2_2=b2_1;

    xite=0.1
    alpha=0.01
    E = np.zeros((1,10))
    I = np.zeros((1,25))
    Iout = np.zeros((1,25))
    FI = np.zeros((1,25))
    x = np.zeros((24,1))
    dw1 = np.zeros((24,25))
    db1 = np.zeros((1,25))
    for ii in xrange(0,10):
        E[0][ii]=0;
        for i in xrange(0,1500):
            x=inputn[i,:]
            for j in xrange(0,hidnum):
                I[0][j]=np.dot(inputn.transpose()[:,i],w1[j,:])+b1[j]
                Iout[0][j]=1/(1+np.exp(-I[0][j]))
            
            yn=np.dot(w2.transpose(),Iout.transpose())+b2
            
            e=output_train[i,:]-yn.transpose()  
            e=e.reshape(4,1)
            E[0][ii]=E[0][ii]+sum(np.abs(e))[0]
        
            dw2=np.dot(e,Iout)
            db2=e.transpose()
            
            for j in xrange(0,hidnum):
                S=1/(1+np.exp(-I[0][j]))
                FI[0][j]=S*(1-S)
            for k in xrange(0,innum):
                for j in xrange(0,hidnum):
                    dw1[k][j]=FI[0][j]*x[k]*(e[0]*w2[j][0]+e[1]*w2[j][1]+e[2]*w2[j][2]+e[3]*w2[j][3])
                    db1[0][j]=FI[0][j]*(e[0]*w2[j][0]+e[1]*w2[j][1]+e[2]*w2[j][2]+e[3]*w2[j][3])
                    
            w1=w1_1+xite*dw1.transpose()
            b1=b1_1+xite*db1.transpose()
            w2=w2_1+xite*dw2.transpose()
            b2=b2_1+xite*db2.transpose()
            
            w1_2=w1_1;w1_1=w1;
            w2_2=w2_1;w2_1=w2;
            b1_2=b1_1;b1_1=b1;
            b2_2=b2_1;b2_1=b2;
            
          
    fore = np.zeros((4,500))
    output_fore = np.zeros((1,500))
    inputn_test=scaler.fit_transform(input_test)
    for i in xrange(0,500):
        for j in xrange(0,hidnum):
            I[0][j]=np.dot(inputn_test.transpose()[:,i],w1[j,:])+b1[j]
            Iout[0][j]=1/(1+np.exp(-I[0][j]))
        fore[:,i]=(np.dot(w2.transpose(),Iout.transpose())+b2).transpose()
    for i in xrange(0,500):
        output_fore[:,i]=np.nonzero(fore[:,i]==max(fore[:,i]))
    
    error=output_fore-output1[n[1500:2000]]+1
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.figure(1)
    plt.plot(output_fore[0], 'r')
    plt.plot(output1[n[1500:2000]].transpose(),'b')
    plt.show()
    
    plt.figure(2)
    plt.plot(error[0])
    plt.show()
    
    k=np.zeros((1,4))
    for i in xrange(0,500):
        if(error[0][i]!=0):
            b,c=output_test[i,:].max(0),output_test[i,:].argmax(0)
            if(c==0):
                k[0][0]+=1
            if(c==1):
                k[0][1]+=1
            if(c==2):
                k[0][2]+=1
            if(c==3):
                k[0][3]+=1
    kk=np.zeros((1,4))
    for i in xrange(0,500):
        b,c=output_test[i,:].max(0),output_test[i,:].argmax(0)
        if(c==0):
            kk[0][0]+=1
        if(c==1):
            kk[0][1]+=1
        if(c==2):
            kk[0][2]+=1
        if(c==3):
            kk[0][3]+=1
            
    rightratio=(kk[0]-k[0])/kk[0]
    print(kk,k)
    print(rightratio)
    
if __name__ == "__main__":
    main()
    
