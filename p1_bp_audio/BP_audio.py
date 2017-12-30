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

def main():
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
    print input_train
    
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
    alfa=0.01
    E = np.zeros((1,10))
    for ii in xrange(0,20):
        E[ii]=0;
        for i in xrange(0,1500):
            x=inputn[:,i]
            for j in xrange(0,hidnum):
                I[j]=inputn[:,i].conj().transpose()*w1[j,:].conj().transpose()+b1[j]
                Iout[j]=1/(1+exp(-I[j]))
            
            yn=w2*Iout+b2
            
            e=output_train[:,i]-yn    
            E[ii]=E[ii]+sum(abs(e))
        
            dw2=e*Iout
            db2=e


            

    

    
if __name__ == "__main__":
    main()
    
