# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:21:21 2016

@author: Ben
"""

import math
import random
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy




from sklearn.naive_bayes import GaussianNB
from PIL import Image

xtest = pd.read_csv('Xtest.csv',header=None)
ytest =pd.read_csv('ytest.csv',header=None)
xtrain=pd.read_csv('Xtrain.csv',header=None)
ytrain=pd.read_csv('ytrain.csv',header=None)
Q=pd.read_csv('Q.csv',header=None)
    

    



def main():
    a,b,c,d = [1.0]*4
    posterior_y0=float((sum(ytrain==0))/len(ytrain))
    posterior_y1=float((sum(ytrain==1))/len(ytrain))
    predictor_y0 =np.zeros((len(ytest),1))
    predictor_y1= np.zeros((len(ytest),1))
    
    j=xtest.shape[1]
    for i in range(0,j):
        posterior_y0_x0 = posterior(1,1,xtest[i],xtrain[i],ytrain,0)*posterior_y0
        posterior_y1_x1 = posterior(1,1,xtest[i],xtrain[i],ytrain,1)*posterior_y1
        predictor_y0 = posterior_y0_x0 + predictor_y0
        predictor_y1=posterior_y1_x1 + predictor_y1
        if (i==j-1):
            print "Done"
    predictors = pd.DataFrame(predictor_y0)
    predictors['1']=predictor_y1
    predictors['Predicted']=predictors.idxmax(1)
    predictors['Actual']=ytest
    
    
    
    probs=predictors.ix[:,0]+predictors.ix[:,1]
    predictors['Probability_y0']=1.0 - predictors.ix[:,0]/probs
    predictors['Probability_y1']=1.0-predictors.ix[:,1]/probs
    
    predictors['Wrong']=np.absolute(np.array(predictors['Predicted'],dtype = int) -predictors['Actual'])
    predictors['Ambiguous']= np.absolute(predictors['Probability_y1']-0.5000)
    

    
     #get accuracy
    
    print pd.crosstab(predictors['Actual'],predictors['Predicted'])
    acc = pd.crosstab(predictors['Actual'],predictors['Predicted']).values
    print "Accuracy : " + str( (acc[0][0] + acc[1][1])/float(len(predictors)))
    print str(acc[0][0] + acc[1][1]) +"/" + str(len(predictors))
   
    
    
    
    
    x,y=predictors.index,predictors['Ambiguous']
    topthree=predictors['Ambiguous'].argsort()[0:3]
    
    notcorrect=predictors.index[predictors['Wrong']==1][0:3]
    
    print "These three are wrong"
    
    for j in notcorrect:
        plotset(Q, xtrain, j, predictors)
    
    print "Three most ambiguous"
    for k in topthree:
        plotset(Q,xtrain,k,predictors)
    
    
def plotset(q,x,index,data):
    temp = np.array(x.iloc[index].T).reshape((15,1))
    next=np.dot(q,temp).reshape((28,28))
    plt.figure()
    plt.imshow(next,interpolation='nearest')
    print data.iloc[index]
    plt.savefig('Plot_Row.jpg' %index)    
    
    
    
def posterior(a,b,xtest,xtrain,ytrain,value):
    read = np.std(np.hstack((xtrain,xtest)))
    tempx=xtrain[np.where(ytrain==value)[0]] #first item where y=value we want
    
    
    xtrain_01= tempx/read
    xmean=np.mean(xtrain_01)
    N=float(xtrain_01.shape[0])
    
    mu = N*xmean/float((1/float(a)+N))
    bot = 2.0*(float(2.0*N + a) / float(2.0 * b + N + a * N * 2.0 *b))
    exp = (-(2.0*b+N+1.0)/2.0)*np.log(1.0+((xtest/read - mu)**2.0)/(bot*(2.0*b+N)))
    
    return exp.reshape((len(xtest),1))


main()
    