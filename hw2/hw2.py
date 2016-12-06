# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 10:52:17 2016

@author: Ben
"""
#Bayesian mOdel machine learning hw 2 bjz2107


import numpy as np
import pandas as pd
import scipy
from scipy.stats import norm

import matplotlib.pyplot as plt

xtest = pd.read_csv('Xtest.csv',header=None)
ytest =pd.read_csv('ytest.csv',header=None)
xtrain=pd.read_csv('Xtrain.csv',header=None)
ytrain=pd.read_csv('ytrain.csv',header=None)
Q=pd.read_csv('Q.csv',header=None)


def ProbitEM(xtrain, ytrain, T, sigma, lambduh):
    #Implementing Probit,
    #first need to set up container variables, with dimensions equal to those provided
    #in the problem. 
    k,d = xtrain.shape
    W_t = np.zeros((d,1))
    
    y_1 = np.where(ytrain==1)
    y_0 = np.where(ytrain==0)
    p = np.zeros((k,1))
    lnpW_xy=[]
    W_tmp = []
    xtranx = np.zeros((d,d))
    for i in xrange(k):
        xtranx = xtranx + np.dot(xtrain.iloc[i].reshape(15,1),xtrain.iloc[i].reshape(1,15))
    #Taken from https://people.duke.edu/~ccc14/sta-663/EMAlgorithm.html
    xtranx = np.dot(xtrain.T,xtrain)
    #Xt Transposed Xt
    for t in xrange(T):
        xw = np.dot(xtrain,W_t).reshape((k,1))
        xw_sigma = np.divide(xw, sigma)
        # Estep
        #This is just pretty standard
        p[y_1] = xw[y_1] + (sigma*norm.pdf(-xw_sigma[y_1]) / (1.0-norm.cdf(-xw_sigma[y_1])))
        p[y_0] = xw[y_0] + (sigma*-norm.pdf(-xw_sigma[y_0]) / (norm.cdf(-xw_sigma[y_0])))
        # Mstep        
        #Take the M step expression, which in reality is the 
        #lambda .* I + x ./ sigma^2 * xi./sigma^2
        #This is written as a painful way ... idk if it's the best way found online
        W_t = np.dot(np.linalg.inv(lambduh* np.diag(np.ones(d)) + np.divide(xtranx,sigma**2.0)),
                     np.sum(np.divide(np.multiply(xtrain,p),sigma**2.0),axis=0))
        W_tmp.append(W_t)
        # Calculating lnp(y,w_t|X)
        
        #d/2 * log(lambdaover2pi - lambga* log(normcdf) (1-normcdf))
        lnpW_xy.append(float((d/2.0) * np.log(lambduh / (2.0*np.pi)) 
                - np.multiply((lambduh/2.0),np.dot(W_t.T,W_t))
                + np.dot(ytrain.T,np.log( norm.cdf(xw_sigma)))
                + np.dot((1.0-ytrain).T,np.log(1.0 - norm.cdf(xw_sigma)))))
           
    return W_t, lnpW_xy, W_tmp
def flattennegatives(prob):
    return [0 if i <= 0 else 1 for i in prob]


def main():

    #From HW1 
    W_t, lnpW_xy, W_tmp = ProbitEM(xtrain,ytrain,100,1.5,1.0)
    h = xtest.shape[0]
    yprobs = np.dot(xtest,W_t).reshape((h,1))
    ypredictors = np.array(flattennegatives(yprobs)).reshape((h),1)
    predictors = pd.DataFrame( norm.cdf(yprobs))   
    predictors.columns = ['PredictedProbability']
    predictors['Predicted']=ypredictors
    predictors['Actual']=ytest
    predictors['Wrong']=np.absolute(np.array(predictors['Predicted'],dtype = int) -predictors['Actual'])
    predictors['Ambiguous']= np.absolute(predictors['PredictedProbability']-0.5000)
     #get accuracy
    print pd.crosstab(predictors['Actual'],predictors['Predicted'])
    accu = pd.crosstab(predictors['Actual'],predictors['Predicted']).values
    print "Accuracy : " + str((accu[0][0] + accu[1][1])/float(len(predictors)))
    print str(accu[0][0] + accu[1][1]) +"/"+ str(len(predictors))
    x,y=predictors.index,predictors['Ambiguous']
    topthree=predictors['Ambiguous'].argsort()[0:3].values
    notcorrect=predictors.index[predictors['Wrong']==1][0:3]
    
    #Confusion Matrix Part 3
    Confusion = pd.crosstab(predictors['Predicted'],predictors['Actual'])
    print Confusion
    #Performance
    performance = pd.crosstab(predictors['Predicted'],predictors['Actual']).values
    #Part 3    
    print( "Model accuracy is " + str( (performance[0][0] + performance[1][1])/float(len(predictors))) )
    #Part 2    
    plt.plot(lnpW_xy)
    plt.title('Joint y,wt log likelihood')
    #Parts 4 5 6 or whatever are done via helper function from part1 that plots

    
    
 
    
    
    

main()