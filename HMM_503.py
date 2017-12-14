# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 16:18:49 2017

@author: Mentat
"""

import matplotlib.dates as dates
import pandas as pd
import datetime
import pandas as pd
from hmmlearn import hmm
import numpy as np
import warnings
#warnings.filterwarnings("ignore")



def nice():
    original = pd.read_csv('IBM.csv')
    mask = (original['Date'] > '2002-12-10') & (original['Date'] <= '2017-12-10')
    original.loc[mask]
    
    
    original['f_high']=(original['High']-original['Open'])/original['Open']
    original['f_low']=(original['Low']-original['Open'])/original['Open']
    original['f_close']=(original['Close']-original['Open'])/original['Open']
    original['label']=original['f_close']>=0
    
    del original['Volume']
    del original['Adj Close']
   
    original.to_csv('data.csv',index=False)
    #print(original)
    
   
    
def value_create():
    np.arange(0, 0.1, 0.01)
    value=[]
    for i in np.arange(-0.1, 0.1, 0.2/50):
        for j in np.arange(0, 0.1, 0.01):
            for k in np.arange(0, 0.1, 0.01):
                value.append([i,j,-k])
    value=np.array(value)
    return value
    
nice()    
data = pd.read_csv('data.csv')
mask = (data['Date'] > '2014-09-01') & (data['Date'] <= '2017-09-05')
train=data.loc[mask]
test=data.loc[(data['Date'] > '2016-12-10') & (data['Date'] <= '2017-02-10')]
test=data.loc[(data['Date'] > '2017-09-01') & (data['Date'] <= '2017-11-01')]

Date=pd.to_datetime(train['Date'])
close=train['Close']

A=np.column_stack([train['f_close'],train['f_high'], train['f_low']])

B=np.column_stack([test['f_close'],test['f_high'], test['f_low']])


rio=11*np.ones((A.shape[0]/11,), dtype=np.int)
model = hmm.GaussianHMM(n_components= 4,covariance_type="full", n_iter=20000)



model.fit(A,rio)
hidden_states = model.predict(A)

#plot
import matplotlib.pyplot as plt   # Import matplotlib\n",


#pylab.rcParams['figure.figsize'] = (15, 9)   

train["Close"].plot(grid = True) # Plot the adjusted closing price of AAPL
plt.show()
plt.gcf().clear()

#plt.figure(figsize=(25, 18)) 

value=value_create()
print('why')
#train




label=[]
#print(B.shape)
#print(value[1,:].reshape(1,3).shape)
fa=0
res=[]
for i in range(B.shape[0]-12):
    #value[:,0]=B[i+11,0]
    maxi=np.inf
    temp=np.zeros((10,3))
    temp[0:10,:]=B[i:i+10,:]
    a=model.decode(B[i:i+10,:])
    res.append(a[1][-1])
    

    label.append(B[i+11,0]>0)
    
    
    
print(model.startprob_)
print(model.transmat_)
print(model.means_)

val=model.means_[:,0]>0
pla=np.argmax(model.transmat_,axis=1)
print(val,pla)

result=val[pla[res]]
np.sum(result==label)/len(label)



def not_worked():
    #do a grid search on possible value of features [(-0.1,0.1),(0,0.1),(0,0.1)]
    #need GMM HMM?
    #caculate the P(O|HMM parameter) O contains the estimated value for today and d days before it    
    argm=[0,0,0]
    result=[]
    result_la=[]
    label=[]
    save=[]
    nyc=[]
    #print(B.shape)
    #print(value[1,:].reshape(1,3).shape)
    fa=0
    for i in range(B.shape[0]-21):
        maxi=-np.inf
        temp=np.zeros((21,3))
        temp[0:20,:]=B[i:i+20,:].copy()
        
        for j in range(value.shape[0]):
            #print(j,value[j,:])
            temp[20,:]=value[j,:].copy()
            a=model.decode(temp)
            #a=model.predict(temp)
            #print(a,a[1])
            bb=a[1]
            a=a[0]
            
            if a>maxi:
                #print(a)
                
                argm=j
                maxi=a
                #print(a,bb)
                
            save.append(a)
        nyc.append(argm)
    
        result.append(np.array(np.copy(value[argm,:])))
        result_la.append(value[argm,0]>0)
        label.append(B[i+10,0]>0)
        #print('last piece of work in BU !!, it has been a pleasure to take this course. I have learned a lot')