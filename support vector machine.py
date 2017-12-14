import pandas as pd
import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
import sklearn
from sklearn import svm
import matplotlib.pyplot as plt


def module(l, s):

	
	df = pd.read_csv('IBM_train.csv')

	Open = df['Open']
	high = df['High'].values
	low = df['Low'].values
	volume = df['Volume'].values
	close = df['Close'].values

	# Reading Panda's dataframe from the IBM's data set

	train_label = Open.copy()
	for i in range(l-1, len(Open) - l):
	    if(close[i + l] > Open[i]):
	        train_label[i] = 1
	    else: train_label[i] = -1
	train_label = train_label[l-1:-(s - l)]

	# Creating training label


	vol = volume.copy()
	for i in range(1, len(vol)):
	    vol[i] = ((volume[i] - volume[i-1])/volume[i-1])*100
	vol = vol[s-1:]

	# Creating volatility as feature for training


	high10 = high.copy()
	low10 = low.copy()
	Open10 = Open.copy()
	volume10 = volume.copy()

	r = len(high)
	for i in range(s, r+1):
	    high10[i - s] = np.mean(high[i-s:i]) 
	    low10[i - s] = np.mean(low[i-s:i]) 
	    Open10[i - s] = np.mean(Open[i-s:i]) 
	    volume10[i - s] = np.mean(volume[i-s:i])
	    
	Open10 = Open10[:r-s+1]
	low10 = low10[:r-s+1]
	high10 = high10[:r-s+1]
	volume10 = volume10[:r-s+1]

	# Recreating training features for the sliding window, with the size s


	df_test = pd.read_csv('IBM_test.csv')

	Open_test = df_test['Open']
	high_test = df_test['High'].values
	low_test = df_test['Low'].values
	volume_test = df_test['Volume'].values
	close_test = df_test['Close'].values

	# Reading testing data


	test_label = Open_test.copy()

	for i in range(l-1, len(Open_test) - l):
	    if(close_test[i + l] > Open_test[i]):
	        test_label[i] = 1
	    else: test_label[i] = -1
	test_label = test_label[l-1:-(s-l)]

	# Creating testing label


	high_test10 = high_test.copy()
	low_test10 = low_test.copy()
	Open_test10 = Open_test.copy()
	volume_test10 = volume_test.copy()

	r = len(high_test)

	for i in range(s, r +1):
	    high_test10[i - s] = np.mean(high_test[i-s:i]) 
	    low_test10[i - s] = np.mean(low_test[i-s:i]) 
	    Open_test10[i - s] = np.mean(Open_test[i-s:i])
	    volume_test10[i - s] = np.mean(volume_test[i-s:i])
	    
	Open_test10 = Open_test10[:r-s+1]
	low_test10 = low_test10[:r-s+1]
	high_test10 = high_test10[:r-s+1]
	volume_test10 = volume_test10[:r-s+1]

	# Recreating the testing features for sliding window



	vol1 = volume_test.copy()
	vol1.shape

	for i in range(1, len(vol1)):
	    vol1[i] = ((volume_test[i] - volume_test[i-1])/volume_test[i-1])*100
	vol1 = vol1[s-1:]

	# Creating the feature of testing data of volatility

	train_feature = np.column_stack([Open10,high10,low10, volume10/10000, vol])   
	# Stacking up training features

	test_feature = np.column_stack([Open_test10,high_test10,low_test10,volume_test10/10000, vol1])
	# Testing feature


	clf = svm.SVC()
	clf.fit(train_feature, train_label.astype('int'))
	
	# Predicting

	global result
	result = clf.score(test_feature.astype('int'), test_label.astype('int'))

	# Getting the result


#module(60, 270)
#print(result)


past = 270
res = []
future = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130] ## Future 


for i in range(0, 13):
	module(future[i], past)
	res.append(result)

plt.plot(future, res)
plt.title('Prediction of stock market price trend')
plt.ylabel('Accuracy')
plt.xlabel('Prediction on future M days based on every past 270 days')
plt.show()




