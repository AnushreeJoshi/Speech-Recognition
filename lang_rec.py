from python_speech_features import mfcc

import os
from os import listdir
from os.path import isfile, join, isdir

import numpy as np

from sklearn.mixture import GaussianMixture
from sklearn.linear_model import LogisticRegression
import scipy.io.wavfile as wav

from scipy import stats

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

train_path="data/lang_rec_data/train/"
test_path="data/lang_rec_data/test/"
vec_length=13   #no. of attributes in a single mfcc feature


#func to create mfcc features
def createMFCC(path):
	(rate,sig)=wav.read(path)
	mfcc_feat=mfcc(sig,rate, winlen=0.025, winstep=0.010)

	#print mfcc_feat.shape
	return mfcc_feat


#GMM models
gmm=[]


#-----------------------------TRAINING THE GMMs--------------------------------
language_files=[f for f in listdir(train_path) if isdir(join(train_path,f))]
print language_files

for i in range(0,len(language_files)):
	audioPath=join(train_path,language_files[i])
	audioFiles=[l for l in listdir(audioPath) if isfile(join(audioPath,l))]

	mfcc_temp=np.empty((0,vec_length))
	for j in range(0,len(audioFiles)):
		filePath=join(audioPath,audioFiles[j])
		temp_feat=createMFCC(filePath)
		mfcc_temp=np.append(mfcc_temp,temp_feat,axis=0)   #mfcc features of all files of one class are used to train one gmm

	gmm.append(GaussianMixture(n_components=32,covariance_type='full',max_iter=20, random_state=None))   #training the models
	gmm[i].fit(mfcc_temp)




#---------------------------TESTING---------------------------
test_files=[f for f in listdir(test_path) if isdir(join(test_path,f))]

print "The number of labels identified correctly are:"
for i in range(0,len(test_files)):
	audioPath=join(test_path,test_files[i])
	#print audioPath
	audioFiles=[l for l in listdir(audioPath) if isfile(join(audioPath,l))]
	#print audioFiles

	res=[]
	print "current label",i
	mfcc_temp=np.empty((0,vec_length))
	for j in range(0,len(audioFiles)):
		filePath=join(audioPath,audioFiles[j])
		temp_feat=createMFCC(filePath)   #create mfcc for test file
		#print temp_feat.shape
		
		num_features=temp_feat.shape[0]
		
		labels=np.empty((num_features,1))
		for k in range(0,num_features):
			y=[None]*4
			for l in range(0,4):
				y[l]=np.exp(gmm[l].score_samples(temp_feat[k]))   #score for each feature

			max_class=y.index(max(y))   #the class with max score is the output
			labels[k]=max_class        #append predicted class to a list

		res.append(stats.mode(labels)[0])	  #the mode class for all mfcc features of one file is given as the predicted class for that file
		
		#output no. of correctly predicted labels for each language
	correct=0
	for x in range(0,4):
		if(res[x]==i):
			correct=correct+1
	print correct

