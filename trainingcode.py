import csv
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
import numpy as np
import gensim
from gensim import corpora, models, similarities
import os
import json
import nltk
from keras.layers.recurrent import LSTM,SimpleRNN
from sklearn.model_selection import train_test_split
import theano
from keras import optimizers

theano.config.optimizer="None"


special_char = ['.',',','!','-','$']

def findchar( str_char ):
	if str_char in special_char:
		return (str_char);
	else:
		return None;

raw_text = []
_set = []
_distinct = []

# reading the csv file data into dataframe
with open('testdataset.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		if(reader.line_num != 0):
			raw_text.append(row[1:7]);

# reading the wordvec model
os.chdir("D:\Deep Learning\ChatBot");
model = gensim.models.Word2Vec.load('word2vec.bin');

os.chdir("D:\Deep Learning\story_making\datasets");
a = []

# reading data into _Input which include parsing it a special charecter like ['.',',','!','-','$']
_Input = []
for sent in raw_text:
	x = []
	for elements in sent:
		a = elements.split();
		for i in range(len(a)):
			temp =''
			for j in a[i]:
				xyz = findchar(j)
				if(xyz == None):
					temp += j;
				else:
					temp += ' ';
					temp += xyz;
					temp += ' ';
				a[i] = temp
		b = []
		for i in a:
			xj = []
			xj = i.split()
			for xji in xj:
				b.append(xji)
		for word in b:
			x.append(word);
	_Input.append(x)
Input = _Input[1:103]
print(Input)


# seperating X and Y
sentend=np.ones((300,),dtype=np.float32) 
seq_length = 5
dataX =[]
dataY =[]
for j in Input:
	for i in range(0, len(j)-seq_length, 1):
		seq_in = j[i:i + seq_length]
		seq_out = j[i + seq_length]
		ga = []
		for seq_in_x in seq_in:
			if seq_in_x in model.vocab:
				ga.append(model[seq_in_x])
			else:
				ga.append(sentend)
		dataX.append(ga)
		if seq_out in model.vocab:
			dataY.append(model[seq_out])
		else:
			dataY.append(sentend)

x_train,x_test,y_train,y_test = train_test_split(dataX, dataY, test_size=0.2, random_state=1)
print("done preparing data")
optimized_Adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-06)
model = Sequential()
model.add(LSTM(300,input_shape=(5,300)))
model.add(Dense(300,activation='softmax'))
model.compile(loss='cosine_proximity', optimizer= optimized_Adam, metrics=['accuracy'])
model.summary()

filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
model.fit(np.array(x_train), np.array(y_train), epochs =10,batch_size=10,validation_data=(np.array(x_test),np.array(y_test)), callbacks = callbacks_list)