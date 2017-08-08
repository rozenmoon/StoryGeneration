import csv
import pickle
# import numpy
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.callbacks import ModelCheckpoint
import numpy as np
# from keras.utils import np_utils
import gensim
from gensim import corpora, models, similarities
import os
from gensim import corpora, models, similarities

special_char = ['.',',','!','-','$']

def findchar( str_char ):
	if str_char in special_char:
		return (str_char);
	else:
		return None;

raw_text = []
_set = []
_distinct = []
with open('ROCStories_spring2016.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		if(reader.line_num != 0):
			raw_text.append(row[1:7]);

os.chdir("D:\Deep Learning\ChatBot");
model = gensim.models.Word2Vec.load('word2vec.bin');

os.chdir("D:\Deep Learning\story_making\datasets");
a = []

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
print(len(_Input))



# _setfinal =[]
# for i in range(2136563):
# 	temp =''
# 	for j in _set[i]:
# 		xyz = findchar(j)
# 		if(xyz == None):
# 			temp += j;
# 		else:
# 			temp += ' ';
# 			temp += xyz;
# 			temp += ' ';
# 	_set[i] = temp 

# for i in _set:
# 	a = i.split();
# 	for word in a:
# 		_setfinal.append(word.lower());	

# _distinct = sorted(list(set(_setfinal)))
# char_to_int = dict((c, i) for i, c in enumerate(_distinct))

# n_vocab = len(_distinct)
# print ("Total Vocab: ", n_vocab)

# i = 0;
# _min = 100;
# _max = 0;
# for j in _Input:
# 	i = i + len(j);
# 	if(len(j) < _min):
# 		_min = len(j)
# 	if(len(j) >_max):
# 		_max = len(j)

# # vec_equivalent = []
# # for j in _Input:
# # 	x = []
# # 	for elements in j:
# # 		x.append((char_to_int[elements]))
# # 	int_equivalent.append(x)

sentend=np.ones((300,),dtype=np.float32) 
seq_length = 5
dataX =[]
dataY =[]
for j in _Input:
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

os.chdir("D:\Deep Learning\story_making\datasets\data");
for i in range(len(dataX)):
	with open("Input_output"+str(i)+".pkl", 'wb') as output:
	    pickle.dump(dataX[i], output, pickle.HIGHEST_PROTOCOL)
	    pickle.dump(dataY[i], output, pickle.HIGHEST_PROTOCOL)


# n_patterns = len(dataX)
# X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# # normalize
# X = X / float(n_vocab)
# X = X[0:1909075]
# dataY = dataY[0:1909075]
# # one hot encode the output variable
# # y = np_utils.to_categorical(dataY)

# model = Sequential()
# model.add(LSTM(256, batch_input_shape=(175,X.shape[1], X.shape[2]),stateful=True))
# model.add(Dropout(0.2))
# model.add(Dense(1))
# model.compile(loss='mean_squared_error', optimizer='adam')

# # # define the checkpoint
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

# model.fit(X, dataY, epochs=1, batch_size=175, callbacks=callbacks_list)
