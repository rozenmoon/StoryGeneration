
# _set = 'dsbksdbv.k'

# temp =''
# for i in 'Rozen.moon':
# 	if(i != "."):
# 		temp += i;
# 	else:
# 		temp += ' ';
# 		temp +='.';
# 		temp += ' ';0
# print(temp)


import sys
# import csv
# import numpy
import pickle
x = [1 , 2 ,3,4,5,6,7,8,9]
y = [1 , 2 ,3,4,5,6,7,8,9]
i = 10
for i in range(10):
	with open("WordTOVector_Input_Output"+str(i)+".pkl", 'wb') as output:
		    pickle.dump(x, output, pickle.HIGHEST_PROTOCOL)
		    pickle.dump(y, output, pickle.HIGHEST_PROTOCOL)
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.layers import Dropout
# from keras.layers import LSTM
# from keras.callbacks import ModelCheckpoint
# from keras.utils import np_utils

# raw_text = []
# _set = []
# _distinct = []
# with open('ROCStories_spring2016.csv') as f:
# 	reader = csv.reader(f)
# 	for row in reader:
# 		if(reader.line_num != 0):
# 			raw_text.append(row[1:7]);

# a = []

# _Input = []
# for sent in raw_text:
# 	x = []
# 	for elements in sent:
# 		a = elements.split();
# 		for word in a:
# 			_set.append(word.lower());
# 			x.append(word.lower());
# 	_Input.append(x)

# print(_set)

# for i in _set:
# 	temp =''
# 	for j in i:
# 		if(j != '.'):
# 			temp += i;
# 		else:
# 			temp += ' ';
# 			temp +='.';
# 			temp += ' ';
# 	i = temp

# special_char = ['.',',','!','-','$']

# def findchar( str_char ):
# 	if str_char in special_char:
# 		return (str_char);
# 	else:
# 		return None;


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
# print(_setfinal)
# _distinct = sorted(list(set(_setfinal)))
# print(len(_distinct))

# _distinct = sorted(list(set(_set)))
# print(len(_distinct))
# char_to_int = dict((c, i) for i, c in enumerate(_distinct))
# int_to_char = dict((i, c) for i, c in enumerate(_distinct))
# # summarize the loaded data
# n_chars = len(raw_text)
# n_vocab = len(_distinct)
# print ("Total Characters: ", n_chars)
# print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers