import sys
import csv
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

raw_text = []
_set = []
_distinct = []
with open('ROCStories_spring2016.csv') as f:
	reader = csv.reader(f)
	for row in reader:
		if(reader.line_num != 0):
			raw_text.append(row[1:7]);

a = []

_Input = []
for sent in raw_text:
	x = []
	for elements in sent:
		a = elements.split();
		for word in a:
			_set.append(word.lower());
			x.append(word.lower());
	_Input.append(x)

_distinct = sorted(list(set(_set)))

char_to_int = dict((c, i) for i, c in enumerate(_distinct))
int_to_char = dict((i, c) for i, c in enumerate(_distinct))
# summarize the loaded data
n_chars = len(raw_text)
n_vocab = len(_distinct)
print ("Total Characters: ", n_chars)
print ("Total Vocab: ", n_vocab)
# prepare the dataset of input to output pairs encoded as integers
seq_length = 5
dataX =[]
dataY =[]
for j in _Input:
	for i in range(0, len(j)-seq_length, 1):
		seq_in = j[i:i + seq_length]
		seq_out = j[i + seq_length]
		dataX.append([char_to_int[char] for char in seq_in])
		dataY.append(char_to_int[seq_out])

n_patterns = len(dataX)
print ("Total Patterns: ", n_patterns)
# reshape X to be [samples, time steps, features]
X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# normalize
X = X / float(n_vocab)
X = X[0:1909075]
dataY = dataY[0:1909075]
# define the LSTM model

model = Sequential()
model.add(LSTM(256, batch_input_shape=(1,X.shape[1], X.shape[2]),stateful=True))
model.add(Dropout(0.2))
model.add(Dense(1, activation='relu'))
model.compile(loss='mean_squared_error', optimizer='adam')

# load the network weights

filename = "weights-improvement-15-252735484.2737.hdf5"
model.load_weights(filename)
model.compile(loss='mean_squared_error', optimizer='adam')

# pick a random seed

start = numpy.random.randint(0, len(dataX)-1)
pattern = dataX[start]
print ("Seed:")
print ( ''.join([int_to_char[value] for value in pattern]))

# generate characters

for i in range(1000):
	x = numpy.reshape(pattern, (1, len(pattern), 1))
	x = x / float(n_vocab)
	prediction = model.predict(x, verbose=0)
	index = numpy.argmax(prediction)
	result = int_to_char[index]
	seq_in = [int_to_char[value] for value in pattern]
	sys.stdout.write(result)
	pattern.append(index)
	pattern = pattern[1:len(pattern)]
