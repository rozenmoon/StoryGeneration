import pickle
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils
import os


os.chdir("D:\Deep Learning\story_making\datasets\data");
Input =[]
Output = []
for i in range(2):
	with open("Input_output"+str(i)+".pkl", 'rb') as input:   
	    dataX = pickle.load(input)
	    dataY = pickle.load(input)
	Input.append(dataX)
	Output.append(dataY)

n_patterns = len(dataX)
print(Input)
print(Output)

# X = numpy.reshape(dataX, (n_patterns, seq_length, 1))
# # normalize
# X = X / float(n_vocab)
# X = X[0:1909075]
# dataY = dataY[0:1909075]
# # one hot encode the output variable
# # y = np_utils.to_categorical(dataY)



# os.chdir("D:\Deep Learning\story_making\datasets\weights for model with Word2Vec");
# model = Sequential()
# model.add(LSTM(512, batch_input_shape=(175,Input.shape[1], Input.shape[2]),stateful=True))
# model.add(Dropout(0.2))
# model.add(Dense(300, activation='sigmoid'))
# model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
# model.summery()

# # # define the checkpoint
# filepath="weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
# callbacks_list = [checkpoint]

# model.fit(Input, Output, epochs=1, batch_size=175, callbacks=callbacks_list)
