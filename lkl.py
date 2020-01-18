import numpy as np
import csv
import sys
import pickle
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support

from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization, LSTM, Embedding, Reshape
from keras.layers import Bidirectional
from keras.utils import np_utils

from keras.layers import Conv1D,MaxPooling1D
from sklearn.metrics import accuracy_score

from sklearn.utils.class_weight import compute_class_weight
import itertools

from keras.callbacks import ModelCheckpoint  
from keras.callbacks import EarlyStopping
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

np.seterr(divide='ignore',invalid='ignore')




with open('headline.csv', encoding="utf8") as csvfile_stance:
    stanceReader=csv.reader(csvfile_stance)
    headline=[]
   
    for row in stanceReader:
        temp=[]
        for c,c_ in zip(row[0],row[0][1:]):
            if c.isalnum() or c.isspace():
                temp.append(c.lower())
            else:
                if not c_.isspace():
                    temp.append(" ")
        headline.append([''.join(temp),row[0]])
sh=np.shape(headline)[0]-1

print("Finish loading training stances\n")
print(headline[1:6])


print("Start loading training bodies\n")
with open('body.csv', encoding="ISO-8859-1") as csvfile_body:
    bodyReader=csv.reader(csvfile_body)
    bodies={}
    for row in bodyReader:
        temp=[]
        for c,c_ in zip(row[0],row[0][1:]):
            if c.isalnum() or c.isspace():
                temp.append(c.lower())
            else:
                if not c_.isspace():
                    temp.append(" ")
        bodies[row[0]]=''.join(temp)
print("Finish loading training bodies")
print(bodies)



##This functions reads the processed data empeddings and divides the data into train and test validation set.


data=pickle.load(open("processed_data_embed.p","rb"))
x = []
y = []
for data_row in data:
    x.append(data_row[0].tolist())
    y.append(data_row[1])

x = x[1:]
y = y[1:]
x = np.array(x)
x_dummy = np.reshape(x,(len(x),50,1))
dummy_y = np_utils.to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(x_dummy, dummy_y, test_size=0.2, random_state=42)
X_test=X_test[range(sh),]
 
    
"""
This function creates the keras model for the input parameters.
"""

def create_model(num_classes,num_lstm_units,X_train_shape,kernel_size):
	# create model
	model = Sequential()
	model.add(Bidirectional(LSTM(units = num_lstm_units ,return_sequences=True),
                         input_shape=(X_train_shape[1],X_train_shape[2])))

	model.add(Conv1D(filters=16, kernel_size=kernel_size, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
    
	model.add(Conv1D(filters=32, kernel_size=kernel_size, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
    
	model.add(Conv1D(filters=64, kernel_size=kernel_size, padding='same', activation='relu'))
	model.add(MaxPooling1D(pool_size=2))
    
	model.add(Dropout(0.3))
	model.add(Flatten())
    
	model.add(Dense(150, activation='relu'))
	model.add(Dropout(0.4))
    
	model.add(Dense(num_classes,activation = 'softmax'))
    
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
	model.summary()
	return model


lstm=32
ker=5
batch=128
num_classes = dummy_y.shape[1]
input_shape = X_train.shape
class_weight = compute_class_weight('balanced', np.unique(y), y)
class_weight_norm = class_weight/np.linalg.norm(class_weight)
model = create_model(num_classes =num_classes,num_lstm_units = lstm,X_train_shape = input_shape,kernel_size = ker)
model.load_weights('results/batch/32_5_128.weights.best.hdf5')
y_pred = model.predict_classes(X_test)

print(y_pred)


