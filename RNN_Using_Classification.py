import json
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow import keras

DATA_PATH = "data.json"


# load data
def load_data(data_path):
    '''loads training data from file
    :param data_path (str):path to json file containing data
    :return X (ndarray) :inputs
    :return Y (ndarray) :targets
    '''
    with open(data_path, "r") as fp:
        data = json.load(fp)

    x = np.array(data["mfcc"])
    y = np.array(data["labels"])
    return x,y

def prepare_sets(test_size,validation_size):

    #load data
    x,y=load_data(DATA_PATH)

    #train/test split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=test_size)

    #train/validation split
    x_train,x_validation,y_train,y_validation=train_test_split(x_train,y_train,test_size=validation_size)

    #RNN don't expected three dimension array --> (130,13,1)

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def build_model(input_shape):
    """Generates RNN-LSTM model

    :param input_shape(tuple):Shape of input set
    :return:RNN-LSTM model
    """
    #create model
    model=keras.Sequential()

    #2 LSTM layers
    model.add(keras.layers.LSTM(64,input_shape=input_shape,return_sequences=True))
    model.add(keras.layers.LSTM(64))

    #Dense layer
    model.add(keras.layers.Dense(64,activation='relu'))
    model.add(keras.layers.Dropout(0.3))

    #output layer
    model.add(keras.layers.Dense(10,activation='softmax'))

    return  model

if __name__=="__main__":
    #create train,validation and test sets
    x_train,x_validation,x_test,y_train,y_validation,y_test=prepare_sets(0.25,0.2)

    #create network
    input_shape=(x_train.shape[1],x_train.shape[2]) #130-->time steps and 13-->coeficcient of mfcc
    model=build_model(input_shape)

    #compile network
    optimizer=keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    #train network
    model.fit(x_train,y_train,validation_data=(x_validation,y_validation),batch_size=32,epochs=30)

    #evaluate network
    test_error,test_accuracy=model.evaluate(x_test,y_test,verbose=2)
    print("Accuracy is this test set :{}".format(test_accuracy))
    print("Error is this test set :{}".format(test_error))
