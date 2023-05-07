import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

    #CNN expected three dimension array --> (130,13,1)

    #add channel/third dimension
    x_train=x_train[...,np.newaxis] #4d array-->(num_of_samples,130,13,1)
    x_validation = x_validation[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    return x_train, x_validation, x_test, y_train, y_validation, y_test

def build_model(input_shape):

    #create model
    model=tf.keras.Sequential()

    #1st conv layer
    model.add(tf.keras.layers.Conv2D(32,(3,3),activation='relu',input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((3,3),strides=(2,2),padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    #2nd conv layer
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    #3rd conv layer
    model.add(tf.keras.layers.Conv2D(32, (2, 2), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
    model.add(tf.keras.layers.BatchNormalization())

    #flatten the output and feed it into dense layer
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(64,activation='relu'))
    model.add(tf.keras.layers.Dropout(0.3))

    #output layer
    model.add(tf.keras.layers.Dense(10,activation='softmax'))

    return  model

def predict(model,x,y):

    x=x[np.newaxis,...]

    #predictions --> [(0.1,0.2...)] --ten values and different score for each class(two dimensional array)
    prediction=model.predict(x)  #x -->(1)+(130,13,1) need new channel on beginnning array

    # extract index with max value
    predicted_index=np.argmax(prediction,axis=1) # output-->[3] one dimensional array
    print("expected index {}, predicted index {}".format(y,predicted_index))


if __name__=="__main__":
    #create train,validation and test sets
    x_train,x_validation,x_test,y_train,y_validation,y_test=prepare_sets(0.25,0.2)

    #build CNN net
    input_shape=(x_train.shape[1],x_train.shape[2],x_train.shape[3])
    model=build_model(input_shape)

    #compile CNN network
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

    #train CNN network
    model.fit(x_train,y_train,validation_data=(x_validation,y_validation),batch_size=32,epochs=30)

    #evaluate CNN network
    test_error,test_accuracy=model.evaluate(x_test,y_test,verbose=1)
    print("Accuracy is this test set :{}".format(test_accuracy))
    print("Error is this test set :{}".format(test_error))

    #make predict on sample
    x=x_test[100]
    y=y_test[100]

    predict(model,x,y)