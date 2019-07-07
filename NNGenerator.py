import os
import keras
import random
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from keras.preprocessing import image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import PIL
from PIL import Image
from sklearn.metrics import classification_report
from keras.models import model_from_json


class NNGenerator:

    neurons = []
    activ_fct = []
    height = 48
    width = 48

    # def __init__(self):


    def create(self, neurons, activ_fct):
        self.model = Sequential()
        self.neurons = neurons
        self.activ_fct = activ_fct
        self.model.add(Flatten(input_shape=(self.height, self.width,1)))
        for i in range(len(self.neurons)):
            self.model.add(Dense(self.neurons[i], activation=self.activ_fct[i] , kernel_initializer="random_uniform", bias_initializer="zeros"))
            self.model.add(Dropout(0.1))
        self.model.add(Dense(4, activation='softmax', kernel_initializer="random_uniform", bias_initializer="zeros"))
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    def get_model(self):
        return self.model

    
    def read_dataset(self, path):
        train_image = []
        train_image_advace = []
        train_names = ['circle','square','star','triangle']


        # relative_path = '/home/jacek/Downloads/erasmus/study/knowladge and reasoning/project/full_dataset/'
        relative_path = path
        size = []
        j = 0
        for i in train_names:
            path_to_basic_data = path + i + '/'
            basic_data_list = self.listdir_nohidden(path_to_basic_data)
            size.append(len(basic_data_list))
            print()
            j+=1
            for x in basic_data_list:
                img = image.load_img(path_to_basic_data + x, target_size=(self.height,self.width,1), grayscale=True)
                img = image.img_to_array(img)
                img = img/255
                train_image_advace.append(img)
            X_advance = np.array(train_image_advace)

        y_advance = []    
        for i in range(4):
            for j in range(size[i]):
                y_advance.append(i)

        y_advance = to_categorical(y_advance)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X_advance, y_advance, random_state=100, test_size=0.3)

    def train_nn(self, epochs=15):
        self.model.fit(self.X_train, self.y_train, epochs, validation_data=(self.X_test, self.y_test), verbose = 0)

    def classify_image(self, img_path):
        test_image = []
        img = image.load_img(path, target_size=(self.height,self.width,1), grayscale=True)
        img = image.img_to_array(img)
        img = img/255

        test_image.append(img)
        test = np.array(test_image)
        prediction = self.model.predict(test)

        text = ''
        train_names = ['circle','square','star','triangle']
        for p in prediction:
            for i, k in enumerate(train_names):
                text += k + ' ' +format(p[i]*100, '.2f') +'%; '

        return text 

    def save_model(self):
        # serialize model to JSON
        model_json = self.model.to_json()
        with open("model.json", "w") as json_file:
            json_file.write(model_json)
        # serialize weights to HDF5
        self.model.save_weights("model.h5")
        # print("Saved model to disk")

    def read_model(self):
        json_file = open('model.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        loaded_model.load_weights("model.h5")
        # print("Loaded model from disk")
        self.model = loaded_model
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        config = self.model.get_config()
        neurons = []
        activ_fct = []
        for layer in config["layers"]:
            if(layer["class_name"] == 'Dense'):
                c = layer["config"]
                neurons.append(c["units"])
                activ_fct.append((c["activation"]))

        del neurons[-1]
        del activ_fct[-1]

        return neurons, activ_fct

    def get_nn_score(self):
        Y_test = np.argmax(self.y_test, axis=1) 
        y_pred = self.model.predict_classes(self.X_test)
        return classification_report(Y_test, y_pred)
        # scores = self.model.evaluate(self.X_test, self.y_test, verbose=0)
        # return "%s: %.2f%%" % (self.model.metrics_names[1], scores[1]*100)

    def listdir_nohidden(self,path):
        dir = []
        for f in os.listdir(path):
            if not f.startswith('.'):
                dir.append(f)
        return dir