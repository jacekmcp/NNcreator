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

random.seed(100)

height = 48
width = 48

train_image = []
train_image_advace = []
train_names = ['circle','square','star','triangle']
relative_path = '/home/jacek/Downloads/erasmus/study/knowladge and reasoning/project/full_dataset/'
for i in train_names:
  for x in range(251):
   # print(relative_path + i + "/" + str(x) + ".png")
    img = image.load_img(relative_path + i + "/" + str(x) + ".png", target_size=(height,width,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image_advace.append(img)
X_advance = np.array(train_image_advace)

y_advance = []    
for i in range(4):
  for j in range(251):
   # print (i)
    y_advance.append(i)

y_advance = to_categorical(y_advance) 
  
for i in train_names:
    img = image.load_img('/home/jacek/Downloads/erasmus/study/knowladge and reasoning/project/Formas_1/0_' + i + ".png", target_size=(height,width,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    train_image.append(img)
X = np.array(train_image)

y=  [0,1,2,3]
y = to_categorical(y)
X_train, X_test, y_train, y_test = train_test_split(X_advance, y_advance, random_state=100, test_size=0.3)

#****************************CNN********************#
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=(200,200,1)))
# model.add(Conv2D(64, (3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10, activation='softmax'))

# model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
# model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

model = Sequential()
model.add(Flatten(input_shape=(height, width,1)))
model.add(Dense(128, activation='relu' , kernel_initializer="random_uniform", bias_initializer="zeros"))
model.add(Dropout(0.1))
model.add(Dense(64, activation='relu' , kernel_initializer="random_uniform", bias_initializer="zeros"))
model.add(Dropout(0.05))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#************************Initial set FORMAS_1*******************#
# model.fit(X, y, epochs=20)

#************************FORMAS_2*******************#
model.fit(X_train, y_train, epochs=20, validation_data=(X_test, y_test))

model.summary()


test_image = []
for i in range(1,4):
    img = image.load_img('/home/jacek/Downloads/erasmus/study/knowladge and reasoning/project/handmade/triangle/' + str(i) + '.png', target_size=(height,width,1), grayscale=True)
    img = image.img_to_array(img)
    img = img/255
    test_image.append(img)

test = np.array(test_image)
prediction = model.predict_classes(test)

for p in prediction:
    for i, k in enumerate(train_names):
        print(k + ' ' +format(p[i]*100, '.2f') +'%;') 
    print('*******************8')
# print(prediction)

#*********************testing************************#
# for i, k in enumerate(train_names):
#     test_image = []
#     for j in range(201,251):
#         img = image.load_img('/home/jacek/Downloads/erasmus/study/knowladge and reasoning/project/Formas_3/'+ k + '/' + str(j) + '.png', target_size=(height,width,1), grayscale=True)
#         img = image.img_to_array(img)
#         img = img/255
#         test_image.append(img)
    
#     test = np.array(test_image)
#     prediction = model.predict_classes(test)

#     counter = 0
#     for p in prediction:
#         if(p == i):
#             counter = counter +1

#     res = (counter / 50) * 100
#     print(k + ' recognition = ' + str(res) + '%')

# test = np.array(test_image)
# prediction = model.predict_classes(test)
# print(prediction)

#ACCURACY FOR EACH 
