import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation,Flatten
import pickle
from tensorflow.keras.layers import Conv2D,MaxPooling2D


pickle_in=open(r"C:\Users\SShyam1\PetImages\x.pickle","rb")
x=pickle.load(pickle_in)
pickle_in=open(r"C:\Users\SShyam1\PetImages\y.pickle","rb")
y=pickle.load(pickle_in)
#print(x) #x shows features
#print(y) #y shows the labels #0 refers to Dog & 1 refers to cat because that's how we have indexed in cnn_creating_data file

x=x/255.0

print(x)

model = Sequential()

model.add(Conv2D(256,(3,3),input_shape=x.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(256,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))

model.add(Dense(1)) #The brain - The piece that memorizes
model.add(Activation('sigmoid')) #since two labels, so sigmoid | If 3 labels, softmax

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,batch_size=8,epochs=10,validation_split=0.3)

model.save(r"C:\Users\SShyam1\dogs_vs_cats_updated_CNN.model")
