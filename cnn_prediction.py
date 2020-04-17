import cv2
import PIL
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from tensorflow.keras.models import load_model

CATEGORIES=['Dog','Cat']

image=r"C:\Users\SShyam1\petimagestest\106.jpg"

def prepare(filepath):
    img_size=100
    img_array=cv2.imread(image,cv2.IMREAD_GRAYSCALE)
    new_array=cv2.resize(img_array,(img_size,img_size))
    return new_array.reshape(-1,img_size,img_size,1)

model=tf.keras.models.load_model(r"C:\Users\SShyam1\dogs_vs_cats_updated_CNN.model") 

prediction = model.predict([prepare(image)])
print(int(prediction[0][0]))
print(CATEGORIES[int(prediction[0][0])])
img=mpimg.imread(image)
imgplot=plt.imshow(img)
plt.title(CATEGORIES[int(prediction[0][0])])
plt.show()
