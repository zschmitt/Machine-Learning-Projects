import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
import keras
import os
from keras.preprocessing.image import load_img, save_img
from keras.preprocessing.image import img_to_array
from simple_image_download import simple_image_download as simp
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.regularizers import *
from tensorflow.keras.utils import to_categorical

from simple_image_download import simple_image_download as simp #This is how I downloaded training data.
                                                               # Credit shall be given to:
                                                               #  https://github.com/RiddlerQ/simple_image_download
                                                               # Please use "pip install simple_image_download" before compiling 



#Function to open camera and predict your emotion
def Activate_Camera(model):
   camera = cv2.VideoCapture(0)
   CurrentFrame = 0
   processed_list = []
   path = os.path.expanduser("~")+"/Downloads/simple_images/simple_images/data/frame"

   while(True):
      ret, frame = camera.read()
      font = cv2.FONT_HERSHEY_SIMPLEX

      if cv2.waitKey(1) & 0xFF == ord('q'): #press "q" to quite out of application
         break

      if ret:
         #Save each frame in the appropriate format
         name = path + str(CurrentFrame) 
         process_img = img_to_array(frame)
         X = cv2.resize(process_img, (32,32))
         cv2.imwrite(name + '.jpg', X)
         X = (X/255)
         X = np.array([X])   # X = formated frame 

         #Make a prediction based on teh current frame 
         prediction = int(model.predict_classes(X))
         prediction = class_names[prediction] #Map numerical class prediction to corresponding emotion

         #Now show the image and prediction on screen
         frame = cv2.putText(frame,  
                (prediction),  
                (50, 50),  
                font, 1,  
                (255, 0, 0),  
                2,  
                cv2.LINE_AA) 
         cv2.imshow('frame', frame)

         CurrentFrame += 1 #update next frame number
      else: 
         break

   camera.release() 
   # Destroy all the windows 
   cv2.destroyAllWindows() 

#get the training data 
def GetPictures(class_names = []):
   for emo in class_names:
      response = simp.simple_image_download
      response().download(emo + 'human face',picture_amount) #use "__ human face" becuase just "___ face" scrapes emojis too 




#Function to preprocess all images into learnable data
#input is the list of emotions so that we can look up corresponding directories 
#output are two n-D
def get_img_array(class_names):
    labels_arr= np.empty(shape=[0,1])
    processed_list = []
    num = 0
    for i in class_names:
       path = os.path.expanduser("~")+"/Downloads/simple_images/"+ i + 'human_face/'
       image_list = [img for img in os.listdir(path)]
       for img_name in os.listdir(path):
            loaded_image = load_img(path + img_name , grayscale = False)
            process_img = img_to_array(loaded_image)
            resized_image = cv2.resize(process_img, (32,32))
            cv2.imwrite(path + img_name + '.jpg', resized_image)
            processed_list.append(resized_image/255)
       labels = np.full((len(image_list),1),num)
       labels_arr= np.append(labels_arr, labels)
       num = num+1
    return processed_list,labels_arr

      
#Call this function to display text(prediction) in camera window
def __draw_label( img, text, pos, bg_color): #Obtained from GeeksforGeeks 
    font_face = cv2.FONT_HERSHEY_PLAIN
    scale = 0.4
    color = (0, 0, 0)
    thickness = cv2.FILLED
    margin = 2

    txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    end_x = pos[0] + txt_size[0][0] + margin
    end_y = pos[1] - txt_size[0][1] - margin

    cv2.rectangle(img, pos, (end_x, end_y), bg_color, thickness)
    cv2.putText(img, text, pos, font_face, scale, color, 1, cv2.LINE_AA)

  

def Main():

   #Get the training data
   GetPictures(class_names)
   X, Y = get_img_array(class_names) #Preform data preprocessing
   Y = to_categorical(Y, 4) #One-Hot-Encode label data 
   X = np.array(X) 


   #define the network
   model = Sequential()

   #layer 1
   model.add(Conv2D(filters = 26,      
                  kernel_size = (3, 3), 
                  strides=(1, 1),
                  activation = 'relu', 
                  input_shape = (32, 32, 3))) #input image is 32x32x3 with RGB

   model.add(MaxPooling2D(pool_size = (2, 2), strides=2))


   #layer 2
   model.add(Conv2D(filters = 32,      
                  kernel_size = (3, 3), 
                  strides = (1,1),
                  activation = 'relu'))

   model.add(MaxPooling2D(pool_size = (2, 2), strides=2))
   model.add(Flatten())

   #fully connected layers
   model.add(Dense(300, activation = 'relu' ))  
   model.add(Dropout(0.7))
   model.add(Dense(56, activation = 'relu')) 
   model.add(Dense(4, activation = "softmax"))

   model.compile(optimizer = RMSprop(),
               loss = 'categorical_crossentropy',
               metrics = ['accuracy'])
   
   #train the model with scraped data
   history = model.fit(X, 
                     Y,
                     batch_size = 30,
                     epochs = 20, 
                     validation_split = 0.2, 
                     shuffle=True,
                     verbose = 1)

   #Turn on the Camera and make predictions 
   Activate_Camera(model)
   return 


#set a few global variables and call main
picture_amount = 500
class_names = ['happy', 'sad', 'angry', 'neutral']
class_dict = {i:class_name for i,class_name in enumerate(class_names)}
Main()