# Import
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from keras.applications.inception_v3 import preprocess_input
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array

def main():
    # Load classe names
    print('Load classe names...')
    class_names = []
    for idx1, folder in enumerate(os.listdir('C:/Users/james/source/python/P6/Images')):
        if folder != 'train' and folder != 'val' and folder != 'test':
            class_name = folder.split('-')[1]
            class_names.append(class_name)
    
    # Load model
    print('Load model...')
    model = tf.keras.models.load_model('C:/Users/james/source/python/P6/InceptionV3_tuned')
    
    # load an image from file
    # C:/Users/james/source/python/P6/Images/n02090622-borzoi/n02090622_5556.jpg
    # C:/Users/james/source/python/P6/Images/n02089973-English_foxhound/n02089973_1303.jpg
    path = input("Enter your image path: ")
        
    print('Load an image from file...')
    image = load_img(path, target_size=(224, 224))
    # convert the image pixels to a numpy array
    image = img_to_array(image)
    # reshape data for the model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # prepare the image for the InceptionV3 model
    image = preprocess_input(image)
    
    # predict the probability across all output classes
    print('Predict class...')
    print('')
    prediction = model.predict(image)
    # convert the probabilities to class labels
    pred_label =  class_names[np.argmax(prediction)]
    
    # print the classification
    print(pred_label, str(prediction[0][np.argmax(prediction)]*100) + '%')
    
if __name__ == "__main__":
    main()