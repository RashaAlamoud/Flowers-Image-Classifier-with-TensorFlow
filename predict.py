import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub
import numpy as np
import json
import logging

logger = tf.get_logger()

parser=argparse.ArgumentParser(description="Image classifier -Prediction part")
parser.add_argument('--Model',default='./flower.h5',help= 'The Model')
parser.add_argument('--Img_path',default='./test_images/cautleya_spicata.jpg',help= 'Image Path')
parser.add_argument('--top_k',default=5,help= 'return the top 5 of classes')
parser.add_argument('--category_names',default='./label_map.json',help= 'Mapping json file with name of flowers')

batch_size = 32
image_size = 224
image_shape=(image_size,image_size,3)
dataset, dataset_info = tfds.load('oxford_flowers102', as_supervised=True ,with_info=True)
num_classes = dataset_info.features['label'].num_classes
def process_image(image):
    image=tf.convert_to_tensor(image,tf.float32)
    image=tf.image.resize(image, (image_size, image_size))
    image/=255
    return image

with open('label_map.json', 'r') as f:
    class_names = json.load(f)


def  predict(image_path, model, top_k):
    

    img=Image.open(image_path)
    test_img=np.asarray(img)
    trandfrom_img=process_image(test_img)
    redim_img=np.expand_dims(trandfrom_img,axis=0)
    prob_pred=model.predict( redim_img)
    prob_pred= prob_pred.tolist()
    probs,classes=tf.math.top_k(prob_pred,k=top_k)
   
    return probs,classes


        
URL="https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
feature_extractor=hub.KerasLayer(URL,input_shape=(image_shape))
feature_extractor.trainable=False
    
rate=0.2
model = tf.keras.Sequential([
                                 feature_extractor,
                                 tf.keras.layers.Dense(600,activation='relu'),
                                 tf.keras.layers.Dropout(rate),
                                 tf.keras.layers.Dense(300,activation='relu'),
                                 tf.keras.layers.Dropout(rate),
                                 tf.keras.layers.Dense(num_classes, activation='softmax')
                                ])

 
    
model.summary()

model.compile(optimizer='adam',
                 loss='sparse_categorical_crossentropy',
                 metrics=['accuracy'])

def filtered(classes):
    return[class_names_new.get(str(key))
         if key else "Placeholder" for key in classes.numpy().squeeze().tolist()]