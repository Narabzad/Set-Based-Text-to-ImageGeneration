import tensorflow as tf 
import cv2
import numpy as np
from scipy import spatial
import math

model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet',input_shape=(300,300,3),pooling='max')

def get_emb_img(img):
    img=cv2.resize(cv2.imread(img),(300,300))
    pred=model.predict(np.array([img])/255)
    return list(pred[0])

def find_relevance(target,img):
    target_emb = get_emb_img(target)
    img_emb = get_emb_img(img)
    cosine_sim = 1 - spatial.distance.cosine(target_emb, img_emb)
    return cosine_sim

def find_relevance_array(target,imgs):
    cosine_sim_grid=[]
    target_emb = get_emb_img(target)
    for img in imgs:
        img_emb = get_emb_img(img)
        cosine_sim = 1 - spatial.distance.cosine(target_emb, img_emb)
        cosine_sim_grid.append(cosine_sim)
    return cosine_sim_grid


