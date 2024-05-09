#Script containing the necessary functions to prepare the dataset to train and test


import numpy as np
import scipy.io
import cv2
import os
import sys
import deepdish as dd
from PIL import Image

from transformers import AutoFeatureExtractor, ViTImageProcessor

def feature_extractor(X,typeTransformer):
  if typeTransformer=='vit':
    feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224-in21k")
    return_tensors='pt'
  else:
     feature_extractor = ViTImageProcessor.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k")
     return_tensors='tf'

  X_transformer=np.zeros((len(X),3,224,224))
  
  nsamples=0
  for i in X:
    X_transformer[nsamples]=feature_extractor(i, return_tensors=return_tensors)['pixel_values']
    nsamples=nsamples+1

  return X_transformer


def get_partition_convNext(set=1):
  # 1. Indexes of the images that will correspond to train and test
  #trainfrs = [1:300]; 
	#testfrs = [301:589];
  p1=[]
  p2=[]
  for i in range(1, 590):
    if i in range(1,301):
      p1.append(i)
    if i in range(301,590):
      p2.append(i)
  
  # 2. Set 
  if set==1:
    train_val=p1
    test=p2
  else:
    train_val=p2
    test=p1

  # TRAIN/VALIDATION/TEST
  train=[]
  validation=[]

  for i in range(0, len(train_val)):
    if i in [1,11,21,31,41,51,61,71,81,91,101,111,121,131,141,151,161,171,181,191,201,211,221,231,241,251,261,271,281,291]:
      validation.append(train_val[i])
    else:
      train.append(train_val[i])


  return train, validation, test




