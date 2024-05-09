#Script containing the necessary functions to test the trained models
# FUNCTIONS:
# -evaluateAP

import numpy as np
import scipy.io
import cv2
import os

from operator import itemgetter
from sklearn.metrics import average_precision_score


def evaluateAP(model,X_test,y_test,predictModel=True):
  #1. Predict X_test
  if predictModel:
    y_pred_test=model.predict(X_test)
  else:
    y_pred_test=X_test

    

  #2. List to save Ap results for each class
  AP=[]

  #3. For each class we do the following (AP per class)
  for classIdx in range(0,6):	
    #3.1. List to store the values of y_pred and y_test of the class to evaluate
    predClass=[]
    #3.2. For each sample we take the y_pred and y_test of the column corresponding to that class.
    for row in range(0,len(y_pred_test)):
      predClass.append([y_pred_test[row][classIdx],y_test[row][classIdx]])
    #3.3. We order from highest to lowest according to the value of the predictions of that class.
    predClass=sorted(predClass, key=itemgetter(0),reverse=True)

    #3.4. The values of the list are distributed in their corresponding y_pred, y_True
    y_pred=[]
    y_true=[]
    for i in predClass:		
      y_pred.append(i[0])
      y_true.append(i[1])

    #print(y_pred)
    #print(y_true)
    
    #4. Apply the average_precision_score function to the y_pred and y_test values of that class.
    AP.append(average_precision_score(y_true, y_pred))

  #5. Save AP
  resAP={}
  resAP['HAND_HAND']=AP[0]
  resAP['HAND_SHOULDER']=AP[1]
  resAP['SHOULDER_SHOULDER']=AP[2]
  resAP['HAND_TORSO']=AP[3]
  resAP['HAND_ELBOW']=AP[4]
  resAP['ELBOW_SHOULDER']=AP[5]
  resAP['mAP']=np.mean(AP)

  return resAP