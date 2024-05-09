# ====================================================================
# MAIN : Get a mosaic with the top K images of a selected class
# ====================================================================

import numpy as np

import argparse
import os
import sys
from operator import itemgetter
import cv2
import deepdish
import matplotlib.pyplot as plt
import math
sys.path.append("..")
from train.load_dataset_train_test import  get_dataset,get_train_test_partition_author






def parse_args():
    parser = argparse.ArgumentParser(description='Training and testing script.')

   
    parser.add_argument('--k',type=int,  help='top K images', required=True, default=5)
    parser.add_argument('--classIdx', type=int,  help='Class to show', required=True, default=0)
    parser.add_argument('--order', type=float,  help='0-from highest to lowest//1-from lowest to highest', required=False, default=0)
    #PATHS
    parser.add_argument('--datasetDIR',type=str,  help='Main Dir where dataset is located', required=True)
    parser.add_argument('--model',type=str,  help='best_keras_model.h5', required=True)
    parser.add_argument('--outdir',type=str,  help='Main Dir where mosaic will be stored', required=True)

    return parser.parse_args()

    
def get_top_K_images_mosaic(outdir,model,X_test,y_test,k,classIdx,order):
  #1. Predict X_test
  y_pred_test=model.predict(X_test)

  #2. List to save Ap results for each class
  AP=[]

  #3. List to store the values of y_pred and y_test of the class to evaluate
  predClass=[]
  
  #4. For each sample we take the y_pred and y_test of the column corresponding to that class.
  for row in range(0,len(y_pred_test)):
    predClass.append([X_test[row],y_pred_test[row][classIdx],y_test[row][classIdx]])
  
  #5. We order according to the value of the predictions of that class.
  if order==0: #from highest to lowest
    predClass=sorted(predClass, key=itemgetter(1),reverse=True)
    plotName='plot_top_'+str(k)+'_highest_images_class'+str(classIdx)+'_set_'+str(useSet)+'.jpg'
  else: #from lowest to highest
    predClass=sorted(predClass, key=itemgetter(1),reverse=False)
    plotName='plot_top_'+str(k)+'_lowest_images_class'+str(classIdx)+'_set_'+str(useSet)+'.jpg'

  #6. Prepare class name
  classNames=['HAND-HAND','HAND - SHOULDER','SHOULDER - SHOULDER','HAND - TORSO','HAND - ELBOW','ELBOW - SHOULDER']
  className='Class  '+ str(classIdx)+' : '+classNames[classIdx]

  #7. We write on the images their corresponding predictions 
  images = []
  for i in range(0, k):
    image=predClass[i][0]*255
    image = cv2.putText(image,'Pred : '+str(predClass[i][1]), (1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    if int(predClass[i][2])==0:
      image = cv2.putText(image,'Label : '+str(int(predClass[i][2])), (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    else:
      image = cv2.putText(image,'Label : '+str(int(predClass[i][2])), (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    image=image/255.0

    imgFinal = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    images.append(imgFinal)

  # 8. Plot images in the grid
  plt.figure(figsize=[20,k]) 
  plt.suptitle(className,fontsize=30)
  num_cols = 5
  num_rows = math.ceil(k / num_cols)
  for i in range(0, k):
    plt.subplot(num_rows, num_cols, i+1) 
    plt.axis('off')
    plt.imshow(images[i])
  
  #9. Save the mosaic
  plotName=os.path.join(outdir,plotName)
  plt.savefig(plotName)

#====================================================================================


if __name__ == '__main__':
    #Parameters read from console
    args = parse_args()
    #print(args)

    # read parameters
    k = args.k
    classIdx = args.classIdx
    order = args.order

    datasetDir=args.datasetDIR
    model=args.model
    outdir=args.outdir
    
    # 1. We get the set that was used for training the model
    useSet=1
    if model.find("set2")!=-1:   #found set2
        useSet=2

    # 2. Load all dataset
    labelFile=os.path.join(datasetDir,'labels_6classes.mat')
   
    X, y = get_dataset(labelFile,datasetDir)

    # 3. We separate the samples equally into Train and Test. 
    X_train, y_train, X_test, y_test = get_train_test_partition_author(X,y,useSet)


    # 4. Normalize from [0-255] to [0-1]
    X_test =X_test.astype('float32')
    X_test /= 255.0

    # 5. Get the model
    # Load model
    import keras
    model = keras.models.load_model(model) 
    
      
    print()
    print(model.summary())

    # 6. get mosaic
    get_top_K_images_mosaic(outdir,model,X_test,y_test,k,classIdx,order)   # 0-mayor a menor // 1-menor a mayor

    print("DONE!")


