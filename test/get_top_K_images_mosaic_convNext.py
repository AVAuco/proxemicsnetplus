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

sys.path.append("..")
sys.path.insert(0, '/pub/experiments/isajim/proxemics/code/test')
sys.path.insert(0, '/opt/data/isa/proxemics/code/test')

from train.model import get_basemodel_convNext
from test import evaluateAP
from train.load_dataset_train_test import  get_partition_convNext
from train.datagenerator import *

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


def parse_args():
    parser = argparse.ArgumentParser(description='Training and testing script.')

   
    parser.add_argument('--k',type=int,  help='top K images', required=True, default=5)
    parser.add_argument('--classIdx', type=int,  help='Class to show', required=True, default=0)
    parser.add_argument('--order', type=float,  help='0-from highest to lowest//1-from lowest to highest', required=False, default=0)
    #PATHS
    parser.add_argument('--datasetDIR',type=str,  help='Main Dir where dataset is located', required=True)
    parser.add_argument('--model',type=str,  help='best_keras_model tf', required=True)
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
    predClass.append([X_test[2][row],y_pred_test[row][classIdx],y_test[row][classIdx]])
  
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
    image=predClass[i][0] + 0.5 
    image=image*255 
    '''
    image = cv2.putText(image,'Pred : '+str(predClass[i][1]), (1, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv2.LINE_AA)
    if int(predClass[i][2])==0:
      image = cv2.putText(image,'Label : '+str(int(predClass[i][2])), (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2, cv2.LINE_AA)
    else:
      image = cv2.putText(image,'Label : '+str(int(predClass[i][2])), (1, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2, cv2.LINE_AA)
    #image=image/255.0
    '''


    imgFinal = cv2.cvtColor(np.float32(image), cv2.COLOR_BGR2RGB)
    images.append(imgFinal)
    

  # 8. Plot images in the grid
  plt.figure(figsize=[20,k]) 
  plt.suptitle(className,fontsize=30)
  num_cols = 5
  num_rows = math.ceil(k / num_cols)
  for i in range(0, k):
    plt.subplot(num_rows, num_cols, i+1) 
    plt.axis('off')
    #plt.imshow(images[i])
    plt.imshow(images[i].astype(np.uint8))
    
  
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

    zipPath=os.path.join(datasetDir,'images/recortes.zip')

    
    # 1. We get the set that was used for training the model
    useSet=1
    if model.find("set2")!=-1:   #found set2
        useSet=2

    #We load the h5 annotations that were created when the TARs were generated.
    print('* Loading data')
    import json
    jsonfile=os.path.join(datasetDir,'labels_6classes_pair.json')
    with open(jsonfile) as file:
        dataset = json.load(file)

    #The information of all samples is stored in the ''all samples'' list.
    allSamples=[]
    imagenamelist=sorted(list(dataset.keys()))
    for image in imagenamelist:
      for pair in dataset[image]['proxemics'].keys():
        #print(image, pair)      # image ='0001.jpg'  / pair='p0-p1'
        
        label= dataset[image]['proxemics'][pair]
        p0=pair[0:2]
        p1=pair[3:]
        muestra=[image[:-4],p0,p1,label]
        #print(muestra)
        allSamples.append(muestra)


    npairs = len(allSamples)

    #We generate the data sets according to the division proposed by the authors
    trainImg,valImg,testImg=get_partition_convNext(useSet)

    #train samples
    trainIdx = []
    valIdx = []
    testIdx = []

    for idx in range(0,len(allSamples)):
      imgname = int(allSamples[idx][0])
      if imgname in trainImg:
        trainIdx.append(idx)
      elif imgname in valImg:
        valIdx.append(idx)
      else:
        testIdx.append(idx)

    partition = {}
    partition['test'] = testIdx #range(nsamples_train, total_samples)
        


    #Datagenerator Parameters
    params = {
              'batch_size': len(partition['test']),
              'shuffle': False,
              'augmentation': False,
              'zipPath' : zipPath,
              'onlyPair': False,
              'isTest': True
              }
    
    

    test_generator =DataGenerator(partition['test'], allSamples, **params)

    X_test, y_test= test_generator.__getitem__(0)

  
    model = keras.models.load_model(os.path.join(model,'checkpoint')) # rollback to best model
    

    print()
    print(model.summary())

    # 6. get mosaic
    get_top_K_images_mosaic(outdir,model,X_test,y_test,k,classIdx,order)   # 0-mayor a menor // 1-menor a mayor

    print("DONE!")


