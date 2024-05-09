# ====================================================================
# MAIN : Train and test a BASE MODEL (RGB) with ConvNext (Proxemics - PISC) - PerIMAGE
# ====================================================================
import numpy as np
import argparse
import os
import sys
import tensorflow as tf
import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping
import deepdish as dd
import pathlib 

import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


sys.path.append("..")
sys.path.insert(0, '../test')
sys.path.insert(0, '../test')

from train.model import get_basemodel_convNext
from test import evaluateAP
from train.load_dataset_train_test import  get_partition_convNext
from train.datagenerator import *

import wandb
from wandb.keras import WandbCallback


def parse_args():
    parser = argparse.ArgumentParser(description='Training and testing script.')

   
    parser.add_argument('--b',type=int,  help='Size of each batch', required=False, default=6)
    parser.add_argument('--e', type=int,  help='Number of epochs', required=False, default=25)
    parser.add_argument('--lr', type=float,  help='lrate', required=False, default=0.01)
    parser.add_argument('--o',type=str,  help='optimizer', required=False, default="Adam")
    parser.add_argument('--g', type=float,  help='GPU rate', required=False, default=0.9)
    parser.add_argument('--set', type=int,  help='Set (1 or 2)', required=False, default=1)
    parser.add_argument("--modeltype", help="pre trained model - convnext", choices=['base', 'large', 'xlarge'], required=True)
    parser.add_argument('--onlyPairRGB',action='store_true',help='Only context brach',default=False)
    parser.add_argument('--nlayersFreeze', type=int,  help='n layers frozen', choices=[0,2,4,6,8,10], required=False, default=0)

    parser.add_argument('--datasetName',type=str,  help='Dataset', required=False, default="proxemics")

    #PATHS
    parser.add_argument('--datasetDIR',type=str,  help='Main Dir where dataset is located', required=True)
    parser.add_argument('--outModelsDIR',type=str,  help='Dir where model will be saved', required=True)

    return parser.parse_args()



#====================================================================================

if __name__ == '__main__':

    #Parameters read from console
    args = parse_args()
    # ===========================================
    # Hyperparameters
    # ===========================================
    verbose=1
    batchsize = args.b
    nepochs = args.e
    gpuRate = args.g
    lr = args.lr
    optimizer = args.o
    useSet = args.set

    augmentation=True

    modeltype=args.modeltype
    onlyPairRGB=args.onlyPairRGB
    nlayersFreeze=args.nlayersFreeze

    datasetName=args.datasetName

    datasetDir=args.datasetDIR
    outdir=args.outModelsDIR

    if datasetName=="proxemics":
      zipPath=os.path.join(datasetDir,'images/recortes.zip')
    else:
      zipPath=os.path.join(datasetDir,'recortes.zip')


    # ===========================================
    # Model name
    # ===========================================
    if augmentation == True:
      aug=1
    else:
      aug=0

    typeImg='RGB'

    modelname = "Model_aug{:d}_bs{:d}_set{:d}_lr{:1.5f}_o{}_fr{:d}".format(aug,batchsize, useSet,lr, optimizer,nlayersFreeze)
    if onlyPairRGB:
      groupname=typeImg+'_onlypair_'+modeltype
    else:
      groupname=typeImg+'_p0p1pair_'+modeltype

    print(groupname)

    model_filepath = os.path.join(outdir, groupname, modelname)

    #Directory where the model will be saved
    print("* The results will be saved to: "+model_filepath+"\n")
    sys.stdout.flush()


    # ===========================================
    # WANDB Parameters
    # ===========================================
    if datasetName=="proxemics":
      projectname="proxemics-convNext"
    else:
      projectname="pisc-convNext"
    
    id='id_'+ groupname+ '_'+ modelname
    
    api = wandb.Api()

    # Access attributes directly from the run object 
    # or from the W&B App 
    username = "proxemicstfm"
    project = projectname
    run_id = id

    run = api.run(f"{username}/{project}/{run_id}")
    
    

    # ===========================================
    # USE GPU
    # ===========================================
    #CHANGE ME
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    tf.executing_eagerly()
    graph = tf.Graph()
    graph.as_default()
    session = tf.compat.v1.Session(graph=graph, config=config)
    session.as_default()

    
    # ===========================================
    # Dataset
    # ===========================================
    print('* Loading data')
    import json
    if datasetName=="proxemics":
      jsonfile=os.path.join(datasetDir,'labels_6classes_pair.json')
    else:
      jsonfile=os.path.join(datasetDir,'labels_6classes_relationship.json')

    with open(jsonfile) as file:
      dataset = json.load(file)

    #The information of all samples is stored in the ''all samples'' list.
    allSamples=[]
    imagenamelist=sorted(list(dataset.keys()))
    if datasetName=="proxemics":
      labeldict="proxemics"
    else:
      labeldict="labels"

    for image in imagenamelist:
      for pair in dataset[image][labeldict].keys():
        #print(image, pair)      # image ='0001.jpg'  / pair='p0-p1'
        
        label= dataset[image][labeldict][pair]
        p0=pair.split('-')[0]
        p1=pair.split('-')[1]
        muestra=[image[:-4],p0,p1,label]
        allSamples.append(muestra)

    npairs = len(allSamples)
    
    #We generate the data sets according to the division proposed by the authors
    if datasetName=="proxemics":
      trainImg,valImg,testImg=get_partition_convNext(useSet)
    else:
      jsonfiletrain=os.path.join(datasetDir,'relationship_split/relation_trainidx.json')
      with open(jsonfiletrain) as trainfile:
        trainImg = json.load(trainfile)
      
      jsonfileval=os.path.join(datasetDir,'relationship_split/relation_validx.json')
      with open(jsonfileval) as valfile:
        valImg = json.load(valfile)

      jsonfiletest=os.path.join(datasetDir,'relationship_split/relation_testidx.json')
      with open(jsonfiletest) as testfile:
        testImg = json.load(testfile)     


    #train samples
    trainIdx = []
    valIdx = []
    testIdx = []

    for idx in range(0,len(allSamples)):
      if datasetName=="proxemics":
        imgname = int(allSamples[idx][0])
      else:
        imgname = allSamples[idx][0]

      if imgname in trainImg:
        trainIdx.append(idx)
      elif imgname in valImg:
        valIdx.append(idx)
      else:
        testIdx.append(idx)

    # set partitions
    partition = {}
    partition['train'] = trainIdx #range(0,nsamples_train)       # IDs
    partition['validation'] = valIdx #range(nsamples_train, total_samples)
    partition['test'] = testIdx #range(nsamples_train, total_samples)
        

    # ===========================================
    #DataGenerators
    # ===========================================
    #Datagenerator Parameters
    params = {
              'batch_size': batchsize,
              'shuffle': True,
              'augmentation': augmentation,
              'zipPath' : zipPath,
              'typeImg' : "RGB",
              'onlyPairRGB': onlyPairRGB,
              'onlyPairPose': False
              }
    #DataGenerator for training
    training_generator = DataGenerator( partition['train'], allSamples, **params)
    partition['train_val'] = partition['train'] + partition['validation']
    training_validation_generator = DataGenerator( partition['train_val'], allSamples, **params)
    #DataGenerator for validation
    paramsVal= params
    paramsVal['shuffle'] = False
    paramsVal['augmentation'] = False
    paramsVal['isTest'] = True
    validation_generator =DataGenerator (partition['validation'], allSamples, **paramsVal)

    print("\n- Data generators are ready!\n")

    sys.stdout.flush()


    # ===========================================
    # TRAINING
    # ===========================================
    print("*** Starting training ***")
    metric='val_auc'
    checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_filepath,'checkpoint'), monitor=metric, save_format = "tf",verbose=1, save_best_only=True, save_freq="epoch", mode='max')
    #early_stopping_cb = keras.callbacks.EarlyStopping(monitor=metric, patience=8, verbose=1, min_delta=1e-4,restore_best_weights=True)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor=metric, factor=0.1,patience=4, verbose=1, min_delta=1e-4)
    
    print('Checkpoint training')
    model_chkt = keras.models.load_model(os.path.join(model_filepath,'checkpoint')) # rollback to best model
    nbatches = training_validation_generator.__len__()
    # 2 more epochs
    history = model_chkt.fit(training_validation_generator, epochs = 2, verbose=1, steps_per_epoch=nbatches) 
    model_chkt.save(os.path.join(model_filepath,'checkpoint'), save_format = "tf")

    print("*** End of training ***")
   

    # ===========================================
    # TEST
    # ===========================================
    ###################################### Evaluate model - PERPAIR (CHKT model)
    paramsTest = params
    paramsTest['shuffle'] = False
    paramsTest['augmentation'] = False
    paramsTest['isTest'] = True
    
    if datasetName=="proxemics":
      paramsTest['batch_size'] = len(partition['test'])
      test_generator =DataGenerator(partition['test'], allSamples, **paramsTest)
      X_test, y_test= test_generator.__getitem__(0)
      AP=evaluateAP(model_chkt,X_test,y_test)
    else:
      testIdx=[]
      y_pred_img=[]
      y_test_img=[]
      # Iterar mientras haya elementos en la lista
      indice=0
      tamano_grupo=25
      while indice < len(testImg):
        print(indice, len(testImg))
        # Tomar el grupo de elementos
        if (len(testImg)-indice)< tamano_grupo:
          testImg_group = testImg[indice:len(testImg)]
        else:
          testImg_group = testImg[indice:indice+tamano_grupo]
        
        testIdx=[]
        for img in testImg_group:
          for pix in range(0,len(allSamples)):				#Cogemos todas las muestras de ese video
            if  int(allSamples[pix][0]) == int(img):
              testIdx.append(pix)

        # Datasets
        partition = {}
        partition['test'] = testIdx
        paramsTest['batch_size'] = len(testIdx)

        test_generator = DataGenerator( partition['test'], allSamples, **paramsTest)
        X_test, y_test= test_generator.__getitem__(0)

        pred= model_chkt.predict(X_test)

        y_pred_img.append(pred[0].tolist())
        y_test_img.append(y_test[0].tolist())
      
        
        # Actualizar el índice
        indice += tamano_grupo

      AP=evaluateAP(model_chkt,y_pred_img,y_test_img,False)

    #Print AP results
    print("- AP results CHECKPOINTS:")
    print(' HAND - HAND : ', AP['HAND_HAND'])
    print(' HAND - SHOULDER : ', AP['HAND_SHOULDER'])
    print(' SHOULDER - SHOULDER : ', AP['SHOULDER_SHOULDER'])
    print(' HAND - TORSO : ', AP['HAND_TORSO'])
    print(' HAND - ELBOW : ',AP['HAND_ELBOW'])
    print(' ELBOW - SHOULDER : ', AP['ELBOW_SHOULDER'])

    print("- mAP : " , AP['mAP'])

    testFile = os.path.join(model_filepath,'best_keras_model_results_chkt.h5')
    dd.io.save(testFile, AP)

    run.summary["AP_chkt.HAND_HAND"] = AP['HAND_HAND']
    run.summary["AP_chkt.HAND_SHOULDER"] = AP['HAND_SHOULDER']
    run.summary["AP_chkt.SHOULDER_SHOULDER"] = AP['SHOULDER_SHOULDER']
    run.summary["AP_chkt.HAND_TORSO"] = AP['HAND_TORSO']
    run.summary["AP_chkt.HAND_ELBOW"] = AP['HAND_ELBOW']
    run.summary["AP_chkt.ELBOW_SHOULDER"] = AP['ELBOW_SHOULDER']
    run.summary["AP_chkt.mAP"] = AP['mAP']
    run.update()

    
    ###################################### Evaluate model - perIMAGE
    testIdx=[]
    y_pred_img=[]
    y_test_img=[]
    for img in testImg:
      testIdx=[]
      for pix in range(0,len(allSamples)):				#Cogemos todas las muestras de ese video
        if  int(allSamples[pix][0]) == int(img):
          testIdx.append(pix)

      # Datasets
      partition = {}
      partition['test'] = testIdx
      paramsTest['batch_size'] = len(testIdx)

    
      test_generator = DataGenerator( partition['test'], allSamples, **paramsTest)
      X_test, y_test= test_generator.__getitem__(0)
      pred= model_chkt.predict(X_test)

      if len(pred) > 1:
        #print(pred)
        pred_per_img=pred.max(axis=0)
        test_per_img=y_test.max(axis=0)
        y_pred_img.append(pred_per_img.tolist())
        y_test_img.append(test_per_img.tolist())

      else:
        y_pred_img.append(pred[0].tolist())
        y_test_img.append(y_test[0].tolist())

    AP=evaluateAP(model_chkt,y_pred_img,y_test_img,False)

    #Print AP results
    print("- AP results CHECKPOINTS per Image:")
    print(' HAND - HAND : ', AP['HAND_HAND'])
    print(' HAND - SHOULDER : ', AP['HAND_SHOULDER'])
    print(' SHOULDER - SHOULDER : ', AP['SHOULDER_SHOULDER'])
    print(' HAND - TORSO : ', AP['HAND_TORSO'])
    print(' HAND - ELBOW : ',AP['HAND_ELBOW'])
    print(' ELBOW - SHOULDER : ', AP['ELBOW_SHOULDER'])

    print("- mAP : " , AP['mAP'])

    testFile = os.path.join(model_filepath,'best_keras_model_results_chkt_perImage.h5')
    dd.io.save(testFile, AP)

    run.summary["AP_chkt_perImage.HAND_HAND"] = AP['HAND_HAND']
    run.summary["AP_chkt_perImage.HAND_SHOULDER"] = AP['HAND_SHOULDER']
    run.summary["AP_chkt_perImage.SHOULDER_SHOULDER"] = AP['SHOULDER_SHOULDER']
    run.summary["AP_chkt_perImage.HAND_TORSO"] = AP['HAND_TORSO']
    run.summary["AP_chkt_perImage.HAND_ELBOW"] = AP['HAND_ELBOW']
    run.summary["AP_chkt_perImage.ELBOW_SHOULDER"] = AP['ELBOW_SHOULDER']
    run.summary["AP_chkt_perImage.mAP"] = AP['mAP']
    run.update()
    

    wandb.finish()
    