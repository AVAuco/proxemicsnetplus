# ====================================================================
# MAIN : Train and test a full MODEL (RGB / POSE / RGB-POSE) with ConvNext (Proxemics - PISC) - PerPAIR
# ====================================================================
# TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  1 --lr 0.00004 --pose --rgb --onlyPairPose --onlyPairRGB --datasetName pisc --crossAttention
# TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  1 --lr 0.00004 --pose --rgb --onlyPairPose --onlyPairRGB --datasetName proxemics 
 

from audioop import cross
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

from train.model import get_basemodel_convNext, get_basemodel_convNext_crossAttention
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
    parser.add_argument('--rgb',action='store_true',help='full RGB  model',default=False)
    parser.add_argument('--pose',action='store_true',help=' full pose model',default=False)
    parser.add_argument('--onlyPairRGB',action='store_true',help='Only context brach of RGB model',default=False)
    parser.add_argument('--onlyPairPose',action='store_true',help='Only context brach of Pose model',default=False)
    parser.add_argument('--nlayersFreeze', type=int,  help='n layers frozen', choices=[0,2,4,6,8,10], required=False, default=0)

    parser.add_argument('--crossAttention',action='store_true',help='Use Coss attention fusion',default=False)
    
    #not implemented
    parser.add_argument('--contrastiveloss',action='store_true',help='Use a contrastive loss',default=False)
    parser.add_argument('--beta', type=float,  help='beta ContrastiveLoss', required=False, default=0.0)
    parser.add_argument('--features',action='store_true',help='Use features branch (only with crossAttention)',default=False)
    
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

    rgbmodel=args.rgb
    posemodel=args.pose
    if rgbmodel ==False and posemodel==False:
      print('You must indicate rgb, pose or both')
      exit()
    onlyPairRGB=args.onlyPairRGB
    onlyPairPose=args.onlyPairPose

    nlayersFreeze=args.nlayersFreeze

    datasetName=args.datasetName

    use_crossAttention=args.crossAttention

    use_contrastiveloss=args.contrastiveloss
    beta=args.beta
    features=args.features
    featuresPath="/opt/data/isajim/experiments/proxemics/dataset/features_pair"
    featuresTextFilePath=os.path.join(featuresPath,"pair_features_text_compressed.npz")
    #print(features,"   :    ", featuresTextFilePath)
    
    datasetDir=args.datasetDIR
    outdir=args.outModelsDIR

    if datasetName=="proxemics":
      zipPath=os.path.join(datasetDir,'images/recortes.zip')
      posezipPath=os.path.join(datasetDir,'poseImg-I/poseImg_I_recortes_thr0_pair.zip')
    else:
      zipPath=os.path.join(datasetDir,'recortes.zip')
      posezipPath=os.path.join(datasetDir,'poseImg-I/poseImg_I_recortes_pair.zip')
      #If it is pisc -- in the datagenerator you have to change the zip load of the images and in the model function loss to softmax.    
    
    
    # ===========================================
    # Model name
    # ===========================================
    if augmentation == True:
      aug=1
    else:
      aug=0

    typefusion='base'
    if use_crossAttention:
      typefusion='crossAttention'

    if use_contrastiveloss and ((use_crossAttention==False) or (rgbmodel==False or posemodel==False)): 
      print('You can only use contrastiveloss with RGB+Pose models and usecrossAttention')
      exit()

    typeImg='RGB'
    if posemodel:
      typeImg='Pose'
    if rgbmodel and posemodel:
      typeImg='RGB_Pose'

    if features:
      modelname = "Model_aug{:d}_bs{:d}_set{:d}_lr{:1.5f}_o{}_fText".format(aug,batchsize, useSet,lr, optimizer)
    else:
      modelname = "Model_aug{:d}_bs{:d}_set{:d}_lr{:1.5f}_o{}_fr{:d}".format(aug,batchsize, useSet,lr, optimizer,nlayersFreeze)
    
    if typeImg == 'RGB_Pose':
      if onlyPairRGB and onlyPairPose:
        groupname='RGB_onlypair_'+modeltype+'_Pose_onlypair_'+modeltype
      elif onlyPairRGB :
        groupname='RGB_onlypair_'+modeltype+'_Pose_p0p1pair_'+modeltype
      elif onlyPairPose :
        groupname='RGB_p0p1pair_'+modeltype+'_Pose_onlypair_'+modeltype
      else:
        groupname='RGB_p0p1pair_'+modeltype+'_Pose_p0p1pair_'+modeltype
    else:
      if onlyPairRGB or onlyPairPose:
        groupname=typeImg+'_onlypair_'+modeltype
      else:
        groupname=typeImg+'_p0p1pair_'+modeltype


    print(groupname)

    if use_contrastiveloss:
       model_filepath = os.path.join(outdir,typefusion,'contrastiveloss',typeImg, groupname, modelname)
    else:
      model_filepath = os.path.join(outdir,typefusion,typeImg, groupname, modelname)
        
    #Directory where the model will be saved
    print("* The results will be saved to: "+model_filepath+"\n")
    sys.stdout.flush()


    # ===========================================
    # WANDB Parameters
    # ===========================================
    if datasetName=="proxemics":
      if use_crossAttention:
        if use_contrastiveloss:
          projectname="proxemics-convNext_crossAttention_contrastiveloss"
        else:
          projectname="proxemics-convNext_crossAttention"
      else:
        projectname="proxemics-convNext"
    else:
      if use_crossAttention:
        projectname="pisc-convNext_crossAttention"
      else:
        projectname="pisc-convNext"
    
  
    id='id_'+ groupname+ '_'+ modelname
    
    wandb.init(project=projectname, group=groupname, name=modelname, id=id, config = {
      "learning_rate": lr,
      "epochs": nepochs,
      "batch_size": batchsize,
      "optimizer":optimizer,
      "set":useSet
    })    
    

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
        
    #gpus = tf.config.experimental.list_physical_devices('GPU')
    gpus=tf.config.list_physical_devices('GPU')


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
              'posezipPath' : posezipPath,
              'typeImg' : typeImg,
              'onlyPairRGB': onlyPairRGB,
              'onlyPairPose': onlyPairPose,
              'features': features,
              'featuresTextFilePath' : featuresTextFilePath
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
    # MODEL
    # ===========================================
    if os.path.exists(os.path.join(model_filepath,'checkpoint')):
      print("INFO: loaded best model so far.")
      model = keras.models.load_model(os.path.join(model_filepath,'checkpoint')) # rollback to best model
    else:
      pathlib.Path(model_filepath).mkdir(parents=True, exist_ok=True)
      # Get model
      #model_path = "gs://tfhub-modules/sayakpaul/convnext_base_21k_1k_224/1/uncompressed"
      if modeltype == "base":
        model_path="../../premodels/convnext_base_21k_1k_224_fe_1/"
      elif modeltype == "large":
        model_path="../../premodels/convnext_large_21k_1k_224_1/"
      elif modeltype == "xlarge":
        model_path="../../premodels/convnext_xlarge_21k_1k_224_1/"
      else:
        print('Error : No existe modelo')

      print("Modelpath: ", model_path)
      if use_crossAttention:
        model=get_basemodel_convNext_crossAttention(model_path, lr,optimizer,typeImg, onlyPairRGB, onlyPairPose,use_contrastiveloss,beta,features)
      else:
        model=get_basemodel_convNext(model_path, lr,optimizer,typeImg, onlyPairRGB, onlyPairPose,nlayersFreeze)
      
      print(model.summary())


    # ===========================================
    # TRAINING
    # ===========================================
    print("*** Starting training ***")
    metric='val_auc'
    checkpoint_cb = keras.callbacks.ModelCheckpoint(os.path.join(model_filepath,'checkpoint'), monitor=metric, save_format = "tf",verbose=1, save_best_only=True, save_freq="epoch", mode='max')
    #early_stopping_cb = keras.callbacks.EarlyStopping(monitor=metric, patience=8, verbose=1, min_delta=1e-4,restore_best_weights=True)
    reduce_lr_cb = keras.callbacks.ReduceLROnPlateau(monitor=metric, factor=0.1,patience=4, verbose=1, min_delta=1e-4)
    #callbacks = [checkpoint_cb, early_stopping_cb, reduce_lr_cb]
    #callbacks = [ early_stopping_cb, reduce_lr_cb,WandbCallback()]
    #callbacks = [ checkpoint_cb, reduce_lr_cb]
    #tf.debugging.experimental.enable_dump_debug_info(log_dir, tensor_debug_mode="NO_TENSOR", circular_buffer_size=-1)
    callbacks = [  reduce_lr_cb, checkpoint_cb, WandbCallback(save_model=False, monitor=metric)]

    # Go!
    nbatches = training_generator.__len__()
    history = model.fit(training_generator, epochs = nepochs, callbacks=callbacks, verbose=1, steps_per_epoch=nbatches, validation_data=validation_generator)
    nbatches = training_validation_generator.__len__()
    history = model.fit(training_validation_generator, epochs = 2, verbose=1, steps_per_epoch=nbatches) 
    
    print("*** End of training ***")
    
    # Save model
    model.save(model_filepath, save_format = "tf")



    # ===========================================
    # TEST - Result of obtained model
    # ===========================================
    #DataGenerator for test
    paramsTest = params
    paramsTest['shuffle'] = False
    paramsTest['augmentation'] = False
    paramsTest['isTest'] = True
    if datasetName=="proxemics":
      paramsTest['batch_size'] = len(partition['test'])
      test_generator =DataGenerator(partition['test'], allSamples, **paramsTest)
      X_test, y_test= test_generator.__getitem__(0)

      AP=evaluateAP(model,X_test,y_test)
    else:  #PISC
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

        pred= model.predict(X_test)

        y_pred_img.append(pred[0].tolist())
        y_test_img.append(y_test[0].tolist())
        
        # Actualizar el índice
        indice += tamano_grupo

      AP=evaluateAP(model,y_pred_img,y_test_img,False)

    #Print AP results
    print()

    print("- AP results:")
    print(' HAND - HAND : ', AP['HAND_HAND'])
    print(' HAND - SHOULDER : ', AP['HAND_SHOULDER'])
    print(' SHOULDER - SHOULDER : ', AP['SHOULDER_SHOULDER'])
    print(' HAND - TORSO : ', AP['HAND_TORSO'])
    print(' HAND - ELBOW : ',AP['HAND_ELBOW'])
    print(' ELBOW - SHOULDER : ', AP['ELBOW_SHOULDER'])

    print()
    print("- mAP : " , AP['mAP'])

    # Save results

    testFile = os.path.join(model_filepath,'best_keras_model_results.h5')
    dd.io.save(testFile, AP)
    
    wandb.log({"AP": AP})






