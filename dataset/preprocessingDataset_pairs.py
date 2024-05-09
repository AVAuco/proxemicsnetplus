#Script that get the labels_6classes.mat ([Imagename, label, h, w]) and create labels_6classes_pair.json for a dataset per pairs. -- New version doctorado
# Dictionary : (imagename, coordinates[po:[], p1:[], p2:[],...(224x224)], labels[pop1:[], p0p2:[],...])
# Use : 
#       - labels.mat :  for getting the coordinates of each person
#       - labels_6classes.mat :  for getting [Imagename, label, h, w]
#
# Output : labels_6classes_pair.json 
#
# EXECUTE : python3 preprocessingDataset_pairs.py ..datasetDIR /pub/experiments/isajim/proxemics/dataset/

import numpy as np
import scipy.io
import cv2
import os
import argparse
import sys
import glob
import json




def parse_args():
    parser = argparse.ArgumentParser(description='Preprocessing script.')

    #PATHS
    parser.add_argument('--datasetDIR',type=str,  help='Main Dir where dataset is located', required=True)
    return parser.parse_args()


if __name__ == '__main__':

    # 1. Parameters read from console
    args = parse_args()
    datasetDir=args.datasetDIR 

    #------------------------------------------------------------------------------------------------------
    # 2. PRE: We check that the labels and images have been preprocessed.
    
    #a) Check Labels 
    labelFile_Original=os.path.join(datasetDir,'labels.mat') # For getting the coordinates
    # The file containing the oroginal labels does not exist. we create it
    if os.path.isfile(labelFile_Original)==False:
        print(' >> FILE labels.mat does not exist. ERROR')
        exit()
  
    labelFile_6classes=os.path.join(datasetDir,'labels_6classes.mat')                     
    # The file containing the new labels does not exist. we create it
    if os.path.isfile(labelFile_6classes)==False:
        print(' >> FILE labels_6classes.mat does not exist. Preprocess Labels')
        exit()
    else:
        print('Preprocessed labels!!')
        print()
    
    #b) Check Images
    #Check operative system to load image correctly
    my_os=sys.platform
    print("System OS : ",my_os,'\n')
    #Load the first image
    imgPathaux = os.path.join(datasetDir,'images/release/0028.jpg')
    if my_os =='linux':
        img = cv2.imread(imgPathaux)
    else:
        #Windows11
        imgPathaux = imgPathaux.replace('/','\\')
        imgAux = cv2.imread(imgPathaux) 
    
    h,w,c=imgAux.shape #height, width, channel
    
    if h!=224 or w!=224:
        print(' >> The images are not 224x224. Preprocess Images')
        exit()
    else:
        print('Preprocessed images!!')
        print()
    #------------------------------------------------------------------------------------------------------

 
    # 3. New labels file -- JSON
    proxemics={} 

    labels_newPair=os.path.join(datasetDir,'labels_6classes_pair.json')

    # 4. Check if the JSON file already exists. 
    if os.path.isfile(labels_newPair)==True:
        print(' >> FILE labels_6classes_pair.json exists.')
        decision = int(input(" 0 - Nuevo // 1 - Comenzar desde --> // 2 - Salir "))
        print()
        # 2. We initialize variables
        # Dictionary where we will store the key : Proxemics 
        # This key will have a list of elements 
        #   where each element will be of the form : [ImageName, label, h, w]
        inicioImg=0
        if decision == 1:
            with open(labels_newPair, 'r') as fp:
                proxemics = json.load(fp)
            inicioImg=int(input("Dígame número de la img por la que continuar "))
            inicioImg=inicioImg-1
            
            #print(proxemics['0003.jpg'])

        if decision == 2:
            exit()

    # 5. Load the mat files
    labels_Original = scipy.io.loadmat(labelFile_Original) 
    labels_6classes = scipy.io.loadmat(labelFile_6classes) 
    
    # 6. Images PATH : to know the number of images to be processed
    release_pair_dirpath=os.path.join(datasetDir,'images/release/')
    
    imgsList = os.listdir(release_pair_dirpath)

    #print(len(labels_Original['proxemics'][0][27][1][0]))
    #print(labels_Original['proxemics'][0][27][1][0][2][0][6])
    
    #,len(imgsList)
    # 7. Process each image one by one (589)
    for idx in range(inicioImg,len(imgsList)):
        imgName=imgsList[idx]

        print(' - IMG Path: ',imgName)
        #print('     > ORIGINAL LABEL: ', labels_Original['proxemics'][0][idx])
        #print('     > ORIGINAL LABEL: ',labels_6classes['Proxemics'][idx][1])
        print()

        #Dictionary
        proxemics[imgName]={'coordinates':{} , 'proxemics' : {} }

        #width and heigt to the image to be processed
        h=labels_6classes['Proxemics'][idx][2][0][0]
        w=labels_6classes['Proxemics'][idx][3][0][0]
        #print(h,w)
        
        #  Calculate the padding needed to make our image square
        padding=int(abs(h-w)/2) 
        # If the division is not exact, we increase padding1 by 1 
        if abs(h-w)%2 != 0:
            padding=padding+1

        # For each person in the image --> we are saving its corresponding coordinates (rescaled to 224x224)
        for p in range(0, len(labels_Original['proxemics'][0][idx][1][0])):
            person='p'+str(p)
            
            #print(labels_Original['proxemics'][0][idx][1][0][p][0])
            kps_persona=[]
            for kp in range(0,10):
                if h>w:
                    kps_persona.append([(labels_Original['proxemics'][0][idx][1][0][p][0][kp][0]+padding)*224/h, labels_Original['proxemics'][0][idx][1][0][p][0][kp][1]*224/h ])
                else:
                    kps_persona.append([labels_Original['proxemics'][0][idx][1][0][p][0][kp][0]*224/w, (labels_Original['proxemics'][0][idx][1][0][p][0][kp][1]+padding)*224/w ])


            proxemics[imgName]['coordinates'][person]=kps_persona
            #print(proxemics[imgName]['coordinates'][person])

        
        # For each pair of people in the image, we ask the user to enter the correct label  -- Save the correct label per pairs
        # We help the user with a possible correct label obtained from the labelFile_6classes
        for p1 in range (0, len(labels_Original['proxemics'][0][idx][1][0])-1):
            for p2 in range(p1+1, len(labels_Original['proxemics'][0][idx][1][0])):

                sampleLabels=[0,0,0,0,0,0]

                pair='p'+str(p1)+'-p'+str(p2)

                print('   --> Pareja : ', pair)
                print('     > ORIGINAL LABEL: ',labels_6classes['Proxemics'][idx][1])
                print()

                decision = int(input("          ** 0 - cambiar label // 1 - dejar label --> ** "))
                print()
                #print('Su decisión: ',decision)

                if decision == 0:
                    sampleLabels[0] = int(input("            -- Hand-Hand : 0 // 1 --> "))
                    sampleLabels[1] = int(input("            -- Hand-Shoulder : 0 // 1 --> "))
                    sampleLabels[2] = int(input("            -- Shoulder-Shoulder : 0 // 1 --> "))
                    sampleLabels[3] = int(input("            -- Hand-Torso : 0 // 1 --> "))
                    sampleLabels[4] = int(input("            -- Hand-Elbow : 0 // 1 --> "))
                    sampleLabels[5] = int(input("            -- Elbow-Shoulder : 0 // 1 -->"))
                elif decision==1:
                    sampleLabels=labels_6classes['Proxemics'][idx][1][0].tolist()
                else:
                    # In case we want to leave without finishing --> save json
                    del proxemics[imgName]
                    
                    with open(labels_newPair, 'w') as fp:
                        json.dump(proxemics, fp,indent=4) 

                    exit()
                
                print()
                print(' >> New label : ', sampleLabels)
                print()


                proxemics[imgName]['proxemics'][pair]=sampleLabels
                
    
                
                
                
    #print(proxemics)


      
    # Save final json
    with open(labels_newPair, 'w') as fp:
        json.dump(proxemics, fp,indent=4)


    print("DONE!")
