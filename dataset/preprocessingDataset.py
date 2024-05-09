#Script containing the necessary functions to preprocess the images (224x224) and the labels -- First Version - no pairs (master)
# Dataset : wget -q https://www.dropbox.com/s/5zarkyny7ywc2fv/PROXEMICS.zip
# FUNCTIONS:
# -label_preprocessing
# -img_preprocessing

import numpy as np
import scipy.io
import cv2
import os



# Labels pre-processingWe modify the labels of the labels.mat file.
# For each image, we transform its corresponding vector of Proxemics,
#  which contains 20 classes (including subclasses), into only 6 classes.
# New results : labels_6classes.mat
def label_preprocessing(originalLabelsPath,newLabelsPath ):
    
    mat = scipy.io.loadmat(originalLabelsPath)  #'labels.mat'


    # 1. We calculate the total number of samples to process
    # There are 1178 labels because it includes the release and fipped labels.
    imglen=len(mat['proxemics'][0])

    # 2. We initialize variables
    # Dictionary where we will store the key : Proxemics 
    # This key will have a list of elements 
    #   where each element will be of the form : [ImageName, label]
    proxemics={}
    # Lista de elementos [ImageName, label]
    labels=[]

    # 3. We run through each sample of the labels.mat file.
    for i in range(0,imglen):
      #print('Image :', mat['proxemics'][0][i][0])

      # 3.1. Vector proxemics of that image
      ProxemicVector=mat['proxemics'][0][i][3]
      # 3.2. We calculate the number of people in the image to go through one by one.
      nPeople=len(ProxemicVector)
      # 3.3. List to count the 6 different kinds that can appear in each image 
      sampleLabels=[0,0,0,0,0,0]

      for p1 in range(0,nPeople):         # We access the set of vectors for each person
        for p2 in range(0,nPeople):       # We access the individual vectors of each set
          for elem in range(0,20):        # We access each element of the vector
            # Depending on the position of the 1 in the image that we are processing, 
            # we will set a 1 in the corresponding class  
            if( ProxemicVector[p1][p2][elem]==1):
             #HAND-HAND
             if(elem==0 or elem==1 or elem==2 or elem==3):
               sampleLabels[0]=1 
             #HAND-SHOULDER
             if(elem==4 or elem==5 or elem==6 or elem==7):
               sampleLabels[1]=1
             #SHOULDER-SHOULDER
             if(elem==8 or elem==9):
               sampleLabels[2]=1
             #HAND-TORSO
             if(elem==10 or elem==11):
               sampleLabels[3]=1
             #HAND-ELBOW
             if(elem==12 or elem==13 or elem==14 or elem==15):
               sampleLabels[4]=1
             #ELBOW-SHOULDER
             if(elem==16 or elem==17 or elem==18 or elem==19):
               sampleLabels[5]=1  
      
          #print(sampleLabels)
  
      # 3.4. Once all the proxemics vectors of that image have been processed, 
      #  the resulting vector and the processed image name are added to the list of labels [imgName, sampleLabels]
      labels.append( np.array([mat['proxemics'][0][i][0][0],sampleLabels], dtype="object"))

    # 4. The list with all the modified labels is added to the Proxemics key 
    proxemics['Proxemics']=labels

    # 5. We save the dictionary in a .mat file (labels_6classes.mat)
    scipy.io.savemat(newLabelsPath, proxemics)     #"labels_6classes.mat"




# Image pre-processing. Save them preprocessed at 224x224.
# We overwrite the original images with the new rescaled ones.
# Create a new labels_6classes.mat with the height and width information of each image (2 de diciembre 2022)
def img_preprocessing(labelsFile, imgDir):   #imgDir=/Proxemics/images/
    
    labels = scipy.io.loadmat(labelsFile)   #'labels_6classes.mat

    # 1. We calculate the total number of samples to process
    # There are 1178 labels because it includes the release and fipped labels.
    imglen=len(labels['Proxemics'])
    #print(imglen)

    # 2. We initialize variables
    # Dictionary where we will store the key : Proxemics 
    # This key will have a list of elements 
    #   where each element will be of the form : [ImageName, label, h, w]
    proxemicsFinal={}
    # Lista de elementos [ImageName, label, h, w]
    labelsFinal=[]

    # 2. We run through each sample of the labels file.
    
    for i in range(0,imglen):
      
      print('Analizando :', labels['Proxemics'][i][0]) 
            
      # To manage the paths of each photo to be processed
      if i <589:
        id=i+1
        if id > 99 :
          imgName='release/0'+str(id)+'.jpg'
        elif id>9 and id<100:
          imgName='release/00'+str(id)+'.jpg'
        else:
          imgName='release/000'+str(id)+'.jpg'
      else:
        id=i+1-589
        if id > 99 :
          imgName='flipped/0'+str(id)+'.jpg'
        elif id>9 and id<100:
          imgName='flipped/00'+str(id)+'.jpg'
        else:
          imgName='flipped/000'+str(id)+'.jpg'

      imgPath = os.path.join(imgDir,imgName)
      #print(imgPath)
      #Windows11
      #imgPath = imgPath.replace('/','\\')
      #print(imgPath)

      #3. We load the image to be processed
      img = cv2.imread(imgPath)
      #cv2_imshow(img)

      # 4. Image size
      h,w,c=img.shape #height, width, channel
      # 3.4. Once all the proxemics vectors of that image have been processed, 
      #  the resulting vector and the processed image name are added to the list of labels [imgName, sampleLabels]
      labelsFinal.append( np.append(labels['Proxemics'][i],[h,w]))
 

      # 5. Calculate the padding needed to make our image square
      padding1=padding2=int(abs(h-w)/2) #Round to the integer (4.6-> 4)
      # 5.1. If the division is not exact, we increase padding1 by 1 
      if abs(h-w)%2 != 0:
        padding1=padding1+1

      # 6. We check where we have to apply padding
      if h < w: # Apply padding to height
         img = cv2.copyMakeBorder (img, padding1, padding2,0,0, cv2.BORDER_CONSTANT, value = [0,0,0]) # agregar borde negro (superior, inferior, izquierda y derecha)
      else:     # Apply padding to width
         img = cv2.copyMakeBorder (img, 0, 0,padding1,padding2, cv2.BORDER_CONSTANT, value = [0,0,0]) # agregar borde

      # 7. We already have the square image.We rescale the image to 224x224
      img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
      #cv2_imshow(img)

      # 8. Save the new image
      cv2.imwrite(imgPath, img)

    # 4. The list with all the modified labels is added to the Proxemics key 
    proxemicsFinal['Proxemics']=labelsFinal

    # 5. We save the dictionary in a .mat file (labels_6classes.mat)
    scipy.io.savemat(labelsFile, proxemicsFinal)     #"labels_6classes.mat"



if __name__ == '__main__':
    datasetDir='C:\\Users\\isabe\\Desktop\TFM\PROXEMICS_dataset\\pp\\PROXEMICS\\PROXEMICS\\'
    print(datasetDir)

    #Preprocess Labels
    originalLabelsPath= os.path.join(datasetDir+'labels.mat')
    newLabelsPath=  os.path.join(datasetDir+'labels_6classes.mat')

    label_preprocessing(originalLabelsPath,newLabelsPath)

    labels = scipy.io.loadmat(newLabelsPath) 

    print('EXAMPLE NEW LABEL: ',list(labels['Proxemics'][3]))


    #Preprocess img
    imgDir= os.path.join(datasetDir,'images')
    img_preprocessing(newLabelsPath, imgDir)

    labels = scipy.io.loadmat(newLabelsPath) 

    print('EXAMPLE NEW LABEL: ',list(labels['Proxemics'][3]))

    print("DONE!")