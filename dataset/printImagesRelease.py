# Script to print the release_pair images (224x224). It shows who is the person 0 and who the person 1 in the rescaled images.
# labels_6classes_pair.json is used
# We use the coordinates of the head of each person in each image
# 



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
    #2. PRE: We check that the labels and images have been preprocessed (224x224). 
    
    #a) Check Labels 
    labelFile_Original=os.path.join(datasetDir,'labels_6classes_pair.json')
    
    #b) Check Images

    #Check operative system to load image correctly
    my_os=sys.platform
    print("System OS : ",my_os,'\n')
    #Load the first image
    imgPathaux = os.path.join(datasetDir,'images/release_pair/0028.jpg')
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
   
    # 3. Read json file that containing the coordinates of each person in the images and their corresponding labels
    import json
    with open(labelFile_Original) as file:
        labels_Original = json.load(file)


    # 4. Directories PATH
    release_pair_dirpath=os.path.join(datasetDir,'images/release_pair/') #Input dir
    release_pair_dirpath2=os.path.join(datasetDir,'images/pp/')   #Output dir
    
    imgsList = os.listdir(release_pair_dirpath)



    #print(len(labels_Original['proxemics'][0][27][1][0]))
    #print(labels_Original['proxemics'][0][27][1][0][2][0][6])
    
    #,len(imgsList)
    
    # 5. Process each image one by one
    for idx in range(0,len(imgsList)):
        imgName=imgsList[idx]
        id=idx+1

        print(' - IMG Path: ',imgName)
        #print('     > ORIGINAL LABEL: ', labels_Original['proxemics'][0][idx])
        #print('     > ORIGINAL LABEL: ',labels_6classes['Proxemics'][idx][1])
        print()

        # 5.1. Image PATH
        imgPath = os.path.join(release_pair_dirpath,imgName)
          

        # 5.2. open the image
        img = cv2.imread(imgPath)
        
        # 5.3. For each image, we obtain the coordinates of the head of each person in this image
        for p in range(0, len(labels_Original[imgName]['coordinates'])):
           person='p'+str(p)
           punto=labels_Original[imgName]['coordinates'][person][0]
           print(punto)
           print(punto[0],punto[1])
            # Print 'p0', 'p1'.... in the image (coordinates --> punto)
           cv2.putText(img,  person,  (int(punto[0]),int(punto[1])),   cv2.FONT_HERSHEY_SIMPLEX , 1,  (0, 255, 255),  2, cv2.LINE_4) 


        # 5.4. output image name
        if id > 99 :
            newimgName='0'+str(id)+'.jpg'
        elif id>9 and id<100:
            newimgName='00'+str(id)+'.jpg'
        else:
            newimgName='000'+str(id)+'.jpg'

        # 5.5. Output image PATH
        finalnewImgPath= os.path.join(release_pair_dirpath2,newimgName)

        print('Image saved : ', finalnewImgPath)
        print('\n\n')

        cv2.imwrite(finalnewImgPath,img)
        cv2.destroyAllWindows()

    print(len(imgsList))

    print("DONE!")
