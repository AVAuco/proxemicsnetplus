################################################
#Code to obtain the clippings of each person and each couple present in an image.
################################################

import numpy as np
import argparse
import json, cv2, random
import sys, os, distutils.core
from retinaface import RetinaFace
import matplotlib.pyplot as plt
import torch
import detectron2
import torchvision
print("CUDA Available:", torch.cuda.is_available())
print("Torch version:", torch.__version__)
print("Torch version:", torchvision.__version__)
print("CUDA version:", torch.version.cuda)
print("Detectron2 version:", detectron2.__version__)

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.config import get_cfg
cfg = get_cfg()
cfg.MODEL.DEVICE = "cuda"  # Asegura que Detectron2 usará CUDA

from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog


# Function that calculates the Euclidean distance (error) between 2 points (x,y)
def calculate_error_kps(p1,p2):
  dist=p2-p1
  error= np.linalg.norm(dist)
  return error

# Function that rescales an image to 224x224 and adds black row or column padding
def resizeImage(img):
  # 1. Image size
  h,w,c=img.shape #height, width, channel
  # 2. Calculate the padding needed to make our image square
  padding1=padding2=int(abs(h-w)/2) #Round to the integer (4.6-> 4)
  # 2.1. If the division is not exact, we increase padding1 by 1
  if abs(h-w)%2 != 0:
    padding1=padding1+1
  # 3. We check where we have to apply padding
  if h < w: # Apply padding to height
     img = cv2.copyMakeBorder (img, padding1, padding2,0,0, cv2.BORDER_CONSTANT, value = [0,0,0]) # agregar borde negro (superior, inferior, izquierda y derecha)
  else:     # Apply padding to width
     img = cv2.copyMakeBorder (img, 0, 0,padding1,padding2, cv2.BORDER_CONSTANT, value = [0,0,0]) # agregar borde
  # 4. We already have the square image.We rescale the image to 224x224
  img = cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_LINEAR)
  return img

#Function normalizing images (test and train)
def normalize(img):
  # Normalize from [0-255] to [0-1] y center between [-0.5,0.5]
  img=img.astype('float32')
  img /= 255.0
  img -= 0.5
  return img

def bbox_iou(boxA, boxB):
    # Determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both the prediction and ground-truth rectangles
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # Return the intersection over union value
    return iou


 
def parse_args():
    parser = argparse.ArgumentParser(description='Retina_Detectron_detections script.')

    #PATHS
    parser.add_argument('--imagesPath',type=str,  help='Main Dir where images are located', required=True)
    parser.add_argument('--outputPath',type=str,  help='Dir where preprocesed images (clippings) will be saved', required=True)

    return parser.parse_args()

################################################## START OF THE CODE ##################################################
if __name__ == '__main__':

  #Parameters read from console
  args = parse_args()

  #1. Some basic setup: PATH
  imagespath=args.imagesPath
  clippingspath=args.outputPath

  imglist=sorted(os.listdir(imagespath))


  # 2. Selected detectron2 models 
  #First model
  print("First model")
  cfg = get_cfg()
  pretrained_model_principal='COCO-Keypoints/keypoint_rcnn_R_101_FPN_3x.yaml'
  cfg.merge_from_file(model_zoo.get_config_file(pretrained_model_principal))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
  cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pretrained_model_principal)
  predictor1 = DefaultPredictor(cfg)
  #Second model
  print("Second model")
  cfg2 = get_cfg()
  pretrained_model_secundario='COCO-Keypoints/keypoint_rcnn_R_50_FPN_3x.yaml'
  cfg2.merge_from_file(model_zoo.get_config_file(pretrained_model_secundario))
  cfg2.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
  cfg2.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(pretrained_model_secundario)
  predictor2 = DefaultPredictor(cfg2)


  ################################# 3. Preprocess IMAGES ################################ 
  # To save the coordinates of the bbs of the pairs of each image
  jsonfileBBs=os.path.join(clippingspath,'pair_BBs.json')
  proxemics={}
  # We begin to process the images one by one.
  for imgname in imglist:
    print('IMG : ', imgname)
    proxemics[imgname]={}
    # Path of the image to be processed
    imgpath=os.path.join(imagespath,imgname)
    # Flag necessary for the choice of articulation that will allow us to make the order of the people.
    eye_flag=False
    shoulder_flag=False

    ################################  3.1 RetinaFace FaceDetector ################################
    # 1. RetinaFace FaceDetector -- For detecting the number of people in the image
    imageFaces = RetinaFace.detect_faces(imgpath)
    nfaces=len(imageFaces)
    # 1.1. In case RetinaFace detects less than 2 persons, the execution is interrupted and we move on to the next image
    if nfaces < 2:  # Detectro de caras ha detectado menos de 2 caras o incluso ninguna
      print('ERROR 1: ' + str(nfaces) +  ' persons have been detected in image '+ imgpath)
      print()
      continue

    ################################  3.2. Detectron2 ################################
    # 2. Once 2 or more faces have been detected -- We apply the detectron models
    img = cv2.imread(imgpath)
    # 2.1. We apply the first detectron model on the image to be processed
    imageDetectron = predictor1(img)
    # 2.2. We check the number of BBs it has detected (one BB corresponds to one person).
    ndetections=len(imageDetectron["instances"].pred_boxes)
    # If detectron has detected less than 2 persons or even no persons at all
    if ndetections < 2:
      print('ERROR 2: ' + str(ndetections) +  ' persons have been detected by detectron in image '+ imgpath)
      # 2.3. We apply the second detectron model
      imageDetectron = predictor2(img)
      # 2.2. We check the number of BBs it has detected (one BB corresponds to one person).
      ndetections=len(imageDetectron["instances"].pred_boxes)
      # If detectron has detected less than 2 persons or even no persons at all
      if ndetections < 2:
        print('ERROR 3: ' + str(ndetections) +  ' persons have been detected by detectron in image '+ imgpath)
        print()
        continue
  
    ################################  3.3. IoU ################################
    # 3.1. Check which retinaFace BBs match detectron BBs - IoU
    # Image size (weight, height)
    image_height, image_width, channels = img.shape

    # Define a minimum relative image size threshold (e.g. 0.05% of the total image area)- delete very small Detectron detections
    size_threshold = (image_width * image_height) * 0.0005
    
    # Dictionary to save final BBs
    correct_detectron_positions = set()

    # Iterate over each retinaFace detection
    # compare each RetinaFace face with all Detectron detections and keep the Detectron BBs that return the highest IoU
    for r_id, r_val in imageFaces.items():
        r_bbox = r_val['facial_area']
        r_area = (r_bbox[2] - r_bbox[0]) * (r_bbox[3] - r_bbox[1])
        # Discard if the area of the retinaFace BBs is less than the threshold.
        if r_area < size_threshold:
            continue
        
        best_iou = 0  
        best_index = None 

        # compare each RetinaFace face with all Detectron detections
        for d_index, d_bbox in enumerate(imageDetectron['instances'][imageDetectron['instances'].pred_classes==0].pred_boxes.tensor.cpu().numpy()):
            iou = bbox_iou(r_bbox, d_bbox)
            if iou > best_iou:  # Seach for the maximun IoU
                best_iou = iou
                best_index = d_index

        # Save the Detectron2 index with the highest IoU for each face - Only save the index where the BBs is in imageDetectron['instances']
        if best_index is not None:
          correct_detectron_positions.add(best_index)

    # Convert the set to a list for sorting the indexes
    correct_detectron_positions = list(correct_detectron_positions)
    correct_detectron_positions.sort()  


    ################################  3.4. Sort the detected persons in order to delimit who is person 0, person 1, etc. The left most person will be person 0.  ###############################
    #  Once we have applied detectron, if either of the 2 models has detected more than 2 persons
    # We start to extract the BBs of each person detected in that image and to order them according to the x-coordinate of 1 of the 3 chosen joints
    personperImage=[]
    # 3.4.1.Order by Nose
    # We store in a list the coordinate of each person's nose together with their corresponding BBs
    for idx in range(0, len(correct_detectron_positions)):
      person=correct_detectron_positions[idx]
      nose=imageDetectron['instances'][imageDetectron['instances'].pred_classes==0].pred_keypoints.cpu().numpy()[person][0][0:2]  #nose
      BBs=list(imageDetectron['instances'][imageDetectron['instances'].pred_classes==0].pred_boxes.tensor.cpu().numpy()[person])
      personperImage.append([list(nose),BBs])
      # 3.1.1.  We calculate the Euclidean distance between the coordinates of the noses of 2 people
      if idx>0:
        dist=calculate_error_kps(np.array(personperImage[idx-1][0]),np.array(personperImage[idx][0]))
        # If it does not exceed a distance threshold, another coordinate is chosen
        if dist < 1:
          personperImage=[]
          eye_flag=True
          break

    #  3.4.2 We order by the RIGHT EYE
    # We store in a list the coordinate of each person's nose together with their corresponding BBs
    if eye_flag:
      for idx in range(0, len(correct_detectron_positions)):
        person=correct_detectron_positions[idx]
        r_eye=imageDetectron['instances'][imageDetectron['instances'].pred_classes==0].pred_keypoints.cpu().numpy()[person][2][0:2]  #eye
        BBs=list(imageDetectron['instances'][imageDetectron['instances'].pred_classes==0].pred_boxes.tensor.cpu().numpy()[person])
        personperImage.append([list(r_eye),BBs])
        # 3.2.1.  We calculate the Euclidean distance between the coordinates of the noses of 2 people
        if idx>0:
          dist=calculate_error_kps(np.array(personperImage[idx-1][0]),np.array(personperImage[idx][0]))
          # If it does not exceed a distance threshold, another coordinate is chosen
          if dist < 1:
            personperImage=[]
            shoulder_flag=True
            break

    # 3.4.3 We order by the RIGHT SHOULDER
    # We store in a list the coordinate of each person's nose together with their corresponding BBs
    if shoulder_flag:
      for person in correct_detectron_positions:
        r_shoulder=imageDetectron['instances'][imageDetectron['instances'].pred_classes==0].pred_keypoints.cpu().numpy()[person][6][0:2]  #shoulder
        BBs=list(imageDetectron['instances'][imageDetectron['instances'].pred_classes==0].pred_boxes.tensor.cpu().numpy()[person])
        personperImage.append([list(r_shoulder),BBs])

    #  We deactivate all coordinate flags
    eye_flag=False
    shoulder_flag=False

    # We sort people by x-axis with the sort() function.
    personperImage.sort()

    ################################  3.5. Cut individuals and pairs of individuals according to their BBs  ###############################
    #Path where we saved all the clippings
    output=os.path.join(clippingspath,'recortes/')
    os.makedirs(output, exist_ok=True)

    # 5.1. We scroll through all persons that have been saved in the personImage list and that are already sorted by p0, p1.....
    for person in range(0, len(personperImage)-1):
      # We generate lists where we will store the x and y coordinates of the corners of the BBs of the pair to be processed
      x_inicial, x_final, y_inicial, y_final=[],[],[],[]

      x_inicial.append(personperImage[person][1][0])
      x_final.append(personperImage[person][1][2])
      y_inicial.append(personperImage[person][1][1])
      y_final.append(personperImage[person][1][3])

      # --> RESIZE OF THE PERSON 0 OF THE PAIR TO BE PROCESSED
      p0=resizeImage(img[int(y_inicial[0]):int(y_final[0]),int(x_inicial[0]):int(x_final[0])])
      p0_name= os.path.join(output, (imgname[:-4]+'_p'+str(person)+'.jpg'))
      p0_aux='p'+str(person)
      cv2.imwrite(p0_name,p0)
      proxemics[imgname][p0_aux]=[ int(y_inicial[0]), int(y_final[0]), int(x_inicial[0]), int(x_final[0]) ]
      # For person 0, we generate all the possible pairs with the rest of the detected persons.
      for person2 in range(person+1, len(personperImage)):
        x_inicial.append(personperImage[person2][1][0])
        x_final.append(personperImage[person2][1][2])
        y_inicial.append(personperImage[person2][1][1])
        y_final.append(personperImage[person2][1][3])

        # --> RESIZE OF THE PERSON 1 OF THE COUPLE TO BE PROCESSED
        p1=resizeImage(img[int(y_inicial[1]):int(y_final[1]),int(x_inicial[1]):int(x_final[1])])
        p1_name= os.path.join(output, (imgname[:-4]+'_p'+str(person2)+'.jpg'))
        cv2.imwrite(p1_name,p1)
        p1_aux='p'+str(person2)
        proxemics[imgname][p1_aux]=[ int(y_inicial[1]), int(y_final[1]), int(x_inicial[1]), int(x_final[1]) ]

        # --> RESIZE OF THE COUPLE TO BE PROCESSED - as a function of the maxima and minima of the x and y coordinates of the 2 persons composing the pair.
        p0p1=resizeImage(img[ int(min(y_inicial)):int(max(y_final)),int(min(x_inicial)):int(max(x_final))  ])
        p0_p1_name=os.path.join(output, (imgname[:-4]+'_pair_p'+str(person)+'-p'+str(person2)+'.jpg'))
        cv2.imwrite(p0_p1_name,p0p1)
        p0_p1_aux='p'+str(person)+'-p'+str(person2)
        proxemics[imgname][p0_p1_aux]= [ int(min(y_inicial)), int(max(y_final)), int(min(x_inicial)), int(max(x_final))   ]

        # We extract the data stored in the lists about person 1 from the pair we have just processed to move on to the next person who forms another pair with person 0.
        x_inicial.pop()
        x_final.pop()
        y_inicial.pop()
        y_final.pop()

  # Save final json
  with open(jsonfileBBs, 'w') as fp:
      json.dump(proxemics, fp,indent=4)

  # ZIP clippings
  #zip -r clippings.zip output
