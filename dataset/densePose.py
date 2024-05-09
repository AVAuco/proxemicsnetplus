######################################################
#CODE TO OBTAIN RGB POSE MAPS FROM DENSEPOSE ESTIMATES
#Densepose returns a pkl file with estimates for each image received. 
#To get this pkl you have to install the densepose GitHub project and apply its code apply_net.py to your images.
#cd /opt/data/isajim/experiments/proxemics/detectron2/projects/DensePose/
#python apply_net.py dump configs/densepose_rcnn_R_50_FPN_s1x.yaml https://dl.fbaipublicfiles.com/densepose/densepose_rcnn_R_50_FPN_s1x/165712039/model_final_162be9.pkl  /opt/data/isajim/experiments/proxemics/input/"*jpg" --output /opt/data/isajim/experiments/proxemics/release_recortes.pkl -v

#This code is in charge of processing that pkl file to obtain the RGB pose maps.
######################################################

import torch, detectron2
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Función que reescala una imagen a 224x224 y le añade padding de filas o columnas negras
def resizeImage(img):
  # 4. Image size
  h,w,c=img.shape #height, width, channel
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
  return img

def calculate_iou(box1, box2):
    # Coordinates of the intersection of the BBs
    x_inter1 = max(box1[0], box2[0])
    y_inter1 = max(box1[1], box2[1])
    x_inter2 = min(box1[2], box2[2])
    y_inter2 = min(box1[3], box2[3])
    # Intersection area
    inter_area = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
    # Areas of individual BBs
    area_box1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area_box2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    # Joint area
    union_area = area_box1 + area_box2 - inter_area
    # IoU
    iou = inter_area / union_area if union_area != 0 else 0

    return iou

def find_max_iou_idx(bbs_list, bb_target):
  ious = [calculate_iou(bb_target, box) for box in bbs_list]
  max_iou = max(ious)  # obtain the highest IoU
  if max_iou >= 0.5:
      return ious.index(max_iou)  #Returns the index of the largest IoU exceeding the threshold.
  else:
      return -1  # Not enough overlap

#Function to remove noise from the pose maps of couples
def subtract2img(img1,img2):
  height, width, channels = img1.shape
  newImg = np.zeros((height, width, channels))
  values = [  [0,127,255],  [0,85,170,255],   [0,120,170,255] ]
  for rows in range(0,height):
    for cols in range(0,width):
      for c in range(0,3):
        newImg[rows][cols][c]=img1[rows][cols][c]-img2[rows][cols][c]

        if newImg[rows][cols][c] <0:
          newImg[rows][cols][c]=0
        if newImg[rows][cols][c] not in values[c]:
          newImg[rows][cols][c]=0

  return newImg

#Function to sum the pose maps of each individual to form the partner map.
def add2img(img1, img2):
    height, width, channels = img1.shape
    newImg = np.zeros((height, width, channels), dtype=np.uint8)
    values = [[0, 127, 255], [0, 85, 170, 255], [0, 120, 170, 255]]

    for row in range(height):
        for col in range(width):
            for c in range(channels):
                if img1[row][col][c] != 0 and img2[row][col][c] != 0:
                    # If both pixels are non-zero, select the value for img1
                    newImg[row][col][c] = img1[row][col][c]
                else:
                    #  Add the values and check if the result is valid.
                    added_value = img1[row][col][c] + img2[row][col][c]
                    # Ensure that the value is within the allowable values.
                    if added_value in values[c]:
                        newImg[row][col][c] = added_value
                    else:
                        # SIf it is not an allowed value, set to zero.
                        newImg[row][col][c] = 0

    return newImg

def printPersonImg(img,imgFinal,imgaux,p_idx,xmin,xmax,ymin,ymax):
  # We draw the I  coordinates map at the corresponding place in the  image (depending on the BBs of the detection we are analyzing)
  m=0
  n=0
  for rows in range(int(ymin), int(ymax)-1):
      n=0
      for cols in range(int(xmin),int(xmax)-1):
          # I map BBs
          value=img[m][n]
          #Different shades within the same channel to distinguish arm, forearm and hand for example.
          #HEAD AND BODY
          if value in [2]:
              imgFinal[p_idx][rows][cols][0]=127
              imgaux[rows][cols][0]=127
          elif value in [23,24]:
              imgFinal[p_idx][rows][cols][0]=255
              imgaux[rows][cols][0]=255
          #LEFT ARM
          elif value in [3]:
            imgFinal[p_idx][rows][cols][1]=255
            imgaux[rows][cols][1]=255
          elif value in [16,18]:
            imgFinal[p_idx][rows][cols][1]=170
            imgaux[rows][cols][1]=170
          elif value in [20,22]:
            imgFinal[p_idx][rows][cols][1]=85
            imgaux[rows][cols][1]=85
          #RIGHT ARM
          elif value in [4]:
            imgFinal[p_idx][rows][cols][2]=255
            imgaux[rows][cols][2]=255
          elif value in [15,17]:
            imgFinal[p_idx][rows][cols][2]=170
            imgaux[rows][cols][2]=170
          elif value in [19,21]:
            imgFinal[p_idx][rows][cols][2]=120
            imgaux[rows][cols][2]=120
          n=n+1
      m=m+1

def recorte(img):
  height, width, channels = img.shape
  x_inicial, y_inicial = width, height
  x_final, y_final = 0, 0

  for c in range(0,3):
    for rows in range(0, height):
      for cols in range(0,width):
        if img[rows][cols][c] !=0:
          # We are saving the coordinates of the corners of our BBs of the pair
          if cols < x_inicial:
            x_inicial=cols
          if cols > x_final:
            x_final=cols
          if rows < y_inicial:
            y_inicial=rows
          if rows > y_final:
            y_final=rows

  # Ensure that the cut is valid
  if x_final > x_inicial and y_final > y_inicial:
    #  Crop the image to the found coordinates
    return resizeImage(img[ int(y_inicial):int(y_final)+1,int(x_inicial):int(x_final)+1  ])  
  else:
    #   If no non-null pixels were found, return the original image.
    return img


################################################## START OF THE CODE ##################################################
#JSON generated after running retinaFace and detectron
import json
jsonfile='/opt/data/isajim/experiments/proxemics/output/pair_BBs.json'
with open(jsonfile) as file:
    BBs = json.load(file)
# Open pkl file
f = open('/opt/data/isajim/experiments/proxemics/release_recortes.pkl', 'rb')
print(f)
data = torch.load(f)


# path where the pose maps will be stored
outpath='/opt/data/isajim/experiments/proxemics/output/densePose/'


# 3. We processed the images
print('n images',len(data))
for idx in range(0,6):
  # 3.2. Name image to be processed
  img_data=data[idx]
  imgname=img_data['file_name']
  img_original=cv2.imread(imgname)
  h_image,w_image,c_image=img_original.shape
  imgname=imgname.split('/')[-1]
  print(imgname)
  # 3.1. Final map of each image (three channels - RGB)
  #Get how many people are in that image (the keys are p0, p1, p2...) so, the final imgFinal will consist of n_img - one per person.  
  imgKeys=BBs[imgname].keys()
  all_p_BBs=[]
  for key in imgKeys:
    if '-' not in key:    #We do not want to know the number of pairs (p0-p1) - so we discard them.
      all_p_BBs.append(key)
      
  n_img=len(all_p_BBs)
  imgFinal = np.zeros((n_img,h_image,w_image,3))
  imgaux = np.zeros((h_image,w_image,3))

  #We extract the BBs of each person in the image that were previously obtained with the retinaface (y0,y1,x0,x1).  
  BBslist=[]
  for p_BB in all_p_BBs:
      BB=BBs[imgname][p_BB]
      y0=BB[0]
      y1=BB[1]
      x0=BB[2]
      x1=BB[3]
      BBslist.append((x0,y0,x1,y1))

  print(BBslist)
  ########################  we generated the first version of the pose maps of the individuals ########################
  # 3.4. For each thing detected in the image we are processing (n_detections), we will obtain the I maps of their BBs and place them in the image
  # Pintamos en cada imagen de la persona correspondiente(p_idx) la detección encontrada de densepose.
  # Para ello, sacamos el BBs que devuelve densepose de esa detección - bbstarget
  # Luego comparamos ese bb con todos los posibles candidatos que hay (uno por persona en la imagen) y que hemos obtenido anteriormente del json (detectron) y guardado een el bbslist
  # La funcion find_max_iou_idx es la que nos devuelve el id del bb del candidato que más contenga al bbtarget ( es decir, nos devuelve la persona)
  # Posterormente pintamos las imagenes como en la celda anterior pero centrándonos en imagen de la persona en concreto - imgFinal[p_idx][rows][cols]
  n_detections=len(img_data['pred_boxes_XYXY'])

  for person_idx in range(0,n_detections):
    # 3.4.1. We check if the idx of the image is within the images with specific thresholds.
    score_thr=0
    # 3.4.2. We check if the score of the detection we are analyzing exceeds the established threshold
    score=img_data['scores'][person_idx]
    if score >= score_thr:
      # We take the coordinates of the bbs
      box=np.array(img_data['pred_boxes_XYXY'][person_idx])
      xmin=box[0]
      ymin=box[1]
      xmax=box[2]
      ymax=box[3]
      BBstarget=(xmin,ymin,xmax,ymax)

      #print(BBsConcrete)
      p_idx=find_max_iou_idx(BBslist,BBstarget)
      if p_idx == -1:
         continue
      # Transform the map with I from detection to cpu and numpy
      img=img_data['pred_densepose'][person_idx].labels.cpu().numpy()

      printPersonImg(img,imgFinal,imgaux,p_idx,xmin,xmax,ymin,ymax)

  ######################## we clean up noise - we refine estimates ########################
  #Como a veces las detecciones de densepose cogen dentro de una persona a la otra persona, reprocesamos las imagenes para quitar de la imagen a las personas que no correspondan.
  #Cómo? Supongamos que la imagen p0 está bien pero la imgen p1 tiene a la persona 1 pero además un poquito de la persona 0. Pues hay que restarle a la imagen p1, la imagen p0.
  # Primero contamos el numero de 0 que tiene cada imagen de personas y comparamos por parejas. De la pareja, la persona con menor número de 0 es la que más pintura tiene y por tanto  a la que se le va a restar la otra persona.
  # De esta manera, si restamos y sale negativo, pues se deja a 0 ya que significa que ahí no había nada y por el contrario, si el resultado de la resta no está entre los valores permitidos por canal(0,85,...255) se deja tambien a 0 - substract2img
  p_zeros_list=[]
  for i in imgFinal:

    #  Counts the number of pixels with value 0 in each colour channel
    r_zeros = np.count_nonzero(i[:,:,0] == 0)
    g_zeros = np.count_nonzero(i[:,:,1] == 0)
    b_zeros = np.count_nonzero(i[:,:,2] == 0)

    p_zeros=r_zeros+g_zeros+b_zeros
    p_zeros_list.append(p_zeros)

  n_total_zeros_perImage=h_image*w_image*c_image
  if n_total_zeros_perImage in p_zeros_list :    #That number means that the 3 channels of 224x224 are set to 0(224x224x3)
    imgFinal = np.zeros((n_img,h_image,w_image,3))
    for p_idx in range(0,n_img):
      # We take the coordinates of the bbs
      box=BBslist[p_idx]  #(x,y,h,w)
      xmin=box[0]
      ymin=box[1]
      xmax=xmin + box[3]
      ymax=ymin + box[2]
      print(xmin,xmax,ymin,ymax)
      for c in range(0,3):
        for rows in range(int(ymin), int(ymax)-1):
            for cols in range(int(xmin),int(xmax)-1):
                imgFinal[p_idx][rows][cols][c]=imgaux[rows][cols][c]
  else:
    for i in range(0,len(imgFinal)-1):
      for j in range(i+1,len(imgFinal)):
        if p_zeros_list[i]<p_zeros_list[j]:
          imgFinal[i]=subtract2img(imgFinal[i],imgFinal[j])
        else:
          imgFinal[j]=subtract2img(imgFinal[j],imgFinal[i])

  # We start generating each of the maps knowing what the BBs really are and refining the detections.
  cont=0
  for i in imgFinal:
    print('persona ', cont)
    name=outpath + 'pose_'+ imgname[:-4] + '_p' + str(cont) + '.jpg'
    imgperson = np.zeros((224,224,3))
    imgperson=recorte(i)

    figure=plt.figure(figsize=(2.24,2.24))
    plt.axis('off')
    plt.imshow(imgperson.astype('uint8'))
    plt.show()
    #3.6. Save final image
    figure.savefig(name)
    plt.close(figure)
    cont=cont+1
  if n_img==2:
    name=outpath + 'pose_'+ imgname[:-4] + '_pair_p0-p1.jpg'
    #print('pair p0-p1')
    imgperson = np.zeros((224,224,3))
    imgperson=recorte(imgaux)
    figure=plt.figure(figsize=(2.24,2.24))
    plt.axis('off')
    plt.imshow(imgperson.astype('uint8'))
    plt.show()
    figure.savefig(name)
    plt.close(figure)
  else:
    for i in range(0,len(imgFinal)-1):
      for j in range(i+1,len(imgFinal)):
        imgaux =  np.zeros((h_image,w_image,3))
        imgperson = np.zeros((224,224,3))
        imgaux=add2img(imgFinal[i],imgFinal[j])
        
        name=outpath + 'pose_'+ imgname[:-4] + '_pair_p'+str(i)+'-p'+str(j)+'.jpg'
        imgperson=recorte(imgaux)
        figure=plt.figure(figsize=(2.24,2.24))
        plt.axis('off')
        plt.imshow(imgperson.astype('uint8'))
        plt.show()
        figure.savefig(name)
        plt.close(figure)
