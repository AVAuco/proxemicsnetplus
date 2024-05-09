

from heapq import nsmallest
import numpy as np
import tensorflow.keras as keras
from   tensorflow.keras.preprocessing.image import ImageDataGenerator
from train.load_dataset_train_test import feature_extractor

import random
import copy
from time import time
import os, tarfile
import sys 
import cv2
from PIL import Image
from zipfile import *

import random


def ij_getImage_PairSeqFromDir(zip,  imgname, p0_name, p1_name, imgDict={}):
	
	imgname_path_p0=os.path.join('recortes/',(imgname+'_'+p0_name+'.jpg'))
	imgname_path_p1=os.path.join('recortes/',(imgname+'_'+p1_name+'.jpg'))
	imgname_path_pair=os.path.join('recortes/',(imgname+'_pair_'+p0_name+'-'+p1_name+'.jpg'))
	
	imgnameList=[imgname_path_p0,imgname_path_p1,imgname_path_pair]
	
	#print(imgnameList)
 
	c_p0 = zip.read(imgname_path_p0)
	c_p1 = zip.read(imgname_path_p1)
	c_pair = zip.read(imgname_path_pair)
  
	images=[]
	if imgname_path_pair in imgDict:
		for i in range(0,3):
			images.append(imgDict.get(imgnameList[i]))  #Ya está guardado

	else:
		i=0
		for c in [c_p0,c_p1,c_pair]:
			if sys.getsizeof(c) > 266:
				na = np.frombuffer(c, dtype=np.uint8)
				im = cv2.imdecode(na, cv2.IMREAD_COLOR)
				images.append(im)
				imgDict[imgnameList[i]]=im
			i=i+1
	
	#print(imgDict)
	return images[0],images[1],images[2]
	



def ij_getPoseImage_PairSeqFromDir(zip,  imgname, p0_name, p1_name ,imgDict={}):
	#proxemics
	imgname_path_p0=os.path.join('poseImg_I_recortes_thr0_pair/',('pose_'+imgname+'_'+p0_name+'.jpg'))
	imgname_path_p1=os.path.join('poseImg_I_recortes_thr0_pair/',('pose_'+imgname+'_'+p1_name+'.jpg'))
	imgname_path_pair=os.path.join('poseImg_I_recortes_thr0_pair/',('pose_'+imgname+'_pair_'+p0_name+'-'+p1_name+'.jpg'))
	'''
	#pisc
	imgname_path_p0=os.path.join('poseImg_I_recortes_pair/',('pose_'+imgname+'_'+p0_name+'.jpg'))
	imgname_path_p1=os.path.join('poseImg_I_recortes_pair/',('pose_'+imgname+'_'+p1_name+'.jpg'))
	imgname_path_pair=os.path.join('poseImg_I_recortes_pair/',('pose_'+imgname+'_pair_'+p0_name+'-'+p1_name+'.jpg'))
	'''
	
	imgnameList=[imgname_path_p0,imgname_path_p1,imgname_path_pair]
	
	#print(imgnameList)
 
	c_p0 = zip.read(imgname_path_p0)
	c_p1 = zip.read(imgname_path_p1)
	c_pair = zip.read(imgname_path_pair)
  
	images=[]
	if imgname_path_pair in imgDict:
		for i in range(0,3):
			images.append(imgDict.get(imgnameList[i]))  #Ya está guardado

	else:
		i=0
		for c in [c_p0,c_p1,c_pair]:
			if sys.getsizeof(c) > 266:
				na = np.frombuffer(c, dtype=np.uint8)
				im = cv2.imdecode(na, cv2.IMREAD_COLOR)
				images.append(im)
				imgDict[imgnameList[i]]=im
			i=i+1
	
	#print(imgDict)
	return images[0],images[1],images[2]



def ij_getfeatures_PairSeqFromDir(featuresPath, imgname, p0_name, p1_name):
	#Estoy abriendo el texto
	data = np.load(featuresPath, allow_pickle=True)
	#print(list(data.keys()))
	featurefilename=imgname+'_pair_'+p0_name+'-'+p1_name+'.jpg'
	#print(featurefilename)
	feature=data[featurefilename]
	#print(feature)
	data.close()
	return feature




 #---------------------------------------------------------------------------------------------------

class DataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self,  list_IDs, allSamples, batch_size=32, shuffle=True, augmentation=True, isTest=False,zipPath='', posezipPath='', onlyPairRGB=False,onlyPairPose=False,  typeImg='', isTransformer=False, typeTransformer='vit', use_contrastiveloss=False, features=False, featuresTextFilePath="" ):
		'Initialization'


		self.batch_size = batch_size

		self.list_IDs = copy.deepcopy(list_IDs)

		if augmentation:
			for aix in range(0,1):
				self.list_IDs.extend(-np.array(list_IDs))  # Minus means to be perturbated
		if isTest == False:
			np.random.shuffle(self.list_IDs)   # Done always!
		
		self.shuffle = shuffle
		self.augmentation = augmentation
	
		self.allSamples = allSamples

		#self.recortesImg = recortesImg
		self.typeImg = typeImg

	
		self.isTest = isTest
		self.onlyPairRGB=onlyPairRGB
		self.onlyPairPose=onlyPairPose

		self.contrastiveloss=use_contrastiveloss   #constractive loss --> mistach en las muestras

		self.isTransformer=isTransformer
		self.typeTransformer=typeTransformer

		if 'RGB' in self.typeImg:
			self.imgDict={}
			zipname=zipPath

			if not os.path.exists(zipname):
				print("ERROR: cannot find tar file: {}".format(zipname))
				exit(-1)

			self.zip =  ZipFile(zipname, mode='r')

			if self.zip is None:
				print("Error reading zip file!!!")
				exit(-1)

		if 'Pose' in self.typeImg:
			self.poseimgDict={}
			posezipname=posezipPath
			if not os.path.exists(posezipname):

				print("ERROR: cannot find tar file: {}".format(posezipname))
				exit(-1)

			self.posezip =  ZipFile(posezipname, mode='r')

			if self.posezip is None:
				print("Error reading zip file!!!")
				exit(-1)

		self.features=features
		self.featuresTextFilePath=featuresTextFilePath


		# Needed for initialization
		self.img_gen = ImageDataGenerator(brightness_range=[0.6,1.3],shear_range=4.5,zoom_range=0.09,horizontal_flip=True)
		
		self.__splitInProxemicsTypes()
		
		self.on_epoch_end()






	def __len__(self):
		'Number of batches per epoch'

		return int(np.floor(len(self.list_IDs) / self.batch_size))






	def __getitem__(self, index):
		'Generate one batch of data'
		#Indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
		
		if len(indexes) < 1:
			rnd = random.randint(0,len(self.indexes)-self.batch_size)
			indexes = self.indexes[rnd:rnd+self.batch_size]

		# Find list of ids
		list_IDs_temp = [self.list_IDs[k] for k in indexes]
		
		# Generate data
		X, y = self.__data_generation(list_IDs_temp)


		# Store classes
		self.list_IDs_temp = list_IDs_temp

		return X, y


	def on_epoch_end(self):
		'Updates indexes after each epoch'
		#Para cada época preparamos todos las muestras que serán cogidas a lo largo de la siguiente época y que se separarán en batches
		
		#self.indexes = np.arange(len(self.list_IDs))
		#
		#if self.shuffle == True:
		#    np.random.shuffle(self.indexes)
		
		if self.isTest:
			self.indexes = np.arange(len(self.list_IDs))
		else:
			if self.shuffle:
				np.random.shuffle(self.HH)
				np.random.shuffle(self.HS)
				np.random.shuffle(self.SS)
				np.random.shuffle(self.HT)
				np.random.shuffle(self.HE)
				np.random.shuffle(self.ES)

			# Balanced batch
			HH = len(self.HH)
			HS = len(self.HS)
			SS = len(self.SS)
			HT = len(self.HT)
			HE = len(self.HE)
			ES = len(self.ES)

			tmpHH = copy.deepcopy(self.HH)
			tmpHS = copy.deepcopy(self.HS)
			tmpSS = copy.deepcopy(self.SS)
			tmpHT = copy.deepcopy(self.HT)
			tmpHE = copy.deepcopy(self.HE)
			tmpES = copy.deepcopy(self.ES)
	 
			#Maximo número de muestras de una clase
			maximun=max(HH,HS,SS,HT,HE,ES)

			#Alargamos todos a ese tamaño
			if HH<maximun:
				df=maximun - HH
				tmpHH.extend(self.HH[0:abs(df)])
			if HS<maximun:
				df=maximun - HS
				tmpHS.extend(self.HS[0:abs(df)])
			if SS<maximun:
				df=maximun - SS
				tmpSS.extend(self.SS[0:abs(df)])
			if HT<maximun:
				df=maximun - HT
				tmpHT.extend(self.HT[0:abs(df)])
				if len(tmpHT)<maximun:
					df=maximun - len(tmpHT)
					tmpHT.extend(self.HT[0:abs(df)])
			if HE<maximun:
				df=maximun - HE
				tmpHE.extend(self.HE[0:abs(df)])
			if ES<maximun:
				df=maximun - ES
				tmpES.extend(self.ES[0:abs(df)])

			totalSamples=maximun*6 #todos tendrán la longitud del maximo
			
			nbatches = int(np.floor(totalSamples/self.batch_size))

			sixthBatch=int(self.batch_size/6)

			indexes = []
		
			for i in range(0,nbatches):

				indexes.extend(tmpHH[i* sixthBatch:(i+1)* sixthBatch])
				indexes.extend(tmpHS[i* sixthBatch:(i+1)* sixthBatch])
				indexes.extend(tmpSS[i* sixthBatch:(i+1)* sixthBatch])
				indexes.extend(tmpHT[i* sixthBatch:(i+1)* sixthBatch])
				indexes.extend(tmpHE[i* sixthBatch:(i+1)* sixthBatch])
				indexes.extend(tmpES[i* sixthBatch:(i+1)* sixthBatch])
		
		

			self.indexes = indexes
			
		
							   
	def __splitInProxemicsTypes(self):
		self.HH = []
		self.HS = []
		self.SS = []
		self.HT = []
		self.HE = []
		self.ES = []

		for ix_ in range(0,len(self.list_IDs)):
			ix = abs(self.list_IDs[ix_])
			if self.allSamples[ix][3][0] == 1: 
				self.HH.append(ix_)
			if self.allSamples[ix][3][1] == 1: 
				self.HS.append(ix_)
			if self.allSamples[ix][3][2] == 1: 
				self.SS.append(ix_)
			if self.allSamples[ix][3][3] == 1: 
				self.HT.append(ix_)
			if self.allSamples[ix][3][4] == 1: 
				self.HE.append(ix_)
			if self.allSamples[ix][3][5] == 1: 
				self.ES.append(ix_)	


	
	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization

			
		this_bsize = len(list_IDs_temp)  # This replaces self.batch_size, as it is more robust for incomplete batches
		#print(list_IDs_temp)
		yG = np.zeros((len(list_IDs_temp), 6), dtype=int)
		

		# Assign correct labels and get samples ((32,224,224,3))
		if 'RGB' in self.typeImg:
			p0_fb=np.zeros((this_bsize,224,224,3))
			p1_fb=np.zeros((this_bsize,224,224,3))
			pair_fb=np.zeros((this_bsize,224,224,3))
		if 'Pose' in self.typeImg:
			pose_p0_fb=np.zeros((this_bsize,224,224,3))
			pose_p1_fb=np.zeros((this_bsize,224,224,3))
			pose_pair_fb=np.zeros((this_bsize,224,224,3))
		if self.features:
			features_pair_text_fb=np.zeros((this_bsize,12,768))



		nSamples=0
		for i in range(0, len(list_IDs_temp)):
			ix = abs(list_IDs_temp[i])
			
			imgname = self.allSamples[ix][0]
			p0_name = self.allSamples[ix][1]
			p1_name = self.allSamples[ix][2]
			label = self.allSamples[ix][3]

			try:
				if 'RGB' in self.typeImg:
					p0, p1, pair= ij_getImage_PairSeqFromDir(zip=self.zip,  imgname=imgname, p0_name=p0_name, p1_name=p1_name, imgDict=self.imgDict)
				if 'Pose' in self.typeImg:
					pose_p0, pose_p1, pose_pair= ij_getPoseImage_PairSeqFromDir(zip=self.posezip, imgname=imgname, p0_name=p0_name, p1_name=p1_name, imgDict=self.poseimgDict)
				if self.features:
					features_pair_text=ij_getfeatures_PairSeqFromDir(featuresPath=self.featuresTextFilePath, imgname=imgname, p0_name=p0_name, p1_name=p1_name)  #imgDict=self.featuresTextDict
					#print(features_pair_text.shape)
			except:
				print('Data load Exception')
				continue
			

			# Augmentation
			
			if self.augmentation and list_IDs_temp[i] < 0:
				transformation = self.img_gen.get_random_transform((224,224))
				#print(transformation)

				if transformation["flip_horizontal"] == 1:
					if 'RGB' in self.typeImg:
						p1_aux = self.img_gen.apply_transform(p0, transformation)
						p0 = self.img_gen.apply_transform(p1, transformation)
						p1= copy.deepcopy(p1_aux)
						pair = self.img_gen.apply_transform(pair, transformation)
					if 'Pose' in self.typeImg:
						posep0_tmp = np.fliplr(pose_p0)
						pose_p0 = posep0_tmp[:, :, [0, 2, 1]]  #Cambiamos los colores de los brazos
						posep1_tmp = np.fliplr(pose_p1)
						pose_p1 = posep1_tmp[:, :, [0, 2, 1]]  #Cambiamos los colores de los brazos
						posepair_tmp = np.fliplr(pose_pair)
						pose_pair = posepair_tmp[:, :, [0, 2, 1]]  #Cambiamos los colores de los brazos
				else:
					if 'RGB' in self.typeImg:
						p0 = self.img_gen.apply_transform(p0, transformation)
						p1 = self.img_gen.apply_transform(p1, transformation)
						pair = self.img_gen.apply_transform(pair, transformation)
			
			
		 	
			# Normalize from [0-255] to [0-1] y center between [-0.5,0.5]
			if self.isTransformer==False:
				if 'RGB' in self.typeImg:
					p0=p0.astype('float32')  
					p0 /= 255.0
					p0 -= 0.5

					p1=p1.astype('float32')  
					p1 /= 255.0
					p1 -= 0.5
					
					pair=pair.astype('float32')  
					pair /= 255.0
					pair -= 0.5
			else:
				p0=np.array(Image.fromarray((p0 * 255).astype(np.uint8)).resize((224, 224)).convert('RGB'))
				p1=np.array(Image.fromarray((p1 * 255).astype(np.uint8)).resize((224, 224)).convert('RGB'))
				pair=np.array(Image.fromarray((pair * 255).astype(np.uint8)).resize((224, 224)).convert('RGB'))


			if 'RGB' in self.typeImg:
				p0_fb[nSamples,] = p0
				p1_fb[nSamples,] = p1
				pair_fb[nSamples,] = pair
			if 'Pose' in self.typeImg:
				pose_p0_fb[nSamples,] = pose_p0
				pose_p1_fb[nSamples,] = pose_p1
				pose_pair_fb[nSamples,] = pose_pair
			if self.features:
				features_pair_text_fb[nSamples,]=features_pair_text
			
			yG[nSamples] = label
			          
			nSamples=nSamples+1

		X = []
		if self.isTransformer==False:
			if 'RGB' in self.typeImg:
				if self.onlyPairRGB==False:
					X.append(p0_fb)
					X.append(p1_fb)
				X.append(pair_fb)
			if 'Pose' in self.typeImg:
				if self.onlyPairPose==False:
					X.append(pose_p0_fb)
					X.append(pose_p1_fb)
				X.append(pose_pair_fb)
			if self.features:
				X.append(features_pair_text_fb)
				
		else:
			if self.onlyPairRGB==False:
				X.append(feature_extractor(p0_fb,self.typeTransformer))
				X.append(feature_extractor(p1_fb,self.typeTransformer))
			X.append(feature_extractor(pair_fb,self.typeTransformer))



		return X, yG
