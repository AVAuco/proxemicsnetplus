#Script containing the necessary functions to generate any model(base_model, RGB+Pose - concatenation/crossAttention)

# Required libraries
from urllib.parse import quote_from_bytes
import numpy as np
import scipy.io
import cv2
import os
import copy

import tensorflow as tf
import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Flatten,Input,Dropout, LayerNormalization
from tensorflow.keras.layers.experimental import preprocessing


from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras import optimizers,layers

from transformers import TFViTForImageClassification, TFSwinForImageClassification, SwinConfig


############################################################################### TRANSFORMER ########################################################################
def get_basemodel_ViT_newInput(lr,optimizador):
        
  #transformer = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6, output_hidden_states=True)
  transformer0 = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6,name='ViT_Transformer_p0')
  rgbinput0 = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput_p0' )
  base_model0 = transformer0(rgbinput0)

  transformer1 = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6,name='ViT_Transformer_p1')
  rgbinput1 = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput_p1' )
  base_model1 = transformer1(rgbinput1)

  transformer_pair = TFViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k",num_labels=6,name='ViT_Transformer_pair')
  rgbinput_pair = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput_pair' )
  base_model_pair = transformer_pair(rgbinput_pair)


  # Concatenate branches
  the_concats = [base_model0[0], base_model1[0],base_model_pair[0]]
  the_inputs = [rgbinput0, rgbinput1, rgbinput_pair]

  concat =layers.concatenate(the_concats, name="concat_encods")

  # Add intermediate layers
  x = layers.Dense(4096, activation='relu', name='fc_1')(concat)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(4096, activation='relu', name='fc_2')(x)

  drp2 = layers.Dropout(0.5, name="top_dropout")(x)

  proxemicsoutput = tf.keras.layers.Dense(6, activation='sigmoid',name='Output')(drp2)

  finalModel = tf.keras.models.Model(inputs = the_inputs, outputs = proxemicsoutput,name="Proxemics_Vit_newInput")

  #  Optimizer
  if optimizador=='Adam':
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
  else:
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

  # Compile model
  finalModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])
  
  return finalModel


def get_basemodel_Swin_newInput(lr,optimizador,onlyPairRGB):
  model_load="microsoft/swin-base-patch4-window7-224-in22k"
  #model_load="microsoft/swin-tiny-patch4-window7-224"
  
  the_concats=[]
  the_inputs=[]
  #config = SwinConfig.from_pretrained("microsoft/swin-base-patch4-window7-224-in22k",output_hidden_states=True)
  # Load model 
  if onlyPairRGB==False:
    swin_model = TFSwinForImageClassification.from_pretrained(model_load,output_hidden_states=True)
    for layer in swin_model.layers:
      layer._name = layer.name + '_Individualbranch'

    rgbinput0 = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput_p0')
    rgbinput1 = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput_p1')

    base_model0 = swin_model.swin(rgbinput0)
    base_model1 = swin_model.swin(rgbinput1)

    the_concats = [base_model0.last_hidden_state, base_model1.last_hidden_state]
    the_inputs = [rgbinput0,rgbinput1]

  Swin_pair = TFSwinForImageClassification.from_pretrained(model_load,output_hidden_states=True)
  for layer in Swin_pair.layers:
    layer._name = layer.name + '_Pairbranch'

  rgbinput_pair = tf.keras.layers.Input(shape=(3,224,224), dtype=tf.float32,name='rgbInput_pair' )
  base_model_pair = Swin_pair.swin(rgbinput_pair)


  the_concats.append(base_model_pair.last_hidden_state)
  the_inputs.append(rgbinput_pair)


  if onlyPairRGB==True:  #Solo una rama ( solo psirRGB, por lo cual no es necesario concatenate)
    concat=the_concats[0]
    the_inputs=the_inputs[0]
  else:
    concat =layers.concatenate(the_concats, name="concat_encods",axis=-1)
    concat = LayerNormalization()(concat)

  flatten_concat = tf.keras.layers.Flatten()(concat)

  #  Add intermediate layers
  x = layers.Dense(4096, activation='relu', name='fc_1')(flatten_concat)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(4096, activation='relu', name='fc_2')(x)

  drp2 = layers.Dropout(0.5, name="top_dropout")(x)

  proxemicsoutput = tf.keras.layers.Dense(6, activation='sigmoid',name='Output')(drp2)

  finalModel = tf.keras.models.Model(inputs = the_inputs, outputs = proxemicsoutput,name="Proxemics_Swin")

  # Optimizer
  if optimizador=='Adam':
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
  else:
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

  # Compile model
  finalModel.compile(loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])

  return finalModel


############################################################################### ConvNext Concatenation ########################################################################
def generate(base_model,nlayersFreeze,name):
   finalModel=Sequential(name=name)
   i=0
   for layer in base_model.layers[1:-1]:
      if i < nlayersFreeze:
        layer.trainable=False
      
      finalModel.add(layer)
      i=i+1
   return finalModel


# Get the convnext base model 
def get_basemodel_convNext(model_gcs_path,lr,optimizador, typeImg, onlyPairRGB, onlyPairPose, nlayersFreeze):
  the_concats=[]
  the_inputs=[]
  if 'RGB' in typeImg:
    if onlyPairRGB==False:
      base_model_convnext =  tf.keras.models.load_model(model_gcs_path,compile=False )

      rgbinputp0 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='rgbinputp0' )
      rgbinputp1 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='rgbinputp1' )
    
      base_model=generate(base_model_convnext,nlayersFreeze,"individualbranch")
      
      base_model0 = base_model(rgbinputp0)
      base_model1 = base_model(rgbinputp1)

      the_concats = [base_model0, base_model1]
      the_inputs = [rgbinputp0,rgbinputp1]

    base_model_pair =  tf.keras.models.load_model(model_gcs_path,compile=False)
    i=0
    for layer in base_model_pair.layers:
      layer._name = layer.name + str("_pair") 
      if i > 0:                     # Freeze layers if we want
        if i <= nlayersFreeze:
          layer.trainable=False
      i=i+1

    the_concats.append(base_model_pair.layers[-2].output)
    the_inputs.append(base_model_pair.input)

  
  if 'Pose' in typeImg:
    if onlyPairPose==False:
      pose_model_convnext =  tf.keras.models.load_model(model_gcs_path,compile=False )

      pose_rgbinputp0 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='pose_rgbinputp0' )
      pose_rgbinputp1 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='pose_rgbinputp1' )

      pose_model=generate(pose_model_convnext,nlayersFreeze,"individualbranch_pose")
      
      pose_model0 = pose_model(pose_rgbinputp0)
      pose_model1 = pose_model(pose_rgbinputp1)

      the_concats.append(pose_model0)
      the_concats.append(pose_model1)
      the_inputs.append(pose_rgbinputp0)
      the_inputs.append(pose_rgbinputp1)


    pose_model_pair =  tf.keras.models.load_model(model_gcs_path,compile=False)
    i=0
    for layer in pose_model_pair.layers:
      layer._name = layer.name + str("_pair_pose") 
      if i > 0:
        if i <= nlayersFreeze:
          layer.trainable=False
      i=i+1
    
    the_concats.append(pose_model_pair.layers[-2].output)
    the_inputs.append(pose_model_pair.input)          
  

  if typeImg != 'RGB_Pose':
      if onlyPairRGB==True or onlyPairPose==True:  #Solo una rama ( solo pairpose o solo psirRGB, por lo cual no es necesario concatenate)
        concat=the_concats[0]
        the_inputs=the_inputs[0]
      else:
        concat =layers.concatenate(the_concats, name="concat_encods",axis=-1)
        concat = LayerNormalization()(concat)
  else:    
    concat =layers.concatenate(the_concats, name="concat_encods",axis=-1)
    concat = LayerNormalization()(concat)

  #  Add intermediate layers
  x = layers.Dense(4096, activation='relu', name='fc_1')(concat)
  x = layers.Dropout(0.5)(x)
  x = layers.Dense(4096, activation='relu', name='fc_2')(x)

  drp2 = layers.Dropout(0.5, name="top_dropout")(x)

  proxemicsoutput = layers.Dense(6, activation='sigmoid', name='output')(drp2)

  #  We generate the model
  finalModel= Model(inputs=the_inputs, outputs=proxemicsoutput)

  # Optimizer
  if optimizador=='Adam':
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
  elif optimizador=='AdamW':
    opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr)
  else:
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

  # Compile model
  finalModel.compile( loss='binary_crossentropy', optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])

  return finalModel


###################################################################################### ConvNeXt CrossAttention (features and constrastiveloss)  #######################
def create_projection_module(output_dim=512, name="projection_module"):
    projection_module = Sequential([
        layers.Dense(output_dim, name=f"{name}_dense1"),
        layers.BatchNormalization(name=f"{name}_batch_norm"),
        layers.Activation('tanh', name=f"{name}_activation"),
        layers.Dense(output_dim, name=f"{name}_dense2")
    ], name=name)
    return projection_module


def compute_modal_contrastive_loss(rgb, pose, temperature=0.1):
    # Normalizar los embeddings
    rgb_norm = tf.math.l2_normalize(rgb, axis=-1)
    pose_norm = tf.math.l2_normalize(pose, axis=-1)
    # Calcular la matriz de similitud coseno entre rgb y pose embeddings  (calculamos S(vN,aN)/temperature)
    #tf.matmul multiplica rgb de la primera muetsra con pose1 de su correspondiente muestra y la pose1 de todas las muestras del batch. genreando una matriz 
    sim_matrix = tf.matmul(rgb_norm, pose_norm, transpose_a=True) / temperature
    #v=K.eval(sim_matrix.output)
    #print(v)
    # Los índices diagonales representan las similitudes correctas
    sim_correct = tf.linalg.diag_part(sim_matrix)
    exp_sim_correct = tf.exp(sim_correct)   #Numerador
    
    # Sumar las exponenciales de todas las similitudes para el denominador
    sum_exp_sim_matrix = tf.reduce_sum(tf.exp(sim_matrix), axis=1)
    
    # Calcular la pérdida contrastiva
    contrastive_loss = -tf.math.log(exp_sim_correct / sum_exp_sim_matrix)
    return tf.reduce_mean(contrastive_loss)

def compute_total_contrastive_loss(branches_proyected_concat, onlyPairRGB, onlyPairPose, temperature=0.1):
    # Separamos nuestros embbedings en funcion de como sea nuestra red
    if onlyPairRGB and onlyPairPose: #solo pairs de rgb y pose (2 ramas)
      rgbpair,  posepair = tf.split(branches_proyected_concat, num_or_size_splits=2, axis=1)
      contrastive_loss = compute_modal_contrastive_loss(rgbpair, posepair, temperature)
    elif onlyPairRGB==False and onlyPairPose: #Full model RGB solo rama pair pose (4 ramas)
      _, _, rgbpair, posepair = tf.split(branches_proyected_concat, num_or_size_splits=4, axis=1)
      contrastive_loss = compute_modal_contrastive_loss(rgbpair, posepair, temperature)
    elif onlyPairRGB==True and onlyPairPose==False: #Solo rama pair RGB Y Full model pose (4 ramas)
      rgbpair, _ , _, posepair = tf.split(branches_proyected_concat, num_or_size_splits=4, axis=1)
      contrastive_loss = compute_modal_contrastive_loss(rgbpair, posepair, temperature)
    else:   #Full model (6 ramas)
      rgb1, rgb2, rgbpair, pose1, pose2, posepair = tf.split(branches_proyected_concat, num_or_size_splits=6, axis=1)
      # Calcular la pérdida contrastiva para cada modalidad
      loss_rgb1_pose1 = compute_modal_contrastive_loss(rgb1, pose1, temperature)
      loss_rgb2_pose2 = compute_modal_contrastive_loss(rgb2, pose2, temperature)
      loss_rgbpair_posepair = compute_modal_contrastive_loss(rgbpair, posepair, temperature)
      # Promediar las pérdidas de las tres modalidades
      contrastive_loss = (loss_rgb1_pose1 + loss_rgb2_pose2 + loss_rgbpair_posepair) / 3

    return contrastive_loss
  


# Get the convnext base model - crossAttention
def get_basemodel_convNext_crossAttention(model_gcs_path,lr,optimizador, typeImg, onlyPairRGB, onlyPairPose,use_contrastiveloss=False,beta=0.0,features=False):
  the_branches_concat=[]
  the_inputs=[]

  if 'RGB' in typeImg:
    if onlyPairRGB==False:
      base_model_convnext =  tf.keras.models.load_model(model_gcs_path,compile=False )

      rgbinputp0 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='rgbinputp0' )
      rgbinputp1 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='rgbinputp1' )
    
      base_model=generate(base_model_convnext,0,"individualbranch")
      
      base_model0 = base_model(rgbinputp0)
      base_model1 = base_model(rgbinputp1)

      # Crear e integrar el módulo de proyección a cada rama
      projection_module = create_projection_module(name="projection_module_individualbranch")
      projected_rgbinputp0 = projection_module(base_model0)
      projected_rgbinputp0=tf.expand_dims(projected_rgbinputp0, axis=1)    # Ahora la forma será [?, 1, 512] en lugar de [?,512] y podrá ser leido por el multiattention
      projected_rgbinputp1 = projection_module(base_model1)
      projected_rgbinputp1=tf.expand_dims(projected_rgbinputp1, axis=1) 

      the_branches_concat=[projected_rgbinputp0,projected_rgbinputp1]
      the_inputs=[rgbinputp0, rgbinputp1]


    base_model_convnext_pair =  tf.keras.models.load_model(model_gcs_path,compile=False)
    rgbinputpair = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='rgbinputpair' )    
    base_model_p=generate(base_model_convnext_pair,0,"pairbranch")
    base_model_pair = base_model_p(rgbinputpair)

    # Crear e integrar el módulo de proyección a cada rama
    projection_module = create_projection_module(name="projection_module_pairbranch")
    projected_rgbinputpair = projection_module(base_model_pair)
    projected_rgbinputpair=tf.expand_dims(projected_rgbinputpair, axis=1) 
    
    the_branches_concat.append(projected_rgbinputpair)
    the_inputs.append(rgbinputpair)
   
  if 'Pose' in typeImg:
    if onlyPairPose==False:
      pose_model_convnext =  tf.keras.models.load_model(model_gcs_path,compile=False )

      pose_rgbinputp0 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='pose_rgbinputp0' )
      pose_rgbinputp1 = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='pose_rgbinputp1' )

      pose_model=generate(pose_model_convnext,0,"individualbranch_pose")

      pose_model0 = pose_model(pose_rgbinputp0)
      pose_model1 = pose_model(pose_rgbinputp1)

       # Crear e integrar el módulo de proyección a cada rama
      projection_module_pose = create_projection_module(name="projection_module_individualbranch_pose")
      projected_pose_rgbinputp0 = projection_module_pose(pose_model0)
      projected_pose_rgbinputp0=tf.expand_dims(projected_pose_rgbinputp0, axis=1)    # Ahora la forma será [?, 1, 512] en lugar de [?,512] y podrá ser leido por el multiattention
      projected_pose_rgbinputp1 = projection_module_pose(pose_model1)
      projected_pose_rgbinputp1=tf.expand_dims(projected_pose_rgbinputp1, axis=1) 

      the_branches_concat.append(projected_pose_rgbinputp0)
      the_branches_concat.append(projected_pose_rgbinputp1)
      the_inputs.append(pose_rgbinputp0)
      the_inputs.append(pose_rgbinputp1)


    pose_model_convnext_pair =  tf.keras.models.load_model(model_gcs_path,compile=False)

    pose_rgbinputpair = tf.keras.layers.Input(shape=(224,224,3), dtype=tf.float32,name='pose_rgbinputpair' )    
    pose_model_p=generate(pose_model_convnext_pair,0,"pose_pairbranch")
    pose_model_pair = pose_model_p(pose_rgbinputpair)

    # Crear e integrar el módulo de proyección a cada rama
    projection_module_pose = create_projection_module(name="projection_module_pairbranch_pose")
    projected_pose_rgbinputpair = projection_module_pose(pose_model_pair)
    projected_pose_rgbinputpair=tf.expand_dims(projected_pose_rgbinputpair, axis=1) 
    
    the_branches_concat.append(projected_pose_rgbinputpair)
    the_inputs.append(pose_rgbinputpair)
  
  """
  if features: 
    features_size=(12, 768)
    features_input = tf.keras.layers.Input(shape=features_size, dtype=tf.float32, name='rgbinput_features')
    # Crear el modelo con una capa de entrada
    base_model_f = Sequential([
        layers.Dense(1024,activation='relu', name="features_dense"),
        layers.BatchNormalization(name="features_batch_norm"),
    ], name="features")
    base_model_features = base_model_f(features_input)
    # Añadir el módulo de proyección al final
    projection_module = create_projection_module(name="projection_module_featuresbranch") 
    projected_features = projection_module(base_model_features) 
    #projected_features=tf.expand_dims(projected_features, axis=1) 

    the_branches_concat.append(projected_features)
    the_inputs.append(features_input)
  """
  # Concatenate the projected features to form the query for cross-attention.
  branches_proyected_concat = tf.keras.layers.Concatenate(axis=1, name='branches_proyected_concat')(the_branches_concat)

  # Initialise MultiHeadAttention layer (CrossAttention)
  multi_head_attention_layer = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=512)

  # Calculate the cross Attention for each feature module.
  attention_layers=[]

  if 'RGB' in typeImg:
    if onlyPairRGB==False:
      attention_rgb_p0 = multi_head_attention_layer(query=branches_proyected_concat, value=projected_rgbinputp0, key=projected_rgbinputp0)
      attention_rgb_p1 = multi_head_attention_layer(query=branches_proyected_concat, value=projected_rgbinputp1, key=projected_rgbinputp1)
      attention_layers.append(attention_rgb_p0)
      attention_layers.append(attention_rgb_p1)
    attention_rgb_pair = multi_head_attention_layer(query=branches_proyected_concat, value=projected_rgbinputpair, key=projected_rgbinputpair)
    attention_layers.append(attention_rgb_pair)
  if 'Pose' in typeImg:
    if onlyPairPose==False:
      attention_pose_p0 = multi_head_attention_layer(query=branches_proyected_concat, value=projected_pose_rgbinputp0, key=projected_pose_rgbinputp0)
      attention_pose_p1 = multi_head_attention_layer(query=branches_proyected_concat, value=projected_pose_rgbinputp1, key=projected_pose_rgbinputp1)
      attention_layers.append(attention_pose_p0)
      attention_layers.append(attention_pose_p1)
    attention_pose_rgb_pair = multi_head_attention_layer(query=branches_proyected_concat, value=projected_pose_rgbinputpair, key=projected_pose_rgbinputpair)
    attention_layers.append(attention_pose_rgb_pair)
  '''
  if features:
    attention_rgb_features = multi_head_attention_layer(query=branches_proyected_concat, value=projected_features, key=projected_features)
    attention_layers.append(attention_rgb_features)
  '''

  # Sum the cross-attention outputs to obtain a unified feature (Fs).
  attention_layers_summed = tf.keras.layers.Add()(attention_layers)
  # Layer of Multi-Head Self-Attention
  multi_head_self_attention = tf.keras.layers.MultiHeadAttention(num_heads=3, key_dim=512)
  # Apply Self-Attention
  self_attention_output = multi_head_self_attention(query=attention_layers_summed, value=attention_layers_summed, key=attention_layers_summed)

  # Residual Connection (Fs + Fs_self_attention_output) CAF_tokens
  CAF_tokens = layers.Add()([attention_layers_summed, self_attention_output])

  # Average tokens
  CAF_tokens_avg_pooling = layers.GlobalAveragePooling1D()(CAF_tokens)

  # Final classification layer
  proxemicsoutput = layers.Dense(6, activation='sigmoid', name='output')(CAF_tokens_avg_pooling)  #proxemics
  #proxemicsoutput = layers.Dense(6, activation='softmax', name='output')(CAF_tokens_avg_pooling) #pisc

  # We generate the model
  finalModel= Model(inputs=the_inputs, outputs=proxemicsoutput)

  intermediate_layer_output = finalModel.get_layer('branches_proyected_concat').output

  #loss = tf.reduce_mean(tf.square(intermediate_layer_output))
  if use_contrastiveloss or beta!=0.0:
    loss=compute_total_contrastive_loss(intermediate_layer_output, onlyPairRGB, onlyPairPose, temperature=0.1)
    #finalModel.add_loss((loss/6.0)*0.5)
    finalModel.add_loss(loss*beta)

  #  Optimizer
  if optimizador=='Adam':
    opt = tf.keras.optimizers.Adam(learning_rate=lr)
  elif optimizador=='AdamW':
    opt = tf.keras.optimizers.experimental.AdamW(learning_rate=lr)
  else:
    opt = tf.keras.optimizers.SGD(learning_rate=lr)

  # Compile model
  if use_contrastiveloss:
    finalModel.compile( loss='binary_crossentropy', loss_weights={'output':1.4}, optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()])
  else:
    finalModel.compile( loss=tf.keras.losses.BinaryCrossentropy(name='binary_crossentropy'), optimizer=opt, metrics=[tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.AUC(num_thresholds=3),tf.keras.metrics.Precision()]) 
  return finalModel
