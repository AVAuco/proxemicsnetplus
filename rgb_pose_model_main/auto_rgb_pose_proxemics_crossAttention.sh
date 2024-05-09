#!/bin/bash
for bs in  6 ;
do 
	for set in 1 2;					#!Learning rate
	do
		echo '##############################################################################################################'
		echo '						NEW TRAINING'
		echo '##############################################################################################################'
		echo 'set ' $set ,  'Opt' $o , batch 6
		#!echo $1, $2, $3
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName proxemics --crossAttention --b $bs --rgb --pose 					# FULL MODEL (RGB+POSE)
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName proxemics --crossAttention --b $bs --rgb --pose  
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName proxemics --crossAttention  --b $bs --rgb --pose --onlyPairRGB   # RGB+POSE - OnlyPairRGB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName proxemics --crossAttention  --b $bs --rgb --pose --onlyPairRGB 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName proxemics --crossAttention  --b $bs --rgb --pose --onlyPairPose  # RGB+POSE - OnlyPairPose
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName proxemics --crossAttention --b $bs --rgb --pose --onlyPairPose 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName proxemics --crossAttention  --b $bs --rgb --pose --onlyPairRGB --onlyPairPose  # RGB+POSE - OnlyPairRGB - OnlyPairPose
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName proxemics --crossAttention  --b $bs --rgb --pose --onlyPairRGB --onlyPairPose 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName proxemics --crossAttention  --b $bs --rgb    #RGB MODEL
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName proxemics --crossAttention --b $bs --rgb   
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName proxemics --crossAttention  --b $bs --rgb  --onlyPairRGB   #RGB - OnlyPairRGB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName proxemics --crossAttention --b $bs --rgb --onlyPairRGB 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName proxemics --crossAttention --b $bs  --pose   #POSE MODEL
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName proxemics --crossAttention --b $bs --pose 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName proxemics --crossAttention --b $bs  --pose --onlyPairPose  #Pose - OnlyPairPOSE
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName proxemics --crossAttention --b $bs  --pose --onlyPairPose 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
	done
done
		
	

