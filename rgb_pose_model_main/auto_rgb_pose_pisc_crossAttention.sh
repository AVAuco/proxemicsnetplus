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
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose 
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose  
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairRGB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairRGB 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairPose
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairPose 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairRGB --onlyPairPose
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.00005   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairRGB --onlyPairPose 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose 
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose  
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairRGB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairRGB 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairPose
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairPose 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairRGB --onlyPairPose
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 rgb_pose_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype base  --o Adam --set  $set --lr 0.0001   --datasetName pisc --crossAttention --contrastiveloss --b $bs --rgb --pose --onlyPairRGB --onlyPairPose 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
	done
done
		
	

