#!/bin/bash

for mo in base large;					#!Learning rate
do
	for set in 1 2;
	do
		for bs in 6 ;				#!Optimizador	
		do
			echo '##############################################################################################################'
			echo '						NEW TRAINING'
			echo '##############################################################################################################'
			echo 'set ' $set ,  'Opt' $o , batch 8
			rm -Rf wandb/
			rm -Rf /home/isajim/.local/share/wandb/
			rm -Rf /tmp/wandb/
			wandb artifact cache cleanup 1GB
			TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 base_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype $mo --b $bs  --o Adam --set  $set --lr 0.0001  --datasetName pisc  
			TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 base_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype $mo --b $bs  --o Adam --set  $set --lr 0.0001  --datasetName pisc  

			rm -Rf wandb/
			rm -Rf /home/isajim/.local/share/wandb/
			rm -Rf /tmp/wandb/
			wandb artifact cache cleanup 1GB
			TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 base_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype $mo --b $bs  --o Adam --set  $set --lr 0.00001  --datasetName pisc 
			TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 base_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype $mo --b $bs  --o Adam --set  $set --lr 0.00001  --datasetName pisc  

			rm -Rf wandb/
			rm -Rf /home/isajim/.local/share/wandb/
			rm -Rf /tmp/wandb/
			wandb artifact cache cleanup 1GB
			TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 base_model_main_convNext.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype $mo --b $bs  --o Adam --set  $set --lr 0.00005  --datasetName pisc 
			TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/" python3 base_model_main_convNext_chkt_and_perImg.py --datasetDIR /opt/data/isajim/experiments/pisc/dataset/    --outModelsDIR /opt/data/isajim/experiments/pisc/models/  --modeltype $mo --b $bs  --o Adam --set  $set --lr 0.00005  --datasetName pisc 

		done
	done
done
