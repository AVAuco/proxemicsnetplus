#!/bin/bash

for b in 6 12;	#size of dense
do
	for set in  1 2;	
	do
		echo '##############################################################################################################'
		echo '						NEW TRAINING'
		echo '##############################################################################################################'
		echo 'set ' $set ,  'Opt' $o , batch $b
		#!echo $1, $2, $3
		#!echo $1, $2, $3
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/"   python3  base_model_main_ViT.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/   --b $b  --o Adam --set  $set --lr 0.00005 --transformerType vit 
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
		TF_CPP_MIN_LOG_LEVEL=2  TF_ENABLE_ONEDNN_OPTS=0 WANDB_CACHE_DIR="/tmp/"   python3  base_model_main_ViT.py --datasetDIR /opt/data/isajim/experiments/proxemics/dataset/    --outModelsDIR /opt/data/isajim/experiments/proxemics/models/   --b $b  --o Adam --set  $set --lr 0.00005 --transformerType vit --onlyPairRGB
		rm -Rf wandb/
		rm -Rf /home/isajim/.local/share/wandb/
		rm -Rf /tmp/wandb/
		wandb artifact cache cleanup 1GB
	done
done
