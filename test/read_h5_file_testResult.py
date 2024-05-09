# ====================================================================
#Code for reading the test result of a Proxemics model
# ====================================================================

import sys
import os

import argparse

sys.path.append("..")

import deepdish as dd



def parse_args():
    parser = argparse.ArgumentParser(description='Test script.')

    parser.add_argument('--modelsdir',type=str,  help='models file dir ', required=True, default="")	
    parser.add_argument("--resultstype", help="type of results", choices=['original', 'chkt', 'chkt_perImage'], required=True)

    return parser.parse_args()



# =================================================================================


if __name__ == '__main__':
    args = parse_args()
    #print(args)

    dirpath = args.modelsdir
    resultsType=args.resultstype
    #File where the results of all the models will be stored
    if resultsType == 'original':
        filename_text= os.path.join(dirpath,"results_all_models.txt")
    else:
        filename_text= os.path.join(dirpath,("results_all_models_"+resultsType+".txt"))
    f = open(filename_text, 'w')	

    #Reads each directory that is hosted within the modeldir directory
    for files in os.listdir(dirpath):   
        #Model to be load
        modelpath= os.path.join(dirpath,files)
        
        #File with the results of the model
        if resultsType == 'original':
            resultsfile=os.path.join(modelpath,"best_keras_model_results.h5")
        else:
            resultsfile=os.path.join(modelpath,"best_keras_model_results_"+resultsType+".h5")

        
        if os.path.isfile(resultsfile):
            print('############################# New Model #############################')
            f.write('############################# New Model #############################\n')
            print("Model : ",files)
            f.write("Model : ")
            f.write(files)
            f.write('\n')
            print('#####################################################################')
            f.write('#####################################################################\n')
            
            # We open for each model the results file that we have already calculated when doing the test (model/.h5)
            AP = dd.io.load(resultsfile)
            
            if AP is None:
                print("Error reading test file!!!")
                continue
            


            #Print AP results
            print()
            print("- AP results:")
            print('\t--> HAND - HAND : ', AP['HAND_HAND'])
            print('\t--> HAND - SHOULDER : ', AP['HAND_SHOULDER'])
            print('\t--> SHOULDER - SHOULDER : ', AP['SHOULDER_SHOULDER'])
            print('\t--> HAND - TORSO : ', AP['HAND_TORSO'])
            print('\t--> HAND - ELBOW : ',AP['HAND_ELBOW'])
            print('\t--> ELBOW - SHOULDER : ', AP['ELBOW_SHOULDER'])

            print()
            print("- mAP : " , AP['mAP'])

            #Guardamos en un .txt
            f.write('- AP results: ')
            f.write('\n')
            f.write('\t--> HAND - HAND : ')
            f.write(str( AP['HAND_HAND']))
            f.write('\n')
            f.write('\t--> HAND - SHOULDER : ')
            f.write(str( AP['HAND_SHOULDER']))
            f.write('\n')
            f.write('\t--> SHOULDER - SHOULDER : ')
            f.write(str( AP['SHOULDER_SHOULDER']))
            f.write('\n')
            f.write('\t--> HAND - TORSO : ')
            f.write(str( AP['HAND_TORSO']))
            f.write('\n')
            f.write('\t--> HAND - ELBOW : ')
            f.write(str( AP['HAND_ELBOW']))
            f.write('\n')
            f.write('\t--> ELBOW - SHOULDER : ')
            f.write(str( AP['ELBOW_SHOULDER']))
            f.write('\n')
            f.write('\n')
            f.write(' - mAP : ')
            f.write(str( AP['mAP']))
            f.write('\n\n')
       
        else:
            print('\n > ERROR : En',files, 'no se encuentra result.h5\n')

    
    f.close()




