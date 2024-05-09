# proxemicsnetplus

# Proxemics-net++: classification of human interactions in still images


<div align="center">

   :page_facing_up: [Paper]([https://link.springer.com/chapter/10.1007/978-3-031-36616-1_32](https://link.springer.com/article/10.1007/s10044-024-01270-3)) &nbsp; | &nbsp; <img src="https://camo.githubusercontent.com/0b93f22ac70b7983e9915edf30ddc1a15713b2c310a214c2996dff49b410b949/68747470733a2f2f63646e2e646973636f72646170702e636f6d2f6174746163686d656e74732f3236373335363138303036343530313736302f3738313937313935303438363239303433322f476f6f676c655f436f6c61626f7261746f72792e737667" alt="Demo" width="24px" height="24px"/> [Demo](https://colab.research.google.com/drive/1ueD8UFvWvFYK-MFL-GO3gteqIAXaM2LT?usp=sharing)  

</div>


&nbsp;


<p align="center">
    <img src="imgs/teaser.png" alt="Examples of human-human interactions." width="700">
</p>
<p align="center">
    <sub><strong>Figure 1: Examples of human-human interactions.</strong> Could the reader indicate what kind of contact exists between the pairs in the examples in (a)? What kind of relationship exists between the people in (b)? These images illustrate the great complexity inherent in the problem of recognizing human interactions in images. The images in (a) highlight situations where it is confusing to determine the type of physical contact (hand-elbow, hand-shoulder, elbow-shoulder, etc.) due to clothing and partial occlusion. In (b), the images show ambiguity in determining the type of social relationship between individuals (family, friends, co-workers, etc.) without additional context.</sub>
</p>

&nbsp;

Human interaction recognition (HIR) is a significant challenge in computer vision that focuses on identifying human interactions in images and videos. HIR presents a great complexity due to factors such as pose diversity, varying scene conditions, or the presence of multiple individuals. Recent research has explored different approaches to address it, with an increasing emphasis on human pose estimation. In this work, we propose Proxemics-Net++, an extension of the Proxemics-Net model, capable of addressing the problem of recognizing human interactions in images through two different tasks: the identification of the types of “touch codes” or proxemics and the identification of the type of social relationship between pairs. To achieve this, we use RGB and body pose information together with the state-of-the-art deep learning architecture, ConvNeXt, as the backbone. We performed an ablative analysis to understand how the combination of RGB and body pose information affects these two tasks. Experimental results show that body pose information contributes significantly to proxemic recognition (first task) as it allows to improve the existing state of the art, while its contribution in the classification of social relations (second task) is limited due to the ambiguity of labelling in this problem, resulting in RGB information being more influential in this task.

&nbsp;
<p align="center">
    <img src="imgs/Proxemics-Net++.png" alt="Our Proxemics-Net++ model" width="700">
</p>
<p align="center">
    <sub><strong>Figure 2: Our Proxemics-Net++ model.</strong>  It consists of six inputs: three branches for the RGB information of the couple and the individuals that compose it (blue branches) and another three branches for the body pose representation of the two individuals and the couple (green branches).
        All branches have the same type of backbone (Base or Large). The outputs of these six branches are passed to a Fusion Block, which can be of two types: Concatenation fusion or CrossAttention fusion. Finally, the type of human interaction (proxemics or social relationship) of the input samples is predicted.</sub>
</p>


&nbsp;

### :chart_with_upwards_trend: Comparison to the State of the Art
<div align="center">
  
  | Model                        | HH   | HS   | SS   | HT   | HE   | ES   | mAP (a) | mAP (b) |
  |------------------------------|------|------|------|------|------|------|---------|---------|
  | [Yang et al.](https://doi.org/10.1109/CVPR.2012.6248095)                  | 37   | 29   | 50   | 61   | 38   | 34   | 42      | 38      |
  | [Chu et al.](https://doi.org/10.1109/ICCV.2015.383)                  | 41.2 | 35.4 | 62.2 | -    | 43.9 | 55   | -       | 46.6    |
  | [Jiang et al.](https://doi.org/10.1109/CVPR.2017.366)                 | 59.7 | 52   | 53.9 | 33.2 | 36.1 | 36.2 | 45.2    | 47.5    |
  | Our ViT                      | 48.2 | 57.6 | 50.4 | 76.6 | 57.6 | 52.8 | 57.2    | 53.3    |
  | Our ConvNeXt_Base            | 56.9 | **53.4** | 61.4 | 83.4 | 68.7 | 58.8 | 63.8    | 59.8    |
  | Our ConvNeXt_Large           | **62.4** | 56.7 | **62.4** | **86.4** | **68.8** | **67.9** | **67.4**    | **63.8**    |
</div>


<p align="center">
<sub><strong>Table 1:</strong> Table compares our three best models concerning the existing state of the art.</sub>
</p>
&nbsp;

In this Table, two values of %mAP are compared: mAP(a) is the value of mAP explained in the previous sections (the mean of the AP values of the six types of proxemics) and mAP(b) is the mean of the AP values but excluding the Hand-Torso (HT) class as done in Chu et al.

Looking at the table, we can see that our three proposed models (which use three branches as input) perform the best in both comparatives (mAP(a-b)), with the model that uses the ConvNeXt network as a backbone achieving the highest %mAP value (67.4% vs 47.5% mAP(a) and 63.8% vs 47.5% mAP(b)). Thus, we outperformed the existing state of the art by a significant margin, with improvements of up to 19.9% of %AP (mAP(a)) and 16.3% of %mAP (mAP(b)).

Therefore, these results demonstrate that the two state-of-the-art deep learning models (ConvNeXt and Vision Transformers) do help in the proxemics recognition problem since, using only RGB information, they can considerably improve the results obtained by all competing models.

&nbsp;

---

&nbsp;
## :rocket: What's new?

- `base_model_main/`: Main directory for the base model.
- `dataset/`: Directory containing the code necessary for dataset preprocessing.
- `ìmg/`: Directory containing the images of this work
- `test/`: Directory containing code and resources related to model testing.
- `train/`: Directory containing code and resources related to model training.
- `dataset_proxemics_IbPRIA.zip`: ZIP file containing the preprocessed dataset.
- `requirements.txt`: File specifying the necessary dependencies for the project.

&nbsp;

## :star2: Quick Start
###  :black_small_square: Installing Dependencies

To install the necessary dependencies to run this project, you can use the following command:

    conda create --name <env> --file requirements.txt

###  :black_small_square: Unzipping the Preprocessed Dataset ZIP

To use the preprocessed dataset, you must first unzip the `dataset_proxemics_IbPRIA.zip` file and place it two directories above the current directory. You can use the following command:

    unzip dataset_proxemics_IbPRIA.zip -d ../

###  :black_small_square: Downloading the pre-trained ConvNeXt models

To use the pre-trained ConvNeXt models that we have selected as a backbone to train our Proxemics-Net models, you need to download them from the following locations:

- Pre-trained Base model: [Download here](https://dl.fbaipublicfiles.com/convnext/convnext_base_22k_224.pth) (350MB)
- Pre-trained Large model: [Download here](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth) (800MB)

Once downloaded, you need to unzip them and place them one level above, i.e., in ../premodels/.

###  :black_small_square: Training a New Model

To train and test a new model, you should access the `base_model_main` directory and execute the following command lines based on the type of model you want to use:

- #### For Backbone Vit
  * Full Model (3 Branches)

         python3 base_model_main_ViT.py --datasetDIR <DIR dataset/> --outModelsDIR <DIR where you'll save the model> --b <batchsize> --set <set1/set2> --lr <learningRate>

  * Only Pair RGB
 
         python3 base_model_main_ViT.py --datasetDIR <DIR dataset/> --outModelsDIR <DIR where you'll save the model> --b <batchsize> --set <set1/set2> --lr <learningRate> --onlyPairRGB

- #### For Backbone ConvNeXt (Base or Large)

  * Full Model (3 Branches)
 
         python3 base_model_main_convNext.py --datasetDIR <DIR dataset/> --outModelsDIR <DIR where you'll save the model> --modeltype <base/large> --b <batchsize> --set <set1/set2> --lr <learningRate>

  * Only Pair RGB

         python3 base_model_main_convNext.py --datasetDIR <DIR dataset/> --outModelsDIR <DIR where you'll save the model> --modeltype <base/large> --b <batchsize> --set <set1/set2> --lr <learningRate> --onlyPairRGB


Be sure to adjust the values between <...> with the specific paths and configurations required for your project.

###  :black_small_square: Downloading the Proxemics-Net models we have trained
Here are 2 of the Proxemics-Net models that we have trained.

   * A model with ConvNeXt Large as backbone. This model has been the one with the best results (see SOTA table). It has been trained with RGB information of the individual clippings and the pairs (the 3 input branches). [Download here](https://drive.google.com/file/d/1spczxf4KHiWzvKCCueWia2cN19x1mZgP/view?usp=sharing) (4.45GB)
   * A model with ConvNext Base as backbone. This model has been trained only with the RGB information of the pairs. [Download here](https://drive.google.com/file/d/14K57Oy7EtByOxn00BRjJ3f5XfCsh2Q3m/view?usp=sharing) (1.13GB)

   
   :star_struck: **You can test these models in the Google Colab [Demo](https://colab.research.google.com/drive/1ueD8UFvWvFYK-MFL-GO3gteqIAXaM2LT?usp=sharing) we have prepared for you.**

&nbsp;
## :memo: Citing Proxemics-Net
If you find Proxemics-Net++ useful in your work, please consider citing the following BibTeX entry:
```bibtex
@InProceedings{jimenezVelasco2023,
   author = "Jiménez, I. and Muñoz, R. and Marín, M. J.",
   title = "Proxemics-Net: Automatic Proxemics Recognition in Images",
   booktitle = "Pattern Recogn. Image Anal.",
   year = "2023",
   pages = "402-413",
   note= "IbPRIA 2023",
   doi = "10.1007/978-3-031-36616-1_32"
}
