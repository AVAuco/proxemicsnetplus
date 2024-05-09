
# Proxemics-net++: classification of human interactions in still images


<div align="center">

   :page_facing_up: [Paper](https://link.springer.com/article/10.1007/s10044-024-01270-3) &nbsp; | &nbsp; [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1ueD8UFvWvFYK-MFL-GO3gteqIAXaM2LT?usp=sharing) 

</div>




&nbsp;


<p align="center">
    <img src="imgs/teaser.png" alt="Examples of human-human interactions." width="700">
</p>
<p align="center">
    <sub><strong>Figure 1: Examples of human-human interactions.</strong> These images illustrate the great complexity inherent in the problem of recognizing human interactions in images. The images in (a) highlight situations where it is confusing to determine the type of physical contact (hand-elbow, hand-shoulder, elbow-shoulder, etc.) due to clothing and partial occlusion. In (b), the images show ambiguity in determining the type of social relationship between individuals (family, friends, co-workers, etc.) without additional context.</sub>
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

### :chart_with_upwards_trend: Comparison to the State of the Art - Proxemics dataset
<div align="center">
   
  | **Model**                                                   | **HH** | **HS** | **SS** | **HT** | **HE** | **ES** | **mAP (a)** | **mAP (b)** |
   |------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------------|--------------|
   | [Yang et al.](https://doi.org/10.1109/CVPR.2012.6248095)                       | 37     | 29     | 50     | 61     | 38     | 34     | 42           | 38           |
   | [Chu et al.](https://doi.org/10.1109/ICCV.2015.383)                                | 41.2   | 35.4   | 62.2   | -      | 43.9   | 55     | -            | 46.6         |
   | [Jiang et al.](https://doi.org/10.1109/CVPR.2017.366)                               | 59.7   | 52     | 53.9   | 33.2   | 36.1   | 36.2   | 45.2         | 47.5         |
   | [Li W. et al.](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_2)                              | 56.7   | 55.1   | 52.8   | 78.4   | 65.0   | 65.5   | 62.3         | 59.1         |
   | [Sousa et al.](https://doi.org/10.1016/j.cviu.2023.103785)                            | 66.2   | 55.1   | 69.5   | 78.8   | 65.6   | 68.1   | 67.2         | 64.9         |
   | [Jiménez et al.](https://doi.org/10.1007/978-3-031-36616-1_32)                   | 62.4   | 56.7   | 62.4   | **86.4**| 68.8  | 67.9   | 67.4         | 63.8         |
   | **Our ConvNeXt_Base (CrossAttention) - (RGB+Pose - Full Model)**                               | **71.5**| **63.2**| **80.5**| 80.7  | **75.6**| **71.3**| **73.8**    | **72.4**     |

</div>

<p align="center">
<sub><strong>Table 1:</strong> Comparison of our best model obtained on the <strong>Proxemics dataset</strong> with the state of the art</sub>
</p>
&nbsp;

In this Table, two values of %mAP are compared: mAP(a) is the value of mAP explained in the previous sections (the mean of the AP values of the six types of proxemics) and mAP(b) is the mean of the AP values but excluding the Hand-Torso (HT) class as done in Chu et al.

The comparison shows that our best model (RGB+Pose with individual and pair branches, Base variant, and CrossAttention Fusion Block) achieves the highest %mAP results across almost all proxemics types, outperforming existing methods with significant improvements of 6.4% (mAP(a)) and 7.5% (mAP(b)). These results show that our combination of RGB and pose data, alongside a deep learning model like ConvNeXt, significantly enhances performance for proxemics recognition.

&nbsp;

### :chart_with_upwards_trend: Comparison to the State of the Art - PISC dataset
<div align="center">
   
| **Model**                                      | **Friends** | **Family** | **Couple** | **Prof.** | **Comm.** | **No Rel.** | **mAP** |
|------------------------------------------------|-------------|------------|------------|-----------|------------|-------------|---------|
| [Li J. et al.](https://doi.org/10.1007/s11263-020-01295-1)                    | 60.6        | 64.9       | 54.7       | 82.2      | 58         | 70.6        | 65.2    |
| [Zhang et al.](https://doi.org/10.1109/ICME.2019.00279)                       | 64.6        | 67.8       | 60.5       | 76.8      | 34.7       | 70.4        | 70.0    |
| [Goel et al.](https://doi.org/10.1109/CVPR.2019.01144)                         | -           | -          | -          | -         | -          | -           | 71.6    |
| [Li W. et al.](https://link.springer.com/chapter/10.1007/978-3-030-58555-6_2)                          | 60.8        | 65.9       | **84.8**   | 73.0      | 51.7       | 70.4        | 72.7    |
| [Li L. et al.](https://doi.org/10.1007/s00371-021-02244-w)                          | **82.2**    | 39.4       | 33.2       | 60.0      | 47.7       | 71.8        | 73.3    |
| [Yang et al.](https://doi.org/10.1109/ACCESS.2021.3096553)                      | 63.1        | 73.5       | 78.3       | **82.7**  | **76.8**   | 71.8        | 73.6    |
| [Sousa et al.](https://doi.org/10.1016/j.cviu.2023.103785)                         | 49.4        | 70.5       | 74.6       | 76.5      | 59.6       | 74.6        | **75.2** |
| **Our ConvNeXt_Base (Concat) (RGB - Full model)** | 56.2        | **83.9**   | 77.6       | 61.0      | 59.0       | **82.9**    | 70.1    |
</div>

<p align="center">
<sub><strong>Table 2:</strong> Comparison of our best model obtained on the <strong>PISC dataset</strong> with the state of the art</sub>
</p>
&nbsp;

Table 2 compares our best model with existing state-of-the-art approaches for social interaction recognition. Although our model (RGB model with individual and pairs branches, Base variant, and Concatenation Fusion Block) performs best in the "Family" and "No Relation" categories, it doesn't surpass the current best overall (70.1% mAP vs. 75.2% mAP).

Notably, most other methods rely on graph-based architectures, unlike ours, which uses a deep neural network. Compared with the PISC authors' deep neural network approach (Li J. et al.), we achieve better results (70.1% mAP vs 65.2% mAP), showing that RGB data combined with ConvNeXt architecture enhances social interaction recognition. However, recent trends in graph-based architectures suggest that this problem requires models focused on relationships due to its complexity.


&nbsp;

---

## :memo: Citing Proxemics-Net
If you find Proxemics-Net++ useful in your work, please consider citing the following BibTeX entry:
```bibtex
@article{Jimenez2024,
  author    = {I. Jiménez-Velasco and J. Zafra-Palma and R. Muñoz-Salinas and others},
  title     = {Proxemics-net++: classification of human interactions in still images},
  journal   = {Pattern Analysis and Applications},
  volume    = {27},
  number    = {1},
  pages     = {49},
  year      = {2024},
  doi       = {10.1007/s10044-024-01270-3}
}

