# Solar Panel Defect Classification System

Official PyTorch Implementation | Paper in Submission Stage

Title： 《FVLS: A Novel Deep Learning Model for Solar Photovoltaic Panel Defect Identification》

Propose an improved Faster VIT model (FVLS) that combines learnable feature fusion (LFF) and spatial channel collaborative attention (SCSA) to achieve high-precision classification of multiple types of defects (cracks, poor contact, grid line breakage, corrosion) in photovoltaic panels, effectively suppressing complex background noise and multi-scale feature conflicts, and assisting in intelligent operation and maintenance of photovoltaic power plants and achieving dual carbon goals.

# 1. Research background and model positioning

This project aims to construct an efficient solar photovoltaic panel defect classification model utilizing deep learning technology. By adopting an enhanced FasterViT network architecture, combined with the SCSA (Spatial-Channel Self-Attention) module and the LFF multi-scale feature fusion module, it achieves automatic multi-label recognition of common defects (cracks, poor contact, interconnection faults, corrosion) on photovoltaic panels, providing a high-precision and lightweight solution for real-time quality inspection of photovoltaic panels.

# 2. Core innovation points

  ## 1\. Dual module collaborative attention and multi-scale feature fusion:

SCSA: Embedding SCSA modules at each stage, utilizing multi receptive field depth separable convolutions (kernel sizes 3, 5, 7, 9) and channel self attention to suppress complex background noise in photovoltaic panels and enhance weak signal response to small defects (such as early microcracks);

LFF: Utilizing multi-scale convolution to extract defect features at different scales in parallel (from pixel level cracks to centimeter level corrosion), and dynamically weighting fusion through learnable parameters, significantly improving the recall and classification accuracy of multi class and multi-scale defects.

  ## 2\. Hierarchical lightweight feature extraction architecture:

Based on the improved FasterVIT, the original Transformer attention is removed to reduce the number of parameters. A four level hierarchical downsampling structure is constructed using depthwise separable convolution, which extracts multi-level features from local edge details to global contextual semantics step by step, adapting to the complex grid texture of photovoltaic panels and defect areas of different sizes.

# 3. Balance between efficiency and accuracy:

The computational complexity is reduced from O (N ²) to O (N+D), the model parameter quantity is only 12.3M, and FLOPs is 6.5G. While achieving 98.02% of the overall recognition accuracy, the reasoning efficiency is significantly better than the traditional Vit model, which can be deployed in edge computing equipment (such as UAV patrol, handheld thermal imaging terminals) to support real-time intelligent operation and maintenance of photovoltaic power plants and dual carbon targets

  ## 3.1 Experimental dataset

This study is based on a self fusion dataset for defect detection of solar photovoltaic panels, The dataset has been uploaded to the dataset/folder in the warehouse along with the project, no additional download is required.

|Include categories|Total number of images|Data distribution (training: validation: testing)|
|-|-|-|
|Crack, Contact,	Interconnect,Corrosion|Please refer to the dataset description file for details|3:1:1|



  ## 3.2 Dataset structure

The dataset folder is organized as follows.
```
SolarPanelDefect/

 dataset/

&#x20;   ├── train/

&#x20;   │ ├── images/          # training set images

&#x20;   │ └── labels.csv       # training set labels

&#x20;   ├── val/

&#x20;   │ ├── images/          # val set images

&#x20;   │ └── labels.csv       # val set labels

&#x20;   └── test/

&#x20;   ├── images/            # test set images

&#x20;   └── labels.csv         # val set labels


```

# 4. Experimental environment configuration

  ## 4.1 Dependency Installation

Recommend using Anaconda to create virtual environments and ensure that the dependent versions match.



```

1. Create and activate a virtual environment

conda create -n fvls-pv python=3.10

conda activate fvls-pv



2. Install PyTorch and TorchVision (compatible with CUDA 11.8, CPU users can replace it with CPU version)

pip install torch==2.9.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118



3. Install other dependency libraries

pip install numpy\~=1.26.0 matplotlib\~=3.8.0 opencv-python\~=4.9.0

pip install pandas\~=2.2.0 pillow\~=10.2.0 tqdm\~=4.66.0 timm\~=1.0.8

pip install scikit-learn\~=1.4.0 seaborn\~=0.13.0
```

  ## 4.2 Hardware Requirements

GPU: Recommended NVIDIA GPU (graphics memory ≥ 8 GB, such as RTX 3060/4060/5060, supporting CUDA 11.8+), training for 100 rounds takes about 3-4 hours, with peak graphics memory usage ≤ 7 GB;

CPU: Only supports inference testing (single image inference takes about 0.2-0.5 seconds), not recommended for complete training.

Python 3.10.19

PyTorch == 2.9.1

\# Dataset acquisition and structure

The dataset should be organized in the following structure:


# 5. Experimental results
5.1 Comparison of Core Indicators
The performance comparison between FVLS model and mainstream deep learning models in the multi label defect classification task of photovoltaic panels is as follows. The model performs better in accuracy, computational efficiency, and long tail distribution processing:

|Category	|Precision (%)|Params (M)| FLOPs (G)|
|-|-|-|-|
|CNN-NAS	|91.74|12.30|2.90|
|EfficientNet-B0|97.71|5.30|0.39|
|AdvEL-Net|97.10|18.50|4.20|
|FVLS|98.02|12.3|6.5|



Note:

The accuracy is the global average accuracy of multiple labels on the test set (including the results after balanced sampling);

The recognition accuracy of rare categories (interconnect gate fracture, corrosion) reached 96.01% and 95.02%, respectively, which is about 5% higher than the baseline model;

FVLS reduces the computational complexity from O (N ²) of traditional Transformers to O (N+D) while maintaining high accuracy, and the inference speed meets the requirements of edge deployment.

Each category folder contains images of solar panel defects corresponding to that category.



# 6. Code usage instructions

  ## 6.1 Model Training

Run the train.exe script to start training, supporting configuration adjustment through parameters (adapted to multi label PV datasets):

Run the main. py script to start training, support adjusting training configuration through parameters.





Main configuration items (can be manually modified):

|Parameter name|Default value|
|-|-|
|data\_root|... (please modify to your own path)|
|img\_size|224|
|batch\_size|32|
|epochs|100|
|lr|0.001|



Automatically complete after running:

Load training/validation/test set

Train the model (output loss, acc, mAP for each round)

Automatically save the best model (with the highest mAP in the validation set) to output/dir/best\_madel.pth

Output the final test set report (Overall Accuracy, mAP, Precision, Recall)



# 7. Project file structure

SolarPanelDefect/

├── dataset/

├──examples/

├──main.py/

├── images/          

└── README.md      



# 8. Known issues and precautions

Multi label threshold selection: When predicting, the default threshold for defects is 0.5. If the actual application has strict requirements for false alarms of a certain type of defect, the threshold for that type can be appropriately increased (such as 0.7).

Image resolution: The input image will automatically scale to 320 × 320 (the model accepts size). If the original EL image resolution is too low (<100 × 100), small crack features may be lost. It is recommended to maintain a resolution of ≥ 256 × 256 during acquisition.

Data distribution offset: This model is trained on laboratory and public datasets. If there are significant differences in the EL imaging equipment, lighting, and noise characteristics of actual power plants, it is recommended to fine tune a small number of samples in the target domain first (freeze the first few layers and train for 20-30 rounds).

CUDA version issue: If you encounter CUDA incompatibility when installing PyTorch, you can downgrade to CUDA 11.8 or switch to the CPU version (training is extremely slow, only inference is recommended).

Memory usage: During training, if batch\_stize=16 and there is insufficient video memory, it can be reduced to 8 or 4, and the learning rate can be appropriately lowered
  
# 9. References and contact information

  ## 9.1 Reference Method

The paper is currently in the submission stage and will be updated to BiBTeX format after its official publication. Currently available for temporary reference:

```

@article{fvls\_pv\_defect\_2026,

&#x20; title={FVLS: A Novel Deep Learning Model for Solar Photovoltaic Panel Defect Identification},

&#x20; author={Xu, Laixiang and Liu, Chenshuo and Yang, Xiaomin and Wang, Weihao and Yang, Shengyuan and Zhao, Junmin},

&#x20; journal={（待录用后补充）},

&#x20; year={2026},

&#x20; note={Manuscript submitted for publication}

}

```

  ## 9.2 Contact Information



If you encounter code running issues or academic exchange needs, please contact:

Email: Email:liuchenshuohuuc@yeah.net

GitHub Issue: Submit an issue directly to this repository and we will respond within 1-3 business days.






