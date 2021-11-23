# Image Segmentation Using Text and Image Prompts
This repository contains the code used in the paper "Image Segmentation Using Text and Image Prompts".


### Dependencies
This code base depends on pytorch, torchvision and clip (`pip install git+https://github.com/openai/CLIP.git`).
Additional dependencies are hidden for double blind review.

### Quick Start

We provide binder notebook. 


### Datasets

* `PhraseCut` and `PhraseCutPlus`: Referring expression dataset
* `PFEPascalWrapper`: Wrapper class for PFENet's Pascal-5i implementation
* `PascalZeroShot`: Wrapper class for PascalZeroShot
* `COCOWrapper`: Wrapper class for COCO.

### Models

* `CLIPDensePredT`: CLIPSeg model with transformer-based decoder.
* `ViTDensePredT`: CLIPSeg model with transformer-based decoder.

### Third Party Dependencies
For some of the datasets third party dependencies are required. Run the following commands in the `third_party` folder.
`git clone https://github.com/cvlab-yonsei/JoEm`
`git clone https://github.com/Jia-Research-Lab/PFENet.git`
`git clone https://github.com/ChenyunWu/PhraseCutDataset.git`
`git clone https://github.com/juhongm999/hsnet.git`

### Weights
CLIPSeg-rd64, CLIPSeg-rd16


### Training

See the experiment folder for yaml definitions of the training configurations. The training code is in `experiment_setup.py`.

### Usage of PFENet Wrappers

In order to use the dataset and model wrappers for PFENet, the PFENet repository needs to be cloned to the root folder.
`git clone https://github.com/Jia-Research-Lab/PFENet.git `

