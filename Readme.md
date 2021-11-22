# Image Segmentation Using Text and Image Prompts
This repository contains the code used in the paper "Image Segmentation Using Text and Image Prompts".


### Dependencies
This code base depends on pytorch, torchvision and clip (`pip install git+https://github.com/openai/CLIP.git`).
Additional dependencies are hidden for double blind review.

### Quick Start

```python
from models.clipseg import CLIPDensePredT
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt

# load model
model = CLIPDensePredT()
model.eval()
# non-strict, because we only stored decoder weights (not CLIP weights)
p.load_state_dict(torch.load('logs/rd64-uni/weights.pth'), strict=False)

# load and normalize image
input_image = Image.open('example_image.jpg')
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.Resize((352, 352)),
])
img = transform(input_image).unsqueeze(0)

# predict
with torch.no_grad():
    preds = p(img_in.repeat(4,1,1,1), ['a glass', 'cutlery', 'wood', 'a jar'])[0]

# visualize prediction
_, ax = plt.subplots(1, 5)
[a.axis('off') for a in ax.flatten()]
ax[0].imshow(input_image)
[ax[i+1].imshow(torch.sigmoid(preds[i][0])) for i in range(4)]
```


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

