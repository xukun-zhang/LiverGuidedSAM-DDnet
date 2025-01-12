# LiverGuidedSAM-DDnet

## Dataset Directory Structure

This document describes the file structure for the dataset used to train our network. Please organize your dataset files according to the structure below to ensure the network can correctly read and process the data.

## Directory Structure

The dataset is organized as follows, with the training set (Train) containing folders for images, labels, and liver regions:

### Train or Val or Test

- **images**: Contains all training image files in `.jpg` format. Filenames must match the corresponding label and liver region files.

- **labels**: Contains all training label images in `.png` format. Filenames must match the corresponding image files.

- **livers**: Contains all training liver region images in `.png` format. Filenames must match the corresponding image and label files.

---

## Quick Environment Setup

You can refer to [SAM](https://github.com/facebookresearch/segment-anything) or [D2GPLand](https://github.com/PJLallen/D2GPLand/tree/main) to configure the required environment.

To enable `SAM_twodecoder.py` to load the SAM encoder parameters, use the following code:

```python
from segment_anything import sam_model_registry

sam = sam_model_registry["vit_b"](
    checkpoint="sam_path/sam_vit_b_01ec64.pth"
)

