# LiverGuidedSAM-DDnet

## Overview

LiverGuidedSAM-DDnet is a novel segmentation framework designed for anatomical landmark detection in augmented reality (AR)-guided laparoscopic liver surgeries. By leveraging liver segmentation as a spatial reference and incorporating pre-trained SAM encoder features, our method improves the accuracy and robustness of landmark segmentation, even in complex surgical environments.

---

## Task Challenges

### Difficulties in AR-Guided Surgery and Laparoscopic Environments

AR navigation in laparoscopic liver surgery relies heavily on accurate 2D-to-3D registration, which demands precise anatomical landmark segmentation. However, the following challenges significantly hinder the segmentation process:  
1. **Complex Surgical Environment:**  
   - Variable lighting conditions, including specular reflections from wet tissues.  
   - Frequent occlusions caused by surgical instruments and surrounding tissues.  
   - Deformations of the liver due to breathing, manipulation, and bleeding.  

2. **Challenges in Landmark Segmentation:**  
   - Anatomical landmarks such as the silhouette, falciform ligament, and liver ridge are often subtle and small, making them difficult to detect.  
   - Traditional approaches struggle to capture the **intrinsic spatial relationships** between landmarks and liver morphology, resulting in suboptimal segmentation accuracy.

---

## Surgeon-Inspired Motivation

This study is motivated by the way expert surgeons intuitively interpret liver appearance and structure, even under challenging laparoscopic conditions. Surgeons rely on:  
1. **Spatial Understanding:**  
   - By observing the overall liver morphology, they can infer the locations of subtle landmarks.  
2. **Robustness to Noise:**  
   - Through extensive experience, surgeons can disregard irrelevant features such as specular highlights or tissue occlusions.  

Our goal is to replicate this human expertise by designing a segmentation framework that integrates **spatial consistency** and **robust feature extraction**.

---

## Key Innovations

### Innovative Design Inspired by Surgeons

To emulate the expertise of surgeons, our framework incorporates several innovative components:

1. **Dual-Decoder Architecture:**  
   - Addresses **feature entanglement** by independently segmenting the liver and landmarks, enabling task-specific optimization.  
   - Liver segmentation provides a reliable spatial reference, guiding the accurate localization of subtle landmarks.

2. **SAM Encoder Integration:**  
   - The pre-trained **Segment Anything Model (SAM)** encoder is employed to leverage its robust feature extraction capabilities, trained on large-scale datasets.  
   - SAM enhances adaptability to laparoscopic challenges such as occlusions and variable lighting.

3. **Liver-Guided Consistency Constraint:**  
   - A fine-grained **internal-external consistency mechanism** enforces spatial alignment between the liver and landmarks, mimicking the logical spatial reasoning process of surgeons.  
   - This ensures that landmarks remain consistent with liver anatomy, even in challenging cases.

---

## Dataset Directory Structure

The dataset used for training and evaluation is structured as follows:

### Train or Val or Test

- **images**: Contains all laparoscopic image files in `.jpg` format. Filenames must match the corresponding label and liver region files.  
- **labels**: Contains ground truth label images for landmarks in `.png` format. Filenames must match the corresponding image files.  
- **livers**: Contains ground truth liver region images in `.png` format. Filenames must match the corresponding image and label files.

---

## Quick Environment Setup

You can refer to [SAM](https://github.com/facebookresearch/segment-anything) or [D2GPLand](https://github.com/PJLallen/D2GPLand/tree/main) for setting up the required environment.

To enable `SAM_twodecoder.py` to load the SAM encoder parameters, use the following code:

```python
from segment_anything import sam_model_registry

sam = sam_model_registry["vit_b"](
    checkpoint="sam_path/sam_vit_b_01ec64.pth"
)

self.sam_encoder = sam.image_encoder

