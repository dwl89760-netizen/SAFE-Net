# SAFE-NET
This repository contains the source code for our paper:

**"Deep Learning–Based Intracranial Aneurysm Segmentation Using Synergy Attention and Edge Guidance"**

The implementation is based on the [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) framework.  
Our contribution replaces the original nnU-Net network module with our proposed **Synergy Attention + Edge Guidance** model.

---

## Requirements

The installation and environment setup are **identical to nnU-Net**.  
Please follow the official nnU-Net installation guide:

- Python ≥ 3.8  
- PyTorch ≥ 1.10  
- CUDA (if using GPU acceleration)  
- Other dependencies: `numpy`, `scikit-learn`, `scipy`, `SimpleITK`, etc.

👉 Detailed installation instructions: [nnU-Net installation guide](https://github.com/MIC-DKFZ/nnUNet)

---

## Data

We conducted experiments on intracranial aneurysm datasets.  
Dataset access information and download links are provided in the **Data Availability Statement** of our paper.  

Once downloaded, please organize the data following the nnU-Net structure:[OpenNEURO link](https://openneuro.org/datasets/ds003949);[ADAM](https://adam.isi.uu.nl/data/)
