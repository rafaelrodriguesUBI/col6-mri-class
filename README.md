
# Learning-based classification of Collagen VI-related muscular dystrophy

This repository contains the model architectures and pre-trained weights for the models evaluated in the paper: *"Learning-based classification of Collagen VI-related muscular dystrophy"*, currently under review. 

This repository provides the raw Keras/TensorFlow architectures (including standard CNNs, MLPs, and Wavelet Scattering-based models) and the weights learned using Leave-one-out cross-validation (LOOCV).

## Repo Structure

* `models.py`: Contains the raw `tf.keras.Model` build functions for all evaluated architectures.
* `demo.py`: A demonstration script showing how to instantiate the models, load specific LOOCV fold weights, and perform inference.
* `./weights/`: Empty directory to place the `.h5` weight files for each model. The pretrained weights are available in this [Google Drive folder](https://drive.google.com/drive/folders/1z3jehsJNK9j43bwR2AJMjzG-oUZRULwd?usp=sharing).

## Research Data Availability

Due to privacy constraints, the original dataset cannot be hosted publicly in this repository. 

The data can be made available upon reasonable request for validation purposes. If you wish to access the data, please contact the corresponding author at **rafael.rodrigues@ubi.pt**.

## Installation

We strongly recommend creating a [Conda](https://docs.conda.io/en/latest/) virtual environment for easier dependency management.

```bash
# Create conda environment (Python 3.10 recommended)
conda create -n your_env_name python=3.10
conda activate your_env_name

# Clone this repo
git clone [https://github.com/rafaelrodriguesUBI/col6-mri-class.git](https://github.com/rafaelrodriguesUBI/col6-mri-class.git)
cd col6_mri_class

# Install dependencies
pip install -r requirements.txt
```


## 📄 License and Citation

This work is licensed under a [Creative Commons Attribution-NonCommercial 4.0 International License](http://creativecommons.org/licenses/by-nc/4.0/). You are free to share and adapt this material for non-commercial purposes, provided you give appropriate credit.

**If you use these models and/or weights in your research, please cite the following papers:**

```bibtex
@inproceedings{rodrigues2022severity,
  title={Severity classification in cases of Collagen VI-related myopathy with Convolutional Neural Networks and handcrafted texture features},
  author={Rodrigues, Rafael and Quijano-Roy, Susana and Carlier, Robert-Yves and Pinheiro, Antonio MG},
  booktitle={2022 IEEE International Conference on Image Processing (ICIP)},
  pages={3596--3600},
  year={2022},
  organization={IEEE}
}
```
