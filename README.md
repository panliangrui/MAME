

<div align="center">
  <a href="(https://github.com/panliangrui/DAEM/blob/main/images/distance.png)">
    <img src="https://github.com/panliangrui/DAEM/blob/main/images/distance.png" width="800" height="400" />
  </a>

  <h1>STAS Diagnostic Platform(https://113.219.237.106:34040/)</h1>

  
  <p>
  Liangrui Pan et al. is a developer helper.
  </p>

  <p>
    <a href="https://github.com/misitebao/yakia/blob/main/LICENSE">
      <img alt="GitHub" src="https://img.shields.io/github/license/misitebao/yakia"/>
    </a>
  </p>

  <!-- <p>
    <a href="#">Installation</a> | 
    <a href="#">Documentation</a> | 
    <a href="#">Twitter</a> | 
    <a href="https://discord.gg/zRC5BfDhEu">Discord</a>
  </p> -->

  <div>
  <strong>
  <samp>

[English](README.md)

  </samp>
  </strong>
  </div>
</div>

# Multimodal Multiscale Attention-Based Learning on Multicenter Lung Cancer CT and Histopathology Images Enhances STAS Diagnosis: A Multicenter Study

## Table of Contents

<details>
  <summary>Click me to Open/Close the directory listing</summary>

- [Table of Contents](#table-of-contents)
- [Feature Preprocessing](#Feature-Preprocessing)
- [Feature Extraction](#Feature-Extraction)
- [Models](#Train-models)
- [Test WSI](#Test WSI)
- [Datastes](#Datastes)
- [Installation](#Installation)
- [License](#license)

</details>

## Feature Preprocessing

Use the pre-trained model for feature preprocessing and build the spatial topology of WSI.
Upload the .svs file to the input folder

### Feature Extraction

Features extracted based on CTransPath.
Please refer to CTransPath: https://github.com/Xiyue-Wang/TransPath

Feature extraction code reference project: https://github.com/mahmoodlab/CLAM
```markdown
python create_patch_fp_256.py; create_patch_fp_512.py; run_wsi_inference.py
```
```markdown
python extract_features_fp_256.py; extract_features_fp_512.py;  summarize_tme_features.py
```
```markdown
python constract_graph_multi_view.py
```

## Models
**DAEM**

  <a href="(https://github.com/panliangrui/DAEM/blob/main/images/Figure1.jpg)">
    <img src="https://github.com/panliangrui/DAEM/blob/main/images/Figure1.jpg" width="912" height="1026" />
  </a>

Workflow of collection and organization of lung cancer STAS dataset, model training and inference, and multi-center validation. a, Resection of lung tumor tissue. b, Digitization of FSs and PSs. c, Histopathology image slice processing, d, Pathologists’ triple cross-validation to annotate data. e, Extraction of multi-scale histopathology image features and parallel expert models to diagnose and localize STAS in histopathology images. f, Details of the expert module, where AMP1d is the AdaptiveMaxPool1d layer. g, Details of the classifier. h, Distribution and quantity of multi-center data.



**Test WSI**
```markdown
python test_STAS.py
```

## Datastes

- Only features of the histopathology image data are provided as the data has a privacy protection agreement.
```markdown
Data access is available upon request via email(lip141772@gmail.com).
```
In this retrospective, multi-center study, we utilized anonymized hematoxylin and eosin (H&E) stained lung cancer slides from six hospitals and two projects in China and the United States, constructing nine cohorts for model training and validation. Based on our research objectives, only patients meeting the following criteria were included: (1) diagnosed with lung adenocarcinoma (LUAD); (2) corresponding routine pathological slices, including primary tumor tissue and adjacent non-tumor tissue; (3) detailed TNM staging; (4) high-quality slides, such as those without bending, wrinkling, blurring, or color changes; (5) absence of tumor cells randomly distributed, with irregular pericellular nests typically located at the tissue slice edge or outside the tissue slice; (6) absence of tumor cell continuity spreading from the tumor edge to the most distant airway; (7) absence of benign cytological features of lung epithelial cells or bronchial cells and/or presence of cilia; (8) absence of linear cellular strips detached from the alveolar wall in histopathological images. Two experienced pathologists, using microscopes and adhering to double-blind principles with cross-validation, labeled STAS (spread through air spaces) for each whole-slide image (WSI) to ensure accuracy and reduce subjectivity, missed diagnoses, or overdiagnosis. We included WSIs from the cohort of the Second Xiangya Hospital of Central South University (SXH-CSU) as internal training and validation data. This cohort selected 550 patients diagnosed with STAS and 170 patients without STAS from 12169 patients who underwent lung nodule resection at the Second Xiangya Hospital between April 2020 and December 2023. The experiment collected 2,494 WSIs (including 435 FSs and 2,057 PSs), immunohistochemical image data, and related clinical information from the selected patients.


The external validation set includes pathological images from eight centers. The cohort from the Third Xiangya Hospital of Central South University (TXH-CSU) provided 304 slides from 68 patients diagnosed with STAS between 2022 and 2023. The cohort from Xiangya hospital of Central South University (XH-CSU) provided 155 WSIs from 127 patients diagnosed with STAS between 2022 and 2023. The cohort from the Affiliated Tumor Hospital of Zhengzhou University (TH-ZZU) provided 91 WSIs from 19 patients diagnosed with LUAD and STAS in 2023. The cohort from the First Affiliated Hospital of Nanhua University (FAH-NHU) selected 130 WSIs from 42 patients diagnosed with LUAD and STAS between 2021 and 2024. The cohort from Changsha Jingkai Hospital (CJH) selected 91 WSIs from 45 patients diagnosed with LUAD and STAS between 2023 and 2024. The cohort from Pingjiang County First People's Hospital (PCPH) selected 78 WSIs from 35 patients diagnosed with LUAD and STAS between 2019 and 2021. The TCGA_LUAD dataset includes 366 patients with 417 WSIs. The CPTAC_LUAD cohort includes 170 patients with 443 WSIs. In the validation cohorts, TCGA_LUAD and CPTAC_LUAD provided patient-related clinical and multi-omics data, offering valuable information for survival and mechanistic studies of STAS patients. For detailed statistical data, please refer to the appendix.

## Installation
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on a single Nvidia GeForce RTX 4090)
- Python (3.7.11), h5py (2.10.0), opencv-python (4.1.2.30), PyTorch (1.10.1), torchvision (0.11.2), pytorch-lightning (1.5.10).

Note: You need to put the files in https://drive.google.com/drive/folders/1hk_i-1D48j0UN7mdqsw788J7treLSFoQ?usp=drive_link into the distance folder,
and put the folders in https://drive.google.com/drive/folders/1GC2EyO6PmYGEbkL15hqjfg0WfLZhs9sg?usp=drive_link, https://drive.google.com/drive/folders/1hk_i-1D48j0UN7mdqsw788J7treLSFoQ?usp=drive_link into the main project.

Note: Due to data security issues, we recommend downloading our distance measurement tool and using it locally.
## License
If you need the original histopathology image slides, please send a request to our email address. The email address will be announced after the paper is accepted. Thank you!

[License MIT](../LICENSE)
