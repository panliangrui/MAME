

<div align="center">
  <a href="(https://github.com/panliangrui/MAME/blob/main/STAS.png)">
    <img src="https://github.com/panliangrui/MAME/blob/main/STAS.png" width="800" height="400" />
  </a>

  <h1>Pathological features of STAS</h1>

  
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

Features extracted based on Prov-GigaPath and CONCHv1.5.
Please refer to Prov-GigaPath and CONCHv1.5: https://huggingface.co/prov-gigapath/prov-gigapath and https://huggingface.co/MahmoodLab/conchv1_5

Feature extraction code reference project: https://github.com/mahmoodlab/TRIDENT

**Or follow step-by-step instructions:**

**Step 1: Tissue Segmentation:** Segments tissue vs. background from a dir of WSIs
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task seg --wsi_dir ./wsis --job_dir ./trident_processed --gpu 0 --segmenter hest
   ```
   - `--task seg`: Specifies that you want to do tissue segmentation.
   - `--wsi_dir ./wsis`: Path to dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--gpu 0`: Uses GPU with index 0.
   - `--segmenter`: Segmentation model. Defaults to `hest`. Switch to `grandqc` for fast H&E segmentation. Add the option `--remove_artifacts` for additional artifact clean up.
 - **Outputs**:
   - WSI thumbnails in `./trident_processed/thumbnails`.
   - WSI thumbnails with tissue contours in `./trident_processed/contours`.
   - GeoJSON files containing tissue contours in `./trident_processed/contours_geojson`. These can be opened in [QuPath](https://qupath.github.io/) for editing/quality control, if necessary.

 **Step 2: Tissue Patching:** Extracts patches from segmented tissue regions at a specific magnification.
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task coords --wsi_dir ./wsis --job_dir ./trident_processed --mag 20 --patch_size 256 --overlap 0
   ```
   - `--task coords`: Specifies that you want to do patching.
   - `--wsi_dir wsis`: Path to the dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--mag 20`: Extracts patches at 20x magnification.
   - `--patch_size 256`: Each patch is 256x256 pixels.
   - `--overlap 0`: Patches overlap by 0 pixels (**always** an absolute number in pixels, e.g., `--overlap 128` for 50% overlap for 256x256 patches.
 - **Outputs**:
   - Patch coordinates as h5 files in `./trident_processed/20x_256px/patches`.
   - WSI thumbnails annotated with patch borders in `./trident_processed/20x_256px/visualization`.

 **Step 3a: Patch Feature Extraction:** Extracts features from tissue patches using a specified encoder
 - **Command**:
   ```bash
   python run_batch_of_slides.py --task feat --wsi_dir ./wsis --job_dir ./trident_processed --patch_encoder uni_v1 --mag 20 --patch_size 256 
   ```
   - `--task feat`: Specifies that you want to do feature extraction.
   - `--wsi_dir wsis`: Path to the dir with your WSIs.
   - `--job_dir ./trident_processed`: Output dir for processed results.
   - `--patch_encoder uni_v1`: Uses the `UNI` patch encoder. See below for list of supported models. 
   - `--mag 20`: Features are extracted from patches at 20x magnification.
   - `--patch_size 256`: Patches are 256x256 pixels in size.
 - **Outputs**: 
   - Features are saved as h5 files in `./trident_processed/20x_256px/features_uni_v1`. (Shape: `(n_patches, feature_dim)`)

```
```markdown
python constract_graph.py
```

## Models
**MAME**

  <a href="(https://github.com/panliangrui/MAME/blob/main/Figure%201.jpg)">
    <img src="https://github.com/panliangrui/MAME/blob/main/Figure%201.jpg" width="912" height="1026" />
  </a>

Workflow for constructing the MAME model to predict STAS in patients with lung cancer using multimodal data. a, Acquisition and annotation pipeline for preoperative CT scans. b, Workflow from surgical resection specimens to digitized histopathology WSIs. c, Multimodal modeling pipeline for predicting STAS status in lung cancer patients. d, Explainability analysis of the multimodal model. e, Identification of auxiliary biomarkers enabled by the multimodal model. f, Distribution of multimodal samples across the multicenter cohorts.



**Train WSI**
```markdown
python train_clam_mb.py
```

## Datastes

- Only features of the histopathology image data are provided as the data has a privacy protection agreement.
```markdown
Data access is available upon request via email(lip141772@gmail.com).
```
In this retrospective, multicenter study, we used anonymized hematoxylin and eosin (H&E) lung cancer slides and matching CT images from five cohorts across five independent hospitals in China for model training and validation. The patient inclusion and exclusion process for the multicenter study is shown in Figure 1. The detailed inclusion and exclusion criteria for WSI data are provided in Supplementary Table 1. To ensure the accuracy of STAS diagnosis, three experienced pathologists performed labeling of each WSI under a double-blind cross-validation process to minimize misdiagnosis and missed diagnoses. Additionally, the use of immunomarkers such as TTF-1, CK, and CD68 further ensured the distinction between STAS spread and tissue cells. Subsequently, through the inclusion and exclusion criteria for CT imaging data (Supplementary 1.2), we ensured the consistency and reliability of the CT and pathological data, providing high-quality samples for multimodal data fusion analysis. The data in this study were sourced from the Second Xiangya Hospital of Central South University (SXH-CSU) cohort. A total of 280 STAS-diagnosed patients and 282 non-STAS patients were selected, with one representative PS chosen for each patient. In total, 562 PS slides, 562 CT image sets, relevant immunohistochemical data, and clinical information were collected. However, compared to PS, the number of FS lung cancer patients meeting both the inclusion/exclusion criteria for patients and multimodal data inclusion/exclusion criteria was smaller. Among them, 152 STAS patients and 107 non-STAS patients contributed 339 FSs. Each patient may have one or more available FS (some patients had multiple nodules), and each FS was matched with a single CT image set.

The external validation dataset comprised preoperative CT imaging and PSs collected from four independent medical centers. For each patient, one CT image series and one PS were selected according to predefined inclusion and exclusion criteria for patient enrollment, WSI quality and CT acquisition. The XH-CSU cohort (Xiangya Hospital, Central South University) contributed 144 STAS and 59 non-STAS lung cancer patients diagnosed between August 2022 and June 2023, with one PS and 153 CT series included for each patient. The TXH-CSU cohort (The Third Xiangya Hospital of Central South University) provided PSs and CT images from 11 patients with pathologically confirmed STAS lung cancer and 52 non-STAS patients diagnosed between March 2022 and May 2023. The FAH-NHU cohort (The First Affiliated Hospital, Hengyang Medical School, University of South China) contributed 69 PSs and 69 CT series from 18 STAS and 51 non-STAS patients with confirmed LUAD diagnosed between 2021 and 2024. The PCPH cohort (Pingjiang County People’s Hospital) provided 66 PSs and 66 CT series from 18 STAS and 48 non-STAS LUAD patients diagnosed between 2019 and 2024. Summary statistics for all cohorts are provided in the Supplementary Tables 1.


## Installation
- Linux (Tested on Ubuntu 18.04)
- NVIDIA GPU (Tested on a single Nvidia GeForce RTX 4090)
- Python (3.7.11), h5py (2.10.0), opencv-python (4.1.2.30), PyTorch (1.10.1), torchvision (0.11.2), pytorch-lightning (1.5.10).

Note: The complete code will be made public after the paper is published.
## License
If you need the original histopathology image slides, please send a request to our email address. The email address will be announced after the paper is accepted. Thank you!

[License MIT](../LICENSE)
