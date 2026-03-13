# README

## Data Availability

Because the dataset used in this project is extremely large, most of the
files cannot be uploaded directly to GitHub. All datasets are therefore
hosted on Google Drive.

If you need to download the data to reproduce the workflow, please use
the following link:

https://drive.google.com/drive/folders/1PNu-T7o5fGCpUrXjBqwvhPJ-VcXsABsX?usp=drive_link

Due to the size of the dataset and the computational cost of the
workflow, many scripts require **hours or even days to complete**. To
make the project easier to explore, the `.ipynb` notebooks containing
the **outputs of each step** are also provided so that users can review
intermediate results without rerunning the full pipeline.

------------------------------------------------------------------------

# 1. Data Generation and Preparation

## 1.1 Data Generation (Matlantis)

The data generation process is performed using **Matlantis**, which is a
licensed platform. Therefore, some packages and functions used in the
following notebooks may **not run in a standard local Python
environment**.

The following notebooks generate `.extxyz` datasets under different
compositions and conditions using **NPT molecular dynamics
simulations**.

    data/random_variation/Generate_random_global.ipynb
    data/random_variation/Generate_random_local.ipynb
    data/diff_com_T_OV/Generate_data_MD.ipynb

These notebooks generate structural configurations for different
materials, temperatures, and oxygen vacancy conditions.

------------------------------------------------------------------------

## 1.2 Data Splitting, FPS Selection, and Visualization

    data/diff_com_T_OV/npt_final/data_process.ipynb

Because the dataset is extremely large, some steps are computationally
expensive:

-   PCA fitting alone takes **more than 10 hours**
-   Running the entire notebook may take **around two days**

This notebook performs the following steps:

1.  Split the generated dataset into **training, validation, and test
    sets**
2.  Compute **SOAP descriptors**
3.  Perform **dimensionality reduction** (PCA)
4.  Visualize the distribution of atomic environments in **2D space**
5.  Apply **Farthest Point Sampling (FPS)**

FPS is used to select **10% of data points from each material**,
reducing redundancy in MD-generated configurations and avoiding
unnecessary computational cost caused by highly similar structures.

------------------------------------------------------------------------

## 1.3 Visualization of Data Generation Methods

    data/diff_com_T_OV/npt_final/UMAP_PCA_mixed.ipynb

This notebook compares two different data generation methods:

-   Random global / random local perturbations
-   Standard MD-generated structures

The visualization demonstrates that **random perturbation methods
generate a much wider distribution of configurations**, while MD
trajectories tend to produce more clustered structures.

------------------------------------------------------------------------

## 1.4 Validation Dataset Trimming

    data/diff_com_T_OV/npt_final/val_ood_r00.ipynb

Originally, the validation dataset contained **three random
configurations for each composition and condition**, which significantly
increased training time.

Running one training cycle on Sherlock with the full validation set
could take **several days**.

To reduce unnecessary computational cost, the validation set was
manually trimmed to include **only one configuration per condition**,
instead of three.

------------------------------------------------------------------------

## 1.5 Combine Datasets

    data/random_variation/mixed_data_combine.ipynb

This notebook combines:

-   **FPS-selected MD training data**
-   **Random variation data**

It also creates a **mixed validation dataset**, which merges:

-   random perturbation structures
-   trimmed MD validation structures

------------------------------------------------------------------------

# 2. Training Preparation (Sherlock)

## 2.1 Environment Setup

Create a new environment with the following dependencies:

-   Python
-   PyTorch
-   MACE
-   Conda

This environment will be used for model training on Sherlock.

------------------------------------------------------------------------

## 2.2 Upload Dataset

Upload the following files to Sherlock:

    train_mixed.extxyz
    val_mixed.extxyz
    test_ood_r00.extxyz

These files will be used for training, validation, and testing.

------------------------------------------------------------------------

# 3. Model Training

## 3.1 Run Training

    run_mace_train.sh

This script contains all required commands and parameters to train the
MACE model and submit the job to Sherlock.

After training is completed, the following file will be generated:

    full_model.model

This file is the **trained MACE model**.

------------------------------------------------------------------------

## 3.2 Monitor Loss and MAE

    epoch_loss.py

This script reads the training log and plots:

-   Training loss vs epoch
-   Validation loss vs epoch
-   Force MAE vs epoch

This is used to monitor training progress and **avoid overfitting**,
allowing the training to stop at an appropriate epoch.

------------------------------------------------------------------------

## 3.3 Test the Model

    test_model.py

Because the test dataset is also large, it must be **split into multiple
files** before evaluation. Otherwise, the evaluation step may run out of
memory.

The script:

1.  Splits the test dataset into **10 chunks**
2.  Runs predictions separately
3.  Computes final metrics

After running the script, the following metrics will be printed:

-   Energy MAE
-   Energy RMSE
-   Per-atom Energy MAE
-   Force MAE
-   Force RMSE

------------------------------------------------------------------------

# 4. Molecular Dynamics Using the Trained Model

## 4.1 Run MD with MACE

    MACE_training/final/mixed_MD/step1_build_structures.py
    MACE_training/final/mixed_MD/step2_mace_sanity_check.py
    MACE_training/final/mixed_MD/step3_short_npt_test.py
    MACE_training/final/mixed_MD/step4_matrix_T_Ov_volume.py

These scripts are used to run **MD simulations using the trained MACE
model** under the target conditions.

The goal is to compute and compare the **Thermal Expansion Coefficient
(TEC)** between two compositions:

-   HE_3
-   LE_1

The simulations cover temperatures from **26°C -- 800°C** using **NPT
molecular dynamics**.

------------------------------------------------------------------------

## 4.2 TEC Calculation

    MACE_training/final/mixed_MD/mace_T_Ov_matrix/pretrain/step4_pretrained_selected_conditions.py

A **pretrained MACE model** is used to simulate selected conditions for
selected compositions.

After running the MD simulations, the resulting structural properties
are saved in:

    ./pretrained_mace_selected_conditions/tables

These tables contain:

-   volume
-   temperature
-   oxygen vacancy conditions

The data are then used to compute the **Thermal Expansion Coefficient
(TEC)**.
