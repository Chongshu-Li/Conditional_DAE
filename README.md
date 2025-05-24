# Neurological Anomaly Detection in MRI Scans through Deep Learning: A Healthy Cohort Training Approach

This repository contains the code used for training and evaluating conditional denoising autoencoder (cDAE) on the BraTS 2021 dataset, as part of a research project focused on unsupervised anomaly detection in brain MRI.

---

## 📁 Dataset: BraTS 2021

We use the publicly available **BraTS 2021** dataset, which contains multimodal brain MRI scans with expert-annotated tumour segmentations. You can request or download the dataset from the following sources:

- Official Site: [http://braintumorsegmentation.org](http://braintumorsegmentation.org)
- Kaggle Mirror: [https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1)

Once downloaded, place the folder named `BraTS2021_Data` inside the project root directory (same level as the code files).

---

## 📊 Data Splits

The dataset is split into **training**, **validation**, and **test** sets. These splits are provided in the `data_splits/` directory as three `.csv` files under three seperate folders:

- `train.csv`
- `val.csv`
- `test.csv`

Each file contains the subject IDs corresponding to that particular data split.

---

## 🧠 Project Structure

The project consists of 15 Python files that together implement the full pipeline, including:

- Data loading and preprocessing
- Model architecture
- Training and validation
- Evaluation and visualization

The main file to run the full pipeline is: **main.ipynb**

## 📂 Output Structure

When running the pipeline, the following directories and files will be created automatically:

- `runs/` – stores experiment logs and TensorBoard output
- `splits/` – stores generated split actual brain MRI slices and metadata
- `saved_models/` – stores trained model checkpoints (**saved one level up** from the code directory)

---

## ⚙️ Environment Setup

We recommend using the `environment.yml` file to set up the conda environment.
