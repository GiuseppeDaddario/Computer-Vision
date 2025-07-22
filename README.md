# Computer Vision
Repository for the Computer Vision project



## PDLPR [RECOGNITION] pipeline
- preprocessing
- igfe
- encoder
- decoder


## Useful Links

- [Dataset](https://drive.google.com/drive/folders/1r7S8z7yIUUpCLBC6y0bq9sBneIMsUxVZ?usp=sharing) - CCPD2019


## Project of Computer Vision: Car Plate recognition and reconstruction with Deep Learning
#### Sapienza University of Rome, Artificial Intelligence and Robotics Master Degree 
#### D'Addario Giuseppe MAT:2177530, Benucci Lorenzo MAT:2219690, Tomassacci Samuele MAT:
***
### Overview
The goal of this project was to develop a solid pipeline for the detection and recognition of carplates in the images coming from the CCPD2019 (Chinese Carplate Parking Dataset)[1]. Two pairs of models were developed and compared, a simple baseline and the approach of YOLOv5-PDLPR presented in [2]. The evaluation metrics that were used are the accuracy on the detection, the accuracy on the full pipeline, the time of inferece (FPS) and other metrics linked to the imbalance of the dataset, namely, the accuracy computed only on the detected samples and the accuracy computed only on the numeric part of the carplate, ignoring the chinese province.
***
### How to run the code
The notebook containing everything needed for the replication of the work is "main.ipynb".

***
### Structure of the Repository

The repository is organized into the following main components:

- **`main.ipynb`**: The notebook of the project.

- **`cineca/`**: Contains files and scripts used to run experiments and generations on the CINECA cluster.

- **`datasets/`**: Contains a toy dataset created with 5 samples per subset, for demonstration purposes.
  - `ccpd_base/`: The training subset
  - `ccpd_*`: Testing subsets with different types of images.

- **`src/`**: Source code of the project used in phase of developing. Is not needed to run the notebook.
    - `YOLO/`: training and results of the YOLOv5 model.
    - `YOLO/yolov5/`: Official repository of YOLOv5 cloned as a sub-repository.
    - `PDLPR/`: training and results of the PDLPR model.
    - `baseline/`: training and results of the baseline model.

- **`docs/`**: Some relevant papers including [1] and [2].

- **`README.md`**: This documentation file.

***
### Structure of the Notebook
- **`Requirements/`**:
- **`Imports/`**:
- **`Globals/`**:
- **`Utils/`**:
- **`Data/`**:
- **`Network/`**:
- **`Train/`**:
- **`Evaluation/`**: