# Project of Computer Vision: Car Plate recognition and reconstruction with Deep Learning
#### Sapienza University of Rome, Artificial Intelligence and Robotics Master Degree 
#### D'Addario Giuseppe MAT:2177530, Benucci Lorenzo MAT:2219690, Tomassacci Samuele MAT:
***
### Overview
The goal of this project was to develop a solid pipeline for the detection and recognition of carplates in the images coming from the CCPD2019 (Chinese Carplate Parking Dataset)[1]. Two pairs of models were developed and compared, a simple baseline and the approach of YOLOv5-PDLPR presented in [2]. The evaluation metrics that were used are the accuracy on the detection, the accuracy on the full pipeline, the time of inferece (FPS) and other metrics linked to the imbalance of the dataset, namely, the accuracy computed only on the detected samples and the accuracy computed only on the numeric part of the carplate, ignoring the chinese province.
***
### How to run the code
To reproduce the results or run the project locally, simply execute the Jupyter notebook main.ipynb.
It contains all the necessary steps, including data loading, model definition, training, and evaluation.

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
- **`Requirements/`**: This section installs the necessary dependencies and clones the project repository
- **`Imports/`**: All external libraries and custom modules used throughout the notebook are imported here, including PyTorch, torchvision, and utility functions.
- **`Globals/`**: Defines global parameters and configuration constants such as seed value, device selection (CPU/GPU), learning rate, and number of epochs.
- **`Utils/`**: Includes utility functions that support data processing, visualization, or any general-purpose functionality required during training and evaluation.
- **`Data/`**: Handles dataset loading and preprocessing steps, including normalization and data augmentation.
- **`Network/`**: Defines the models used, in particular way both the baseline and the YOLOv5-PDLPR model.
- **`Train/`**: Contains the training loop logic, including forward pass, loss computation, backpropagation, and model optimization.
- **`Evaluation/`**: Performs evaluation on the test sets, including metric computation like accuracy or FPS

***
## Useful Links

- [Unzipped Dataset](https://drive.google.com/drive/folders/1r7S8z7yIUUpCLBC6y0bq9sBneIMsUxVZ?usp=sharing) - CCPD2019

***
## References

- [1](https://www.researchgate.net/publication/380201742_A_Real-Time_License_Plate_Detection_and_Recognition_Model_in_Unconstrained_Scenarios#fullTextFileContent) - Tao, L., Hong, S., Lin, Y., Chen, Y., He, P. and Tie, Z. (2024). A Real-Time License Plate Detection and
Recognition Model in Unconstrained Scenarios. Sensors, 24(9), 2791

- [2](https://openaccess.thecvf.com/content_ECCV_2018/papers/Zhenbo_Xu_Towards_End-to-End_License_ECCV_2018_paper.pdf) - Xu, Z.; Yang, W.; Meng, A.; Lu, N.; Huang, H.; Ying, C.; Huang, L. Towards end-to-end license plate
detection and recognition: A large dataset and baseline. In Proceedings of the European Conference on
Computer Vision (ECCV), Munich, Germany, 8–14 September 2018.
