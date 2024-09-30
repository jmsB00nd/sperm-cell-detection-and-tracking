# Investigating Dedicated Methods of Small Object Detection Using YOLO for Sperm Cell Detection and Tracking

This repository contains the code, sample data, and weights used during my internship project. The aim of the project was to investigate the performance of dedicated methods for detecting and tracking small objects, specifically sperm cells, using the YOLOv5 architecture. The model was trained on the MSD dataset.

## Project Structure

- **`detect.py`**: This script is used for the detection of sperm cells using the YOLOv5 model.
- **`tracking.py`**: This script is responsible for tracking detected sperm cells using the DeepSORT algorithm.
- **Notebooks**: These Jupyter notebooks were used to train the YOLOv5 model on the MSD dataset. They contain preprocessing, training, and evaluation procedures.
- **Sample Data**: The repository contains some sample data from the MSD dataset, used for training and testing purposes.
- **Weights**: Pretrained weights for the YOLOv5 model are included, which can be used for inference.

## Setup and Installation

1. Clone the repository:
    ```bash
    git clone <repository-link>
    ```
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Download the MSD dataset 
4. Place the dataset in the appropriate directory.

## Usage

### Detection

To run the detection model, use the `detect.py` script:
```bash
python detect.py // and don't forget to specify the path of image
