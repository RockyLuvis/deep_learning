from datetime import datetime
from typing import List

import torch

TIMESTAMP: datetime = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")

#Data Ingestion Constants
ARTIFACT_DIR: str = "artifacts"
BUCKET_NAME: str = "dlxrayproject"
S3_DATA_FOLDER: str = "data"

#Data Transformation constants

CLASS_LABEL_1: str = "NORMAL"
CLASS_LABEL_2: str = "PNEUMONIA"

# Get these values from PyTorch Documentation
BRIGHTNESS: int = 0.10
CONTRAST: int = 0.1
SATURATION: int = 0.10
HUE: int = 0.1
RESIZE: int = 224 #https://keras.io/api/applications/vgg/#vgg16-function
CENTERCROP: int = 224 # Default image resolution taken by all models and it is represented as 224,224,3 
#where 3 is number of channels - R G B
RANDOMROTATION: int = 10
NORMALIZE_LIST_1: List[int] = [0.485, 0.456, 0.406]
NORMALIZE_LIST_2: List[int] = [0.229, 0.224, 0.225]
TRAIN_TRANSFORMS_KEY: str = "xray_train_transforms"
TRAIN_TRANSFORMS_FILE: str = "train_transforms.pkl"
TEST_TRANSFORMS_FILE: str = "test_transforms.pkl"
BATCH_SIZE: int = 2
SHUFFLE: bool = False
PIN_MEMORY: bool = True


#model trainer constants

TRAINED_MODEL_DIR: str = "trained_model"
TRAINED_MODEL_NAME: str = "cvmodel.pt"

DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # If GPU (CUDA) is available then use 
#GPU or else choose CPU , hence these are needed.

STEP_SIZE: int = 6
GAMMA: int = 0.5
EPOCH: int = 1 # Choosen one for POC, but for real usecase use min 1000 and see accuracy and loss

"""
In the context of model training, an epoch refers to one complete cycle 
through the entire training dataset. During an epoch, the model processes every sample in the dataset exactly once.

Here's what happens during an epoch:
Data Pass: All data points in the training set are passed through the model.
Weights Update: After processing each batch of data, the model's weights are 
updated based on the gradients computed by the loss function.
Multiple Epochs: Generally, you don't train a model for just one epoch. 
Instead, you train for multiple epochs to allow the model to progressively improve its predictions by adjusting its weights.
"""

BENTOML_MODEL_NAME: str = "xray_model"
BENTOML_SERVICE_NAME: str = "xray_service"
BENTOML_ECR_URI: str = "xray_bento_image"

PREDICTION_LABEL: dict = {"0": CLASS_LABEL_1, 1: CLASS_LABEL_2}