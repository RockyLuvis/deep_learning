# What are steps we can include in data Transformation

# Data Augmentation - To enchance size and quality of ML training dataset
# Resizing operation
# Play with brightness/contrast
# lower saturation/high saturation etc
# crop the image/resize image --- All these is called Data Augumentation
# Random rotation -
# Normalization 
# Train model so model can learn different representation of the data
###############################################################
# PyTorch Dataset Loader and Transformer is used for this purpose
# pytorch has all these out of box - i,e Train data loader, Test data loader
###################################################
# Keras applications - provides pretrained models that can also be used - https://keras.io/api/applications/
# In order to hugedata set use Kaggle - https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
# After data transformation we get Train_data.pkl and Test_data.pkl file , these are needed for model training
# 
import os
import sys
from typing import Tuple

import joblib
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


from xray.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)

from xray.entity.config_entity import DataTransformationConfig
from xray.exception import XRayException
from xray.logger import logging

class DataTransformation:

    def __init__(
        self,
        data_transformation_config: DataTransformationConfig,
        data_ingestion_artifact: DataIngestionArtifact,
    ):
        self.data_transformation_config = data_transformation_config

        self.data_ingestion_artifact = data_ingestion_artifact

    def transforming_training_data(self) -> transforms.Compose:
        
        try:
            logging.info(
                "transforming_training_data:Entered the transforming_training_data method of Data transformation class"
            )

            train_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ColorJitter(
                        **self.data_transformation_config.color_jitter_transforms
                    ),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(
                        self.data_transformation_config.RANDOMROTATION
                    ),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )

            logging.info(
                "transforming_training_data:Exited the transforming_training_data method of Data transformation class"
            )

            return train_transform

        except Exception as e:
            raise XRayException(e, sys)
    
    # Only resize and cropping operation are performed as all agumentation technique need not be performed.

    def transforming_testing_data(self) -> transforms.Compose:
        logging.info(
            "transforming_testing_data:Entered the transforming_testing_data method of Data transformation class"
        )

        try:
            test_transform: transforms.Compose = transforms.Compose(
                [
                    transforms.Resize(self.data_transformation_config.RESIZE),
                    transforms.CenterCrop(self.data_transformation_config.CENTERCROP),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        **self.data_transformation_config.normalize_transforms
                    ),
                ]
            )

            logging.info(
                "transforming_testing_data:Exited the transforming_testing_data method of Data transformation class"
            )

            return test_transform

        except Exception as e:
            raise XRayException(e, sys)

    def data_loader(
        self, train_transform: transforms.Compose, test_transform: transforms.Compose
    ) -> Tuple[DataLoader, DataLoader]:
        try:
            logging.info("data_loader:Entered the data_loader method of Data transformation class")

            train_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.train_file_path),
                transform=train_transform,
            )

            test_data: Dataset = ImageFolder(
                os.path.join(self.data_ingestion_artifact.test_file_path),
                transform=test_transform,
            )

            logging.info("data_loader:Created train data and test data paths")

            train_loader: DataLoader = DataLoader(
                train_data, **self.data_transformation_config.data_loader_params
            )

            test_loader: DataLoader = DataLoader(
                test_data, **self.data_transformation_config.data_loader_params
            )

            logging.info("data_loader:Exited the data_loader method of Data transformation class")

            return train_loader, test_loader

        except Exception as e:
            raise XRayException(e, sys)

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        
        try:
            logging.info(
                "initiate_data_transformation:Entered the initiate_data_transformation method of Data transformation class"
            )

            train_transform: transforms.Compose = self.transforming_training_data()

            test_transform: transforms.Compose = self.transforming_testing_data()

            os.makedirs(self.data_transformation_config.artifact_dir, exist_ok=True)

            joblib.dump(
                train_transform, self.data_transformation_config.train_transforms_file
            )

            joblib.dump(
                test_transform, self.data_transformation_config.test_transforms_file
            )

            train_loader, test_loader = self.data_loader(
                train_transform=train_transform, test_transform=test_transform
            )

            data_transformation_artifact: DataTransformationArtifact = DataTransformationArtifact(
                transformed_train_object=train_loader,
                transformed_test_object=test_loader,
                train_transform_file_path=self.data_transformation_config.train_transforms_file,
                test_transform_file_path=self.data_transformation_config.test_transforms_file,
            )
            logging.info(
                f"initiate_data_transformation: train_transform_file_path = train_transform_file_path,test_transform_file_path=test_transform_file_path"
            )
            logging.info(
                "initiate_data_transformation:Completed data_transformation method of Data transformation class"
            )

            return data_transformation_artifact
        
        except Exception as e:
            raise XRayException(e, sys)