'''
Prepare Model Trainer configuration:
 Prepare constant -> Model Trainer Config <- Take data transformation training data pkl file use to train model (Transformed train and test obj)
 Initiate Model Trainer (from Model Trainer config) which will generate model.pt 
 Prepare Model trainer Artifact with Triner data.
 After generating Model Trainer artifact we go ahead with evaluation of the model use
 
 Trained Model Path is the path returned to Model Evaluation
 
 Follow the following workflow
 - constants -- From the flow diagram , Trained Model dir, Trained Model path, params epochs, step size, gamma, Epoch, Model, Optimizer, Device
 - config_entity
 - artifact_entity
 - components
 - pipeline
 - main

 https://pytorch.org/tutorials/beginner/introyt/trainingyt.html
'''

import os
import sys
from dataclasses import dataclass

import bentoml
import joblib
import torch

import torch.nn.functional as F
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import StepLR, _LRScheduler
from tqdm import tqdm

from xray.constant.training_pipeline import *
from xray.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from xray.entity.config_entity import ModelTrainerConfig
from xray.exception import XRayException
from xray.logger import logging
from xray.ml.model.arch import Net

#Image classification models
# Search Keras applications
# Xception , VGG16 etc these are pretrained models but we are creating our own
# convenlution 2D, Maxpool2d
# Read about LeNet(Module)

class ModelTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact, # Get the path to the test and train data path
        model_trainer_config: ModelTrainerConfig, # Get model trainer configuration
    ):
        self.model_trainer_config: ModelTrainerConfig = model_trainer_config

        self.data_transformation_artifact: DataTransformationArtifact = (
            data_transformation_artifact
        )

        self.model: Module = Net()

    def train(self, optimizer: Optimizer) -> None:
        """
        Description: To train the model

        input: model,device,train_loader,optimizer,epoch

        output: loss, batch id and accuracy
        """
        logging.info("Entered the train method of Model trainer class")

        try:
            self.model.train() #Train the model

            pbar = tqdm(self.data_transformation_artifact.transformed_train_object) #Show the progress on the terminal

            correct: int = 0

            processed = 0

            '''
            Gradient Calculation and Weight Update (Training Stage):
            Zero Gradients: optimizer.zero_grad() ensures that previous gradients are cleared so they don't accumulate.
            Forward Pass: y_pred = self.model(data) performs the prediction.
            Loss Calculation: loss = F.nll_loss(y_pred, target) calculates the difference between predictions (y_pred) and actual targets (target).
            Backpropagation: loss.backward() computes gradients with respect to the model's weights based on the loss.
            Weight Update: optimizer.step() updates the model's weights using the computed gradients, 
            moving the model's predictions closer to the actual target in future iterations.
            Training Steps:
            Input Data: data, target = data.to(DEVICE), target.to(DEVICE) loads a batch of data and target labels.
            Clear Old Gradients: optimizer.zero_grad()

            '''
            for batch_idx, (data, target) in enumerate(pbar): # in the loop , take data, train and calculate loss and perform back propagation
                data, target = data.to(DEVICE), target.to(DEVICE)

                # Initialization of gradient
                optimizer.zero_grad()  #Clear Old gradients

                # In PyTorch, gradient is accumulated over backprop and even though thats used in RNN generally not used in CNN
                # or specific requirements
                ## prediction on data

                y_pred = self.model(data) # This step is called forward pass, we predict in this step.

                # Calculating loss given the prediction,, Loss is Actual vs predicted loss.
                loss = F.nll_loss(y_pred, target)

                # Backpropogate to calculate gradient, 
                loss.backward() # This step calculates gradients with respects to the current weight.

                optimizer.step() # Update weights using the newly calcuated gradients so predicted is closer to the actual.

                # get the index of the log-probability corresponding to the max value
                pred = y_pred.argmax(dim=1, keepdim=True)

                correct += pred.eq(target.view_as(pred)).sum().item() #calculate loss and accuracy

                processed += len(data)

                pbar.set_description(
                    desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}" 
                )

            logging.info("Exited the train method of Model trainer class")

        except Exception as e:
            raise XRayException(e, sys)
        


    def test(self) -> None:
        try:
            """
            Description: To test the model

            input: model, DEVICE, test_loader

            output: average loss and accuracy

            """
            logging.info("Entered the test method of Model trainer class")

            self.model.eval() # Perform prediction with test data

            test_loss: float = 0.0

            correct: int = 0

            with torch.no_grad():
                for (
                    data,
                    target,
                ) in self.data_transformation_artifact.transformed_test_object:
                    data, target = data.to(DEVICE), target.to(DEVICE)

                    output = self.model(data)

                    test_loss += F.nll_loss(output, target, reduction="sum").item()

                    pred = output.argmax(dim=1, keepdim=True)

                    correct += pred.eq(target.view_as(pred)).sum().item()

                test_loss /= len(
                    self.data_transformation_artifact.transformed_test_object.dataset
                )

                print( #Calculate loss and accuracy
                    "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                        test_loss,
                        correct,
                        len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                        100.0
                        * correct
                        / len(
                            self.data_transformation_artifact.transformed_test_object.dataset
                        ),
                    )
                )

            logging.info(
                "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)".format(
                    test_loss,
                    correct,
                    len(
                        self.data_transformation_artifact.transformed_test_object.dataset
                    ),
                    100.0
                    * correct
                    / len(
                        self.data_transformation_artifact.transformed_test_object.dataset
                    ),
                )
            )

            logging.info("Exited the test method of Model trainer class")

        except Exception as e:
            raise XRayException(e, sys)
        


    def initiate_model_trainer(self) -> ModelTrainerArtifact:
        try:
            logging.info(
                "Entered the initiate_model_trainer method of Model trainer class"
            )

            model: Module = self.model.to(self.model_trainer_config.device)

            optimizer: Optimizer = torch.optim.SGD(
                model.parameters(), **self.model_trainer_config.optimizer_params
            )

            scheduler: _LRScheduler = StepLR(
                optimizer=optimizer, **self.model_trainer_config.scheduler_params
            )

            for epoch in range(1, self.model_trainer_config.epochs + 1):
                print("Epoch : ", epoch)

                self.train(optimizer=optimizer)

                optimizer.step()

                scheduler.step()

                self.test()

            os.makedirs(self.model_trainer_config.artifact_dir, exist_ok=True)

            torch.save(model, self.model_trainer_config.trained_model_path)
            #os.system(f"cp {self.model_trainer_config.trained_model_path} model/")

            train_transforms_obj = joblib.load(
                self.data_transformation_artifact.train_transform_file_path
            )

            bentoml.pytorch.save_model( # Model is saved as bentoml 
                name=self.model_trainer_config.trained_bentoml_model_name,
                model=model,
                custom_objects={
                    self.model_trainer_config.train_transforms_key: train_transforms_obj
                },
            )

            model_trainer_artifact: ModelTrainerArtifact = ModelTrainerArtifact(
                trained_model_path=self.model_trainer_config.trained_model_path
            )

            logging.info(
                "Exited the initiate_model_trainer method of Model trainer class"
            )

            return model_trainer_artifact

        except Exception as e:
            raise XRayException(e, sys)
