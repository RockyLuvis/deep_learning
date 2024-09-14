import sys
from xray.entity.config_entity import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig
)

from xray.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
    ModelTrainerArtifact
)

from xray.components.data_ingestion import DataIngestion
from xray.components.data_transformation import DataTransformation

from xray.exception import XRayException
from xray.logger import logging


class TrainPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.model_trainer_config = ModelTrainerConfig()


    def start_data_ingestion(self) -> DataIngestionArtifact:
        logging.info (" Started Data ingestion training pipeline")

        try:
            data_ingestion = DataIngestion(
                data_ingestion_config = self.data_ingestion_config
                )

            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info ("Completed Data ingestion")

            return data_ingestion_artifact


        except XRayException as e:
            raise XRayException (e, sys)
    
    def start_data_transformation(self, data_ingestion_artifact: DataIngestionArtifact) -> DataTransformationArtifact:
        
        logging.info("Entered the start_data_transformation method of TrainPipeline class")

        try:
            data_transformation = DataTransformation(
                data_ingestion_artifact=data_ingestion_artifact,
                data_transformation_config=self.data_transformation_config,
            )

            data_transformation_artifact = (
                data_transformation.initiate_data_transformation()
            )

            logging.info(
                "Exited the start_data_transformation method of TrainPipeline class"
            )

            return data_transformation_artifact

        except Exception as e:
            raise XRayException(e, sys)
    
    def run_pipeline(self) -> None:
        logging.info(" Entered the run_pipeline")

        try:
            data_ingestion_artifact: DataIngestionArtifact = self.start_data_ingestion()

            data_transformation_artifact: DataTransformationArtifact = (
                self.start_data_transformation (
                    data_ingestion_artifact = data_ingestion_artifact
                )
            )

        
        except XRayException as e:
            raise XRayException (e, sys)


