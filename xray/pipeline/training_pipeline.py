import sys
from xray.entity.config_entity import DataIngestionConfig
from xray.entity.artifact_entity import DataIngestionArtifact
from xray.components.data_ingestion import DataIngestion
from xray.exception import XRayException
from xray.logger import logging


class TrainPipeline:

    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

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


