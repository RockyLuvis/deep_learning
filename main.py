import sys
from xray.exception import XRayException
from xray.pipeline.training_pipeline import TrainPipeline

def start_training():

    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()
        
    except XRayException as e:
        raise XRayException (e, sys)

if __name__ == "__main__":
    start_training()
