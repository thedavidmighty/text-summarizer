from text_summarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from text_summarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from text_summarizer.pipeline.stage_03_data_transformation import DataTranformationTrainingPipeline
from text_summarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from text_summarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
from text_summarizer.logging import logger

stage_name = "Data Ingestion Stage"

try:
    logger.info(f"Stage {stage_name} has been initiated")
    data_ingestion_training_pipeline = DataIngestionTrainingPipeline()
    data_ingestion_training_pipeline.main()
    logger.info(f"Stage {stage_name} has run successfully")
except Exception as e:
    logger.exception(e)
    raise e

stage_name = "Data Validation Stage"

try:
    logger.info(f"Stage {stage_name} has been initiated")
    data_validation_training_pipeline = DataValidationTrainingPipeline()
    data_validation_training_pipeline.main()
    logger.info(f"Stage {stage_name} has run successfully")
except Exception as e:
    logger.exception(e)
    raise e

stage_name = "Data Transformation Stage"

try:
    logger.info(f"Stage {stage_name} has been initiated")
    data_transformation_training_pipeline = DataTranformationTrainingPipeline()
    data_transformation_training_pipeline.main()
    logger.info(f"Stage {stage_name} has run successfully")
except Exception as e:
    logger.exception(e)
    raise e

stage_name = "Model Training Stage"

try:
    logger.info(f"Stage {stage_name} has been initiated")
    model_trainer_training_pipeline = ModelTrainerTrainingPipeline()
    model_trainer_training_pipeline.main()
    logger.info(f"Stage {stage_name} has run successfully")
except Exception as e:
    logger.exception(e)
    raise e

stage_name = "Model Evaluation Stage"

try:
    logger.info(f"Stage {stage_name} has been initiated")
    model_evaluation_training_pipeline = ModelEvaluationTrainingPipeline()
    model_evaluation_training_pipeline.main()
    logger.info(f"Stage {stage_name} has run successfully")
except Exception as e:
    logger.exception(e)
    raise e