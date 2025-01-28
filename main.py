from text_summarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
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