from text_summarizer.config.configuration import ConfigurationManager
from text_summarizer.components.model_evaluation import ModelEvaluation
from text_summarizer.logging import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        """
        Orchestrates the model evaluation pipeline.

        This method does the following:

        - fetches the model evaluation configuration
        - instantiates the ModelEvaluation component
        - calls the evaluate method of the ModelEvaluation component

        :return: None
        """
        config = ConfigurationManager()
        model_evaluation_config = config.get_model_evaluation_config()
        model_evaluation_config = ModelEvaluation(config=model_evaluation_config)
        model_evaluation_config.evaluate()

    