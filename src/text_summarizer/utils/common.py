import os
from box.exceptions import BoxValueError
import yaml
from text_summarizer.logging import logger
from ensure import ensure_annotations
from box import ConfigBox
from pathlib import Path
from typing import Any


#first we will create a function to read yaml files
@ensure_annotations
def read_yaml(file_path: Union[str, Path]) -> ConfigBox:
    try:
        with open(file_path) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.into(f"yaml file: {file_path} loaded successfully")
            return ConfigBox(content)
    except BoxValueError:
        raise ValueError("yaml file is empty")
    except Exception as e:
        raise e
    

@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    try:
        for directory in path_to_directories:
            os.makedirs(directory, exist_ok = True)
            if verbose:
                logger.info(f"directory created at: {directory}")
    except Exception as e:
        raise e


@ensure_annotations
def get_size(path: Path) -> str:
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~{size_in_kb} KB"
