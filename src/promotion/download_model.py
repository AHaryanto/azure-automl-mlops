import os
import sys

from azureml.core.model import Model
from loguru import logger

sys.path.append(os.getcwd())
import config as f  # noqa: E402

model_name = f.params["registered_model_name"]
output_dir = os.path.join('models', model_name)

# Retrive an Azure Machine Learning model
model = Model(
    workspace=f.ws,
    name=model_name)
logger.debug(f'{model.name} successfully retrieved')

# Download the model
downloaded_model = model.download(
    target_dir=output_dir,
    exist_ok=True)
logger.debug(f'{model.name} saved to {downloaded_model}')
logger.debug(f'##vso[task.setvariable variable=MODEL_PATH]{downloaded_model}')
