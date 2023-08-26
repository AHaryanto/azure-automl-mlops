import os
import sys

from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AciWebservice
from loguru import logger

sys.path.append(os.getcwd())
import config as f  # noqa: E402

# Set ACI web service configuration
aci_config = AciWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1)

# Get the environment
environment = Environment.get(
    workspace=f.ws,
    name=f.params['environment_name'])

# Create the inference config that will be used when deploying the model
entry_script_path = os.path.join('src', 'deployment', 'score.py')
inference_config = InferenceConfig(
    entry_script=entry_script_path,
    environment=environment)

# Get the latest model
model = Model(
    workspace=f.ws,
    name=f.params['registered_model_name'])
logger.debug(f'Found model {model.name} in {f.ws.name}')

# Deploy web service to ACI
real_time_endpoint_name = f.params['aci_endpoint_name']
aci_service = Model.deploy(
    workspace=f.ws,
    name=real_time_endpoint_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=aci_config,
    overwrite=True)
aci_service.wait_for_deployment(show_output=True)
logger.debug(aci_service.state)
