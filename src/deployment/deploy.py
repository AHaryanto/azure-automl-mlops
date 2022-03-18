import os
import sys

from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import AksWebservice
from loguru import logger

sys.path.append(os.getcwd())
import config as f  # noqa: E402

# Provision AKS cluster
# Compute name must be between 2 to 16 characters long
aks_cluster_name = f.params['aks_cluster_name']
# Verify that cluster does not exist already
try:
    aks_target = ComputeTarget(workspace=f.ws, name=aks_cluster_name)
    logger.debug('Found existing cluster')
except ComputeTargetException:
    logger.debug("Creating new cluster")
    provisioning_config = AksCompute.provisioning_configuration(
        agent_count=1,  # 1 node for a development cluster
        vm_size='Standard_D3_v2',  # 4 cores, 14 GB RAM, 28 GB disk
        location=f.params['location'],
        cluster_purpose='DevTest')  # DevTest or FastProd

    # Can contain only lowercase letters, numbers and hyphens.
    # The value must be between 3 and 16 characters long.
    provisioning_config.enable_ssl(
        leaf_domain_label="mlops-dev",
        overwrite_existing_domain=True)

    aks_target = ComputeTarget.create(
        workspace=f.ws,
        name=aks_cluster_name,
        provisioning_configuration=provisioning_config)

# Wait for the cluster to complete, show the output log
if aks_target.get_status() != "Succeeded":
    aks_target.wait_for_completion(show_output=True)

# Set AKS web service configuration
aks_config = AksWebservice.deploy_configuration(
    scoring_timeout_ms=300000)  # 5 minutes

# Create environment object from a YAML file
conda_yaml_file_path = os.path.join('env.yaml')
myenv = Environment.from_conda_specification(
    name=f.params['aks_env_name'],
    file_path=conda_yaml_file_path)

# Create the inference config that will be used when deploying the model
entry_script_path = os.path.join('src', 'deployment', 'score.py')
inference_config = InferenceConfig(
    entry_script=entry_script_path,
    environment=myenv)

# Get the latest model
model = Model(
    workspace=f.ws,
    name=f.params['registered_model_name'])
logger.debug(f'Found model {model.name} in {f.ws.name}')

# Deploy web service to AKS
real_time_endpoint_name = f.params['aks_endpoint_name']
aks_service = Model.deploy(
    workspace=f.ws,
    name=real_time_endpoint_name,
    models=[model],
    inference_config=inference_config,
    deployment_config=aks_config,
    deployment_target=aks_target,
    overwrite=True)
aks_service.wait_for_deployment(show_output=True)
logger.debug(aks_service.state)
