import json
import os
import sys

from azureml.core import VERSION, Environment, Workspace
from azureml.core.authentication import (InteractiveLoginAuthentication,
                                         ServicePrincipalAuthentication)
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.pipeline.core import PipelineEndpoint
from loguru import logger

sys.path.append(os.getcwd())
logger.debug(f"Azure ML SDK Version: {VERSION}")

# loading project params
params = json.load(open('config.json'))

# Connect to ML Service
if params['remote_run'] is True:
    try:
        logger.debug("Authenticate using service principal")
        auth = ServicePrincipalAuthentication(
            tenant_id=os.getenv('tenantId'),
            # 1.0.33 change to service_principal_id
            service_principal_id=os.getenv('servicePrincipalId'),
            service_principal_password=os.getenv('servicePrincipalKey'),
            # 1.0.33 change to service_principal_password
        )
    except KeyError:
        raise Exception('Error getting Service Principal Authentication')
elif params['remote_run'] is False:
    logger.debug("Authenticate interactively")
    auth = InteractiveLoginAuthentication(
        tenant_id=params["tenant_id"])
else:
    raise Exception('remote_run unknown value. The value was: ' +
                    params['remote_run'])

ws = Workspace.from_config(auth=auth)
logger.debug(f"Found workspace {ws.name} at location {ws.location}")

try:
    # Check for existing compute target
    compute_target = ComputeTarget(
        workspace=ws,
        name=params["compute_name"])
    logger.debug(f'Found existing "{params["compute_name"]}" cluster.')
except ComputeTargetException:
    # If it doesn't already exist, create it
    try:
        logger.debug('Creating a new compute cluster...')
        compute_config = AmlCompute.provisioning_configuration(
            vm_size=params['vm_size'],
            min_nodes=0,
            max_nodes=1,
            idle_seconds_before_scaledown=120,
            location=ws.location)
        pipeline_cluster = ComputeTarget.create(
            ws,
            name=params["compute_name"],
            provisioning_configuration=compute_config)
        pipeline_cluster.wait_for_completion(show_output=True)
        logger.debug('Compute cluster created successfully.')
    except Exception as ex:
        logger.debug(ex)

# Printing file systems
logger.debug("These are the connected file systems available now:")
datastores = ws.datastores
for name, datastore in datastores.items():
    logger.debug(f'-{name}, {datastore.datastore_type}')

# Setting up runtime environments
# conda env export > env.yml
environment = Environment.from_conda_specification(
    name=params['environment_name'],
    file_path='env.yaml')

environment.register(workspace=ws)

environment = Environment.get(
    workspace=ws,
    name=params['environment_name'])

pipestep_run_config = RunConfiguration()
pipestep_run_config.target = compute_target
pipestep_run_config.docker.use_docker = True
pipestep_run_config.environment = environment


def publish_pipeline_endpoint(
        workspace,
        published_pipeline,
        pipeline_endpoint_name,
        pipeline_endpoint_description):
    try:
        pipeline_endpoint = PipelineEndpoint.get(
            workspace=workspace,
            name=pipeline_endpoint_name)
        logger.debug("Found existing PipelineEndpoint.")
        pipeline_endpoint.add_default(published_pipeline)
    except Exception as e:
        logger.debug(e)
        logger.debug("PipelineEndpoint does not exist. " +
                     "Creating a new PipelineEndpoint...")
        pipeline_endpoint = PipelineEndpoint.publish(
            workspace=workspace,
            name=pipeline_endpoint_name,
            pipeline=published_pipeline,
            description=pipeline_endpoint_description)
    return pipeline_endpoint
