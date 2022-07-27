import json
import os
import sys

from azureml.core import VERSION, Workspace
from azureml.core.authentication import (InteractiveLoginAuthentication,
                                         ServicePrincipalAuthentication)
from azureml.core.compute import AmlCompute, ComputeTarget
from azureml.core.compute_target import ComputeTargetException
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import CondaDependencies, RunConfiguration
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
custom_conda = CondaDependencies(conda_dependencies_file_path='env.yaml')
pipestep_run_config = RunConfiguration(conda_dependencies=custom_conda)
