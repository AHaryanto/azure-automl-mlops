import json
import os
import sys

from azureml.core import VERSION, Workspace
from azureml.core.authentication import (InteractiveLoginAuthentication,
                                         ServicePrincipalAuthentication)
from azureml.core.compute import ComputeTarget
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

compute_target = ComputeTarget(
    workspace=ws,
    name=params["compute_name"])

# Printing file systems
logger.debug("These are the connected file systems available now:")
datastores = ws.datastores
for name, datastore in datastores.items():
    logger.debug(f'-{name}, {datastore.datastore_type}')

# Setting up runtime environments
# conda env export > env.yml
custom_conda = CondaDependencies(conda_dependencies_file_path='env.yaml')
pipestep_run_config = RunConfiguration(conda_dependencies=custom_conda)
