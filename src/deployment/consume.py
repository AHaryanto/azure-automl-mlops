import json
import os
import sys

import pandas as pd
from azureml.core.webservice import AksWebservice
from loguru import logger

sys.path.append(os.getcwd())
import config as f  # noqa: E402

# Load test data
data_path = os.path.join('data', 'regression_kaggle_retail_data_analytics',
                         'processed', 'test_data_sample_100.csv')
test_df = pd.read_csv(data_path)
x_test_df = test_df.drop(['Weekly_Sales'], axis=1)
print(x_test_df)

data_dict = x_test_df.to_dict()
data_json = json.dumps(data_dict)

# Retrieve a Webservice
aks_service = AksWebservice(
    workspace=f.ws,
    name=f.params['aks_endpoint_name'])
logger.debug('Found Webservice {} in {}'.format(
    aks_service.name, aks_service.workspace.name))

# Call the Webservice with the provided input
y_pred = aks_service.run(input_data=data_json)
y_pred_df = pd.DataFrame.from_dict(y_pred)
print(y_pred_df)
