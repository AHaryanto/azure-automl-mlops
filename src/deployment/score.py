import json
import logging
import os

import joblib
import pandas as pd
from azureml.automl.core.shared import log_server, logging_utilities
from azureml.telemetry import INSTRUMENTATION_KEY

try:
    log_server.enable_telemetry(INSTRUMENTATION_KEY)
    log_server.set_verbosity('INFO')
    logger = logging.getLogger('azureml.automl.core.scoring_script')
except Exception:
    pass


def init():
    global model
    # This name is model.id of model that we want to deploy deserialize the
    # model file back into a sklearn model
    model_path = os.path.join(
        os.getenv('AZUREML_MODEL_DIR'), 'best_model_data')
    path = os.path.normpath(model_path)
    path_split = path.split(os.sep)
    log_server.update_custom_dimensions({
        'model_name': path_split[-3],
        'model_version': path_split[-2]})

    try:
        logger.info("Loading model from path.")
        model = joblib.load(model_path)
        logger.info("Loading successful.")
    except Exception as e:
        logging_utilities.log_traceback(e, logger)
        raise


def run(raw_data):
    try:
        # Parse a JSON string and convert it into a Python Dictionary
        data_dict = json.loads(raw_data)

        # Convert dictionary to pandas
        data_df = pd.DataFrame.from_dict(data_dict)

        # Predict
        y_pred = model.predict(data_df)
        y_pred_df = pd.DataFrame(y_pred, columns=['Weekly_Sales_Prediction'])
        return y_pred_df.to_dict()
    except Exception as e:
        error = str(e)
        return error
