import os
import sys

from azureml.core.model import Model

sys.path.append(os.getcwd())
import config as f  # noqa: E402

model_name = f.params["registered_model_name"]

if f.params['remote_run'] is True:
    model_path = os.environ['MODEL_PATH']
elif f.params['remote_run'] is False:
    model_path = os.path.join('models', model_name, 'best_model_data')
else:
    raise Exception('remote_run unknown value. The value was: ' +
                    f.params['remote_run'])
print(f'Registering {model_name} from {model_path}')

model = Model.register(
    model_path=model_path,
    model_name=model_name,
    tags={
        'industry': 'retail',
        'type': 'regression'
    },
    description="Retail AutoML regression model.",
    workspace=f.ws)
print(f'{model.name} successfully registered to {f.ws.name}')
