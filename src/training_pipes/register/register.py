import argparse
import os

from azureml.core import Workspace
from azureml.core.model import Model
from azureml.core.run import Run, _OfflineRun
from loguru import logger

parser = argparse.ArgumentParser(
    description="getting inputs from the pipeline setup")
parser.add_argument("--input_path", type=str, default=os.path.join('models'))
parser.add_argument("--model_name", type=str)
args, unknown = parser.parse_known_args()

logger.debug(f'{args.input_path = }')
logger.debug(f'{args.model_name = }')

run = Run.get_context()
ws = Workspace.from_config() if type(
    run) == _OfflineRun else run.experiment.workspace

model = Model.register(
    workspace=ws,
    model_path=args.input_path,
    model_name=args.model_name)

logger.debug("Registered version {0} of model {1}".format(
    model.version, model.name))
