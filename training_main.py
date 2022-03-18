import os

from azureml.core import Dataset, Experiment
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData, TrainingOutput
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import AutoMLStep, PythonScriptStep
from azureml.train.automl import AutoMLConfig
from loguru import logger

import config as f

exp = Experiment(
    workspace=f.ws,
    name=f.params["training_experiment_name"])

workspaceblobstore = f.ws.datastores[f.params['datastore_name']]

raw_data = DataReference(
    datastore=workspaceblobstore,
    data_reference_name='raw_data',
    mode='mount')

dataset_name = PipelineParameter(
    name="dataset_name",
    default_value=f.params['registered_dataset_name'])

model_name = PipelineParameter(
    name="model_name",
    default_value=f.params['registered_model_name'])

datasets = Dataset.get_all(f.ws)
if f.params['registered_dataset_name'] in datasets:
    train_data = Dataset.get_by_name(
        workspace=f.ws,
        name=f.params['registered_dataset_name'])
    logger.debug("Retrieved version {0} of dataset {1}".format(
        train_data.version, train_data.name))
else:
    raise FileNotFoundError('Please register a training dataset first by ' +
                            'running training_pipes/transform/transform.py ' +
                            'locally.')

transform_out = PipelineData(
    name="transform_out",
    datastore=workspaceblobstore,
    is_directory=True)

metrics_data = PipelineData(
    name='metrics_data',
    datastore=workspaceblobstore,
    pipeline_output_name='metrics_output',
    training_output=TrainingOutput(type='Metrics'))

model_data = PipelineData(
    name='best_model_data',
    datastore=workspaceblobstore,
    pipeline_output_name='model_output',
    training_output=TrainingOutput(type='Model'))

score_out = PipelineData(
    name='score_out',
    datastore=workspaceblobstore,
    is_directory=True)

transform_step = PythonScriptStep(
    name="transform",
    script_name="transform.py",
    arguments=[
        "--input_path", raw_data,
        "--output_path", transform_out,
        "--register_dataset", f.params['register_dataset'],
        "--dataset_name", dataset_name],
    compute_target=f.compute_target,
    inputs=[raw_data],
    outputs=[transform_out],
    runconfig=f.pipestep_run_config,
    source_directory=os.path.join(
        os.getcwd(), "src", 'training_pipes', 'transform'),
    allow_reuse=True)

automl_config = AutoMLConfig(
    task="regression",
    path='automl',
    iterations=2,
    primary_metric='normalized_root_mean_squared_error',  # price prediction
    compute_target=f.compute_target,
    featurization="auto",
    max_cores_per_iteration=-1,
    max_concurrent_iterations=15,
    iteration_timeout_minutes=5,
    experiment_timeout_hours=0.25,  # minimum 15 minutes
    model_explainability=True,
    debug_log='automl_errors.log',
    training_data=train_data,
    label_column_name="Weekly_Sales")

train_step = AutoMLStep(
    name='automl_regression',
    automl_config=automl_config,
    inputs=[transform_out],
    outputs=[metrics_data, model_data],
    enable_default_model_output=False,
    enable_default_metrics_output=False,
    allow_reuse=True)

register_step = PythonScriptStep(
    name="register_model",
    script_name="register_model.py",
    arguments=[
        "--input_path", model_data,
        "--model_name", model_name],
    compute_target=f.compute_target,
    inputs=[model_data],
    runconfig=f.pipestep_run_config,
    source_directory=os.path.join(
        os.getcwd(), "src", 'training_pipes', 'register'),
    allow_reuse=True)

score_step = PythonScriptStep(
    name="score",
    script_name="score.py",
    arguments=[
        "--data_path", transform_out,
        "--model_path", model_data,
        "--output_path", score_out],
    compute_target=f.compute_target,
    inputs=[transform_out, model_data],
    outputs=[score_out],
    runconfig=f.pipestep_run_config,
    source_directory=os.path.join(
        os.getcwd(), "src", 'training_pipes', 'score'),
    allow_reuse=True)

pipeline_steps = [transform_step, train_step, register_step, score_step]
if not f.params['register_model']:
    pipeline_steps.remove(register_step)

pipeline = Pipeline(
    workspace=f.ws,
    steps=pipeline_steps)

if f.params['run_pipeline']:
    pipeline_run = exp.submit(
        pipeline,
        regenerate_outputs=False,
        continue_on_step_failure=False,
        tags=f.params)
    pipeline_run
