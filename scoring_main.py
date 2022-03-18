import os
from datetime import datetime

import pytz
from azureml.core import Experiment
from azureml.data.data_reference import DataReference
from azureml.pipeline.core import Pipeline, PipelineData
from azureml.pipeline.core.graph import PipelineParameter
from azureml.pipeline.steps import PythonScriptStep
from loguru import logger

import config as f

exp = Experiment(
    workspace=f.ws,
    name=f.params["scoring_experiment_name"])

workspaceblobstore = f.ws.datastores[f.params['datastore_name']]

raw_data = DataReference(
    datastore=workspaceblobstore,
    data_reference_name='raw_data',
    mode='mount')

transform_out = PipelineData(
    name="transform_out",
    datastore=workspaceblobstore,
    is_directory=True)

model_name = PipelineParameter(
    name="model_name",
    default_value=f.params['registered_model_name'])

score_out = PipelineData(
    name='score_out',
    datastore=workspaceblobstore,
    is_directory=True)

transform_step = PythonScriptStep(
    name="transform",
    script_name="transform.py",
    arguments=[
        "--input_path", raw_data,
        "--output_path", transform_out],
    compute_target=f.compute_target,
    inputs=[raw_data],
    outputs=[transform_out],
    runconfig=f.pipestep_run_config,
    source_directory=os.path.join(
        os.getcwd(), "src", 'training_pipes', 'transform'),
    allow_reuse=False)

score_step = PythonScriptStep(
    name="score",
    script_name="score.py",
    arguments=[
        "--input_path", transform_out,
        "--output_path", score_out,
        "--model_name", model_name],
    compute_target=f.compute_target,
    inputs=[transform_out],
    outputs=[score_out],
    runconfig=f.pipestep_run_config,
    source_directory=os.path.join(
        os.getcwd(), "src", 'scoring_pipes', 'score'),
    allow_reuse=False)

pipeline = Pipeline(
    workspace=f.ws,
    steps=[transform_step, score_step])

if f.params['run_pipeline']:
    pipeline_run = exp.submit(
        pipeline,
        regenerate_outputs=True,
        continue_on_step_failure=False,
        tags=f.params)
    pipeline_run

if f.params["publish_pipeline"]:
    tz = pytz.timezone(f.params["timezone"])
    timenow = datetime.now(tz=tz).strftime('%Y-%m-%d_%H:%M_%Z')
    published_pipeline = pipeline.publish(
        name='Published_' + f.params["scoring_experiment_name"],
        description='Published_' + f.params["scoring_experiment_name"] +
        '_pipeline_' + timenow,
        continue_on_step_failure=False)
    logger.debug(published_pipeline)
