import argparse
import os

import pandas as pd
from azureml.core import Dataset, Datastore, Workspace
from azureml.core.run import Run, _OfflineRun
from loguru import logger
from sklearn.model_selection import train_test_split


def str_to_bool(value):
    """
    *Converts a string to a boolean

    *Args:
        *value (Str): The value to convert to a boolean

    *Returns:
        *vallue: True or False depending on value of input
    """
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


parser = argparse.ArgumentParser(
    description="getting inputs from the pipeline setup")
parser.add_argument("--input_path", type=str, default=os.path.join(
    'data', 'regression_kaggle_retail_data_analytics', 'raw'))
parser.add_argument("--output_path", type=str, default=os.path.join(
    'data', 'regression_kaggle_retail_data_analytics', 'processed'))
parser.add_argument("--register_dataset", type=str_to_bool, default=False)
parser.add_argument("--dataset_name", type=str,
                    default='retail_training_tabular_dataset')
args, unknown = parser.parse_known_args()

os.makedirs(args.output_path, exist_ok=True)

input_path = args.input_path
output_path = args.output_path
logger.debug(f'{input_path = }')
logger.debug(f'{output_path = }')
logger.debug(f'{args.register_dataset = }')
logger.debug(f'{args.dataset_name = }')

run = Run.get_context()
if type(run) == _OfflineRun:
    data_folder_path = input_path
    ws = Workspace.from_config()
else:
    data_folder_path = os.path.join(
        input_path, 'data', 'retail', 'train', 'raw')
    ws = run.experiment.workspace
logger.debug(f'{data_folder_path = }')

'''
Source: https://www.kaggle.com/manjeetsingh/retaildataset
'''
features_df = pd.read_csv(
    os.path.join(data_folder_path, 'Features data set.csv'))
sales_df = pd.read_csv(
    os.path.join(data_folder_path, 'sales data-set.csv'))
stores_df = pd.read_csv(
    os.path.join(data_folder_path, 'stores data-set.csv'))

# Merge data
merged_df = pd.merge(
    left=sales_df,
    right=features_df,
    how='left',
    on=['Store', 'Date', 'IsHoliday'])
merged_df = pd.merge(
    left=merged_df,
    right=stores_df,
    how='left',
    on=['Store'])
logger.debug(f'{merged_df.shape = }')  # (421570, 16)

# Convert booleans into integers
merged_df['IsHoliday'] = merged_df['IsHoliday'].astype(int)

print(merged_df.head())
print(merged_df['Weekly_Sales'].describe())

# Split data into train and test
train_df, test_df = train_test_split(merged_df, test_size=0.2, random_state=0)
logger.debug(f'{train_df.shape = }')  # (337256, 16)
logger.debug(f'{test_df.shape = }')  # (84314, 16)

# Sample data
sample_100_df = test_df.sample(n=100, random_state=0)
logger.debug(f'{sample_100_df.shape = }')  # (100, 16)

# Save train and test into csv files
train_df_path = os.path.join(output_path, 'train_data.csv')
test_df_path = os.path.join(output_path, 'test_data.csv')
sample_100_df_path = os.path.join(output_path, 'test_data_sample_100.csv')

train_df.to_csv(train_df_path, index=False)
test_df.to_csv(test_df_path, index=False)
sample_100_df.to_csv(sample_100_df_path, index=False)

logger.debug(f'train_df saved to {train_df_path}')
logger.debug(f'test_df saved to {test_df_path}')
logger.debug(f'sample_100_df saved to {sample_100_df_path}')

if args.register_dataset:
    workspaceblobstore = Datastore.get(ws, 'workspaceblobstore')
    logger.debug(f'Connected to datastore {workspaceblobstore.name}')
    training_tabular_dataset = Dataset.Tabular.register_pandas_dataframe(
        dataframe=train_df,
        target=workspaceblobstore,
        name=args.dataset_name,
        show_progress=True)
    logger.debug("Registered version {0} of dataset {1}".format(
        training_tabular_dataset.version, training_tabular_dataset.name))
