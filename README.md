# Introduction 
Azure Machine Learning's automated ML capability helps you discover high-performing models without you reimplementing every possible approach. Combined with Azure Machine Learning pipelines, you can create deployable workflows that can quickly discover the algorithm that works best for your data. This article will show you how to efficiently join a data preparation step to an automated ML step. Automated ML can quickly discover the algorithm that works best for your data, while putting you on the road to MLOps and model lifecycle operationalization with pipelines.

# Machine Learning Pipelines
* Training pipeline:

    ![training pipeline diagram](media/training_pipeline.png)

* Scoring pipeline:
    
    ![scoring pipeline diagram](media/scoring_pipeline.png)

# Prerequisites

* An Azure subscription. If you don't have an Azure subscription, create a free account before you begin. Try the [free or paid version of Azure Machine Learning](https://azure.microsoft.com/free/) today.

* An Azure Machine Learning workspace. See [Create an Azure Machine Learning workspace](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-manage-workspace?tabs=python).  

* Familiarity with Azure's [automated machine learning](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml) and [machine learning pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/concept-ml-pipelines) facilities and SDK.

# Getting Started
1. Setup a local conda environment.

    > ! Note: replace myenv with the environment name.
    - Create a new environment from an env.yaml file:

        `> conda env create --name myenv --file env.yaml`
    
        OR:
    - Update an existing environment:

        `> conda env update --name myenv --file env.yaml`

2. Configure your workspace settings in [config.json](config.json).

    ```
    {
    "subscription_id": "my_subscription_id",
    "resource_group": "my_resource_group",
    "workspace_name": "my_workspace_name",
    "tenant_id": "my_tenant_id",
    "compute_name": "my_compute_cluster",
    ...
    }
    ```

3. Register initial training dataset in [transform.py](src/training_pipes/transform/transform.py).
    ```
    training_tabular_dataset = Dataset.Tabular.register_pandas_dataframe(
            dataframe=train_df,
            target=workspaceblobstore,
            name=args.dataset_name,
            show_progress=True)
    ```

4. Run your automated ML training pipeline.

    ```
    > cd project_directory
    > python training_main.py
    ```

5. Run your scoring pipeline.

    ```
    > cd project_directory
    > python scoring_main.py
    ```

# Getting Help
This project is under active development by Alvin Haryanto.

If you have questions, comments, or just want to have a good old-fashioned chat about MLOps with Azure Machine Learning, please reach out to me at haryanto.alvin@gmail.com, alvin.haryanto@avanade.com, or [linkedin.com/in/alvinharyanto](https://www.linkedin.com/in/alvinharyanto).