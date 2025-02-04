# Post Client b2b pricing (WTP model)

Model Training:
- Iterates through countries and configurations:
	- For each country and configuration combination, it:
		- Sets the current configuration
		- Prints progress (e.g., "Country 1/1 - Config 1/1")
		- Prepares the dataset:
			- Takes data for the current country from data_dict
			- If cfg.sample is set, samples a subset of data
			- Cleans the dataset using _clean_dataset (removes outliers, handles missing values)
				- Data Filtering:
					- Removes rows where discount is outside the valid range (clip_min to clip_max)
					- Removes rows with missing acceptance values (accepted.notna())
					- If a country is specified, filters to only that country's data
					- Removes duplicate rows
				- Column Validation:
					- Checks three types of columns:
						- bin_cols: Binary columns
						- cat_cols: Categorical columns
						- num_cols: Numerical columns
					- Issues warnings for any missing columns
					- Creates new lists containing only the columns that exist in the data
				- Returns:
					- Cleaned dataset
					- List of validated columns (bcn_cols) for each type (binary, categorical, numerical)
		- Splits data into training and test sets using split_trn_tst
		- Augments the training data:
			- Uses augment_datato create additional synthetic training examples
			- This helps improve model robustness
		- Trains the model:
			- Splits data into features (X) and target (y)
			- Determines number of features after preprocessing
			- Calls _train_model which:
 				- Creates an Optuna study for hyperparameter optimization
				- Trains an XGBoost classifier with the best parameters
				- Returns the trained model and study results
		- Saves results (if save_immediately_to is provided):
			- Creates directory structure for results
			- Saves:
				- Classifier evaluation metrics
				- ROM evaluation metrics
				- Model configuration
				- Evaluation details
	- The results are stored in the trainer's internal dictionaries:
		- elf.models: Trained models
		- self.studies: Optimization study results
		- self.data: Training/test data splits
		- Various evaluation dataframes
The key feature is that it handles the entire pipeline from data preparation through training to evaluation and result storage, with support for multiple countries and configurations.


# pricing

[![Powered by Kedro](https://img.shields.io/badge/powered_by-kedro-ffc900?logo=kedro)](https://kedro.org)

## Overview

This is your new Kedro project, which was generated using `kedro 0.19.11`.

Take a look at the [Kedro documentation](https://docs.kedro.org) to get started.

## Rules and guidelines

In order to get the best out of the template:

* Don't remove any lines from the `.gitignore` file we provide
* Make sure your results can be reproduced by following a data engineering convention
* Don't commit data to your repository
* Don't commit any credentials or your local configuration to your repository. Keep all your credentials and local configuration in `conf/local/`

## How to install dependencies

Declare any dependencies in `requirements.txt` for `pip` installation.

To install them, run:

```
pip install -r requirements.txt
```

## How to run your Kedro pipeline

You can run your Kedro project with:

```
kedro run
```

## How to test your Kedro project

Have a look at the file `src/tests/test_run.py` for instructions on how to write your tests. You can run your tests as follows:

```
pytest
```

You can configure the coverage threshold in your project's `pyproject.toml` file under the `[tool.coverage.report]` section.


## Project dependencies

To see and update the dependency requirements for your project use `requirements.txt`. You can install the project requirements with `pip install -r requirements.txt`.

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in scope: `context`, 'session', `catalog`, and `pipelines`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so once you have run `pip install -r requirements.txt` you will not need to take any extra steps before you use them.

### Jupyter
To use Jupyter notebooks in your Kedro project, you need to install Jupyter:

```
pip install jupyter
```

After installing Jupyter, you can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab
To use JupyterLab, you need to install it:

```
pip install jupyterlab
```

You can also start JupyterLab:

```
kedro jupyter lab
```

### IPython
And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`
To automatically strip out all output cell contents before committing to `git`, you can use tools like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in `.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed to `git`.

> *Note:* Your output cells will be retained locally.

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html)
