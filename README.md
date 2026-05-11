# XAInyPredictor

XAInyPredictor is a Python-based application built using Shiny for running the Radioiodine (RAI) classification model trained with a dataset of 294 records and 18 features and visually exploring the features that contribute most to the classificaiton.

[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)

## Table of Contents

- [XAInyPredictor](#xainypredictor)
  - [Table of Contents](#table-of-contents)
  - [Getting Started](#getting-started)
    - [What is XAInyPredictor?](#what-is-xainypredictor)
    - [Installation](#installation)
    - [Initiate the app](#initiate-the-app)
  - [Usage](#usage)
    - [1. Data Input](#1-data-input)
    - [2. Data Exploration](#2-data-exploration)
    - [3. Prediction](#3-prediction)
  - [Development](#development)
    - [Architecture](#architecture)
    - [Testing](#testing)
    - [Building](#building)
  - [Documentation](#documentation)

## Getting Started

### What is XAInyPredictor?

XAInyPredictor is a Python-based application built using Shiny that allows to run the **Radioiodine (RAI) classification model** on patient data and explore the results through intuitive data tables and plots.

### What is the RAI classification model?

The RAI classification model is a machine learning algorithm based on Kolmogorov-Arnold Networks (KAN) that is used to identify thyroid cancer patients that are more prone to develop Radioiodine-refractory disease (RAI-R).

RAI-R typically develops in a subset of patients with differentiated thyroid cancer —primarily papillary and follicular thyroid carcinomas—who lose the ability to uptake or respond to radioactive iodine (I-131). This condition is associated with poor prognosis and presents significant therapeutic challenges.

The RAI classification model was trained in a dataset of 295 patients and 16 features.

### Installation

Install XAInyPredictor using `pip`. First, make sure that pip is updated:

```bash
python -m pip install --upgrade pip
```

Then, there are two options to install the app:

* From a cloned repository:

```bash
git clone https://github.com/REPO4EU/XAInyPredictor.git
pip install path/to/XAInyPredictor
```

* From a `.zip` file. In this case, first download the zip file from the [GitHub website](https://github.com/REPO4EU/XAInyPredictor/archive/refs/heads/main.zip) and then execute the following command:

```bash
pip install XAInyPredictor_version.zip
```

### Initiate the app

The basic command to start the application is the following:

```bash
xainypredictor
```

This will automatically open a browser web page [http://127.0.0.1:8001](http://127.0.0.1:8001) to access the XAInyPredictor interface.

Alternatively, you can use this command:

```bash
shiny run src/XAInyPredictor/app.py
```

Upon running the application, open a web browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) to access the XAInyPredictor interface.

If you get the following error:

```bash
ERROR:    [Errno 48] error while attempting to bind on address ('127.0.0.1', 8000): [errno 48] address already in use
```

It means that the port 8000 is already being used. To solve this, specify a different port, like this:

```bash
shiny run src/XAInyPredictor/app.py --port 8787
```

Another alternative is to use Docker. First build the docker image:

```bash
docker build -t xainypredictor-docker .
```

And then, run the docker image:

```bash
docker run -p 8000:8000 xainypredictor-docker
```

The app should be available in [http://localhost:8000](http://localhost:8000).


## Usage

### 1. Data Input

The first step for using the app is to provide input data. The app gives you three options in the left panel:
* **Manual Entry**: This option provides you with a form in which patient data can be entered manually. Every time a patient is introduced, it appears in the table below the form (see **Current Patient Cohort**). Patient data can also be removed by clicking at one or multiple rows of the table (by using *Control + Click*) and clicking the Delete Selected button.
* **Upload File**: This gives you the possibility to upload a file containing all patient data that you want to analyze. The data can be introduced using a tab-separated file (TSV), comma-separated file (CSV) or Excel file that contains the required columns. Please, use the [Excel template](https://github.com/REPO4EU/XAInyPredictor/blob/development/src/XAInyPredictor/shinyapp/www/data_template.xlsx) as reference.
* **Example Cohort**: You can also use the app with the example cohort, which is the data that has been used to train the model.

The features used to train the model are the following:

* **Age**: Age of the participant in years.
* **Gender**: The gender of the participant. Possible values: M (Male) or F (Female).
* **BMI**: The Body Mass Index of the participant.
* **Tumor size (cm)**: The size of the tumor in centimeters.
* **Tumor stage**: The stage of the tumor, based on standard staging criteria. Possible values: T1a, T1b, T2, T3a, T3b, T4a, T4b, Tx.
* **Node stage**: The stage of the lymph node involvement. Possible values: N0, N1a, N1b, Nx.
* **Metastases**: Indicates the presence of metastases. Possible values: Mx or M1.
* **ATA risk**: The risk classification according to the American Thyroid Association. Possible values: Low, Intermediate, High.
* **Histology**: The type of tumor histology observed in the participant. Possible values: OTC (oncocytic thyroid carcinoma), PTC (papillary thyroid carcinoma), FTC (follicular thyroid carcinoma), PDTC (poorly differentiated thyroid carcinoma).
* **ETE**: Whether extrathyroidal extension was observed. Possible values: YES or NO.
* **Multifocality**: Indicates whether the tumor was multifocal. Possible values: YES or NO.
* **Vascular invasion**: Whether there was evidence of vascular invasion. Possible values: YES or NO.
* **Resection**: The status of the surgical resection. Possible values: R0, R1, R2.
* **Goal of RAI**: The goal of the radioiodine treatment. Possible values: THERAPEUTIC, ’ADIUVANT, ABLATION.


You can use this [template](https://github.com/REPO4EU/XAInyPredictor/blob/development/src/XAInyPredictor/shinyapp/www/data_template.xlsx) as example:

```tsv
ID	Age	Gender	BMI	Tumor size (cm)	Tumor stage	Node stage	Metastases	ATA risk	Histology	ETE	Multifocality	Vascular invasion	Resection	Goal of RAI
RAI001	50	M	27	5	T1a	N0	M1	Low	PTC	NO	NO	NO	R0	ABLATION
RAI002	23	F	19	2	T1b	N1a	Mx	Intermediate	OTC	NO	YES	NO	R1	THERAPEUTIC
```


### 2. Data Exploration

The second step (**Data Exploration**) allows you to explore the distribution (center panel) and statistics (right panel) of each features in your input data. The three options in the left panel allow you to control the visualization:
* **Feature to plot**: To select the feature to visualize. A numeric feature (e.g., Age, BMI...) will produce a hisogram plot, and a categorical feature (e.g., Tumor stage, Node stage...) will produce a bar plot.
* **Reference population**: This allows you to select the population that you want to show in the distribution plot. The options are **Input data** (it will show the distribution of the input data provided by the user) and **Model** (it will show the distribution of the data used to train the model).
* **Select patient**: This allows you to compare the feature value from a specific participant with the reference population selected. The value of the patient will be highlighted in red in the plot.


### 3. Prediction

The third step (**Prediction**) shows the prediction results and some plots to help interpret the prediction. The results are shown in the center, divided in two panels:
* **Prediction Results Table**: The table shows the predictions of the model, in two columns:
  * **Probability**: Indicates the probability of a participant to be RAI-Refractory (resistant). The higher the probability, the more likely it is.
  * **RAI-R class**: Classifies the patient in Refractory or Not Refractory. YES = Refractory (i.e., resistant to the treatment); NO = Not Refractory (non-resistant to the treatment).

* **Results Plot**: There are two types of plots (depending on the view selected in the left panel):
  * **Feature Analysis (Radar)**: The radar plot compares the values of the features from a selected patient with three different average values. Comparing these values help to evaluate if a patient has been correctly classified or not. The average values are:
    * Average all patients: The average values from all patients in the model.
    * Avg. RAI-R negative: The average values from all RAI-R negative patients in the model (green).
    * Avg. RAI-R positive: The average values from all RAI-R positive patients in the model (red).
  * **Distance Analysis (Curves)**: The curves plot displays the distribution of values from a specific feature across the patients in the model, ordered from lowest to highest (blue dots). It highlights in red the patient selected, putting in context the feature values of the patient in comparison with the values from the rest of patients.

The left panel gives you different options to modify the visualization of the results:
* **Select patient**: This allows to select the participant from which the results will be visualized. This prediction results will be highlighte in yellow in the Prediction Results Table, and the values of the features will be shown in red in the Radar and Curve plots.
* **Min. % false negatives**: This threshold controls the False Negative Ratio of the model.
  * 0% False Negative Ratio: We refuse to miss any true RAI-Refractory patients. The model will classify a patient as Refractory even if the probability is low (High Sensitivity).
  * Higher False Negative Ratio: We accept missing some Refractory patients to ensure those we treat are definitely Refractory (High Specificity).
* **Select View**: Allows you to change the results plot (either Radar or Curve plot).
* **Select features to view**: Allows you to select the features to be visualized in the Results Plot.
* **Radar Plot Elements:**: Allows you to control the different elements shown in the Radar Plot.


## Development

### Architecture

The organization of the XAInyPredictor repository is summarized here:

```
XAInyPredictor/
├── build
├── CHANGELOG.md
├── Dockerfile                         <- Definition of the Docker image
├── docs                               <- Documentation generated using Doxygen
├── pyproject.toml                     <- Definition of the python project
├── README.md                          <- The top-level README for developers using this package.
├── requirements.txt                   <- Python packages required
├── setup.md
├── sonar-project.properties
├── src                                <- Source code
│   ├── XAInyPredictor
│   │   ├── __init__.py
│   │   ├── __pycache__
│   │   ├── app.py                    <- Main script that defines the app
|   |   |-- data
|   |   |   |-- Radioiodine\ database_October_RUMC.xlsx  <- Dataset used to train the model
|   |   |   |-- current_best_formula.txt  <- Formula obtained from training the model
|   |   |   |-- formula.pkl           <- Pickle file with formula obtained from training the model
|   |   |   |-- mock.csv              <- Mock raw data
|   |   |   |-- mock_processed.csv    <- Mock processed data
|   |   |   |-- processed_data.csv    <- Dataset used to train the model normalized
|   |   |   `-- raw_data.csv          <- Dataset used to train the model in csv
│   │   ├── data_processing_test.ipynb  <- Python notebook to test the data processing pipeline step by step (could be removed)
│   │   ├── main.py                   <- Script to launch the app
│   │   ├── modules
|   |   |   |-- data_processing.py    <- Script the data processing functions (to clean and normalize the data)
|   |   |   `-- rai.py                <- Script containing the functions of the RAI model
|   |   `-- shinyapp                  <- Scripts for specific aspects of the app
|   |       |-- __init__.py
|   |       |-- data_exploration.py   <- Data Exploration tab
|   |       |-- data_input.py         <- Data Input tab
|   |       |-- prediction.py         <- Prediction tab
|   |       `-- www                   <- Scripts related with the layout of the app
│   └── XAInyPredictor.egg-info
└── test                              <- Unit tests
```

### Testing

No tests have been implemented for the moment.

### Building

Build XAInyPredictor with:

```bash
python3 -m build
```

## Documentation

Documentation is generated in HTML during CI/CD, but it can be manually generated with:

```bash
doxygen ./docs/Doxyfile
```

The generated html files will be available at `docs/html` folder.
