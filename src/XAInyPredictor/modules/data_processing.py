import os
import math
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, zscore
import seaborn as sns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import StandardScaler


def process_raw_data(raw_df: pd.DataFrame, output_file: str | None = None, encoding_config: dict = None, target: str = 'class_target', allowed_columns: list | None = None) -> pd.DataFrame:
    """
    Processes raw data to be used by the XAI algorithm.

    :param raw_df: Pandas dataframe containing the raw data.
    :type raw_df: pd.DataFrame
    :param output_file: Path to the file where the output processed data frame
        will be saved. If None, it will not be saved.
    :type output_file: str | None
    :param target: Name of the column containing the target feature.
    :type target: str
    :param allowed_columns: List of column names to keep. If None, keeps all columns.
    :type allowed_columns: list | None
    :return: Pandas dataframe containing the processed data.
    :rtype: DataFrame
    """
    if output_file != None and os.path.exists(output_file):
        print("Processed dataset already exists! Skipping processing.")
        df = pd.read_csv(output_file)
        return df

    # Rename the target column to class_target and move it to the end of the df
    if target:
        raw_df = raw_df.rename(columns={target: 'class_target'})
        target_column = raw_df.pop('class_target')
        raw_df['class_target'] = target_column

    # Include target column within allowed_columns    
    if allowed_columns and ('class_target' not in allowed_columns):
        allowed_columns.append('class_target')

    # Get encoding dictionary
    if encoding_config is None:
        encoding_dict = {}
    else:
        encoding_dict = encoding_config.copy()
    encoding_dict.pop('class_target', None)

    # Process data
    clean_df = clean_data(raw_df, allowed_columns=allowed_columns)
    if target:
        target_prop_plot = check_target_proportion(clean_df)
    non_numeric_summary = check_non_numeric_features(clean_df)
    clean_nomissing_df = handle_missing_values(clean_df)
    num_feat_results = plot_distribution_numerical_features(clean_nomissing_df)
    cat_feat_results = plot_distribution_categorical_features(clean_nomissing_df)
    (outliers_iqr, outliers_z) = detect_outliers(clean_nomissing_df)
    cramers_results = calculate_cramers_v_matrix(clean_nomissing_df)
    norm_df = normalize_and_encode_features(clean_nomissing_df, encoding_dict)

    # Save the processed dataset to a CSV file
    if output_file != None:
        norm_df.to_csv(output_file, index=False)
        print(f"Processed dataset saved to {output_file}")
    return norm_df


def process_raw_form_data(raw_df: pd.DataFrame, example_raw_df: pd.DataFrame, encoding_config: dict = None, allowed_columns: list | None = None) -> pd.DataFrame:
    """
    Processes raw data from the XAInyPredictor form to be used by the XAI algorithm.

    :param raw_df: Pandas dataframe containing the raw data.
    :type raw_df: pd.DataFrame
    :param example_raw_df: Pandas dataframe containing the raw example data. The
        data is used as reference to imputate missing values.
    :type example_raw_df: pd.DataFrame
    :param encoding_config: Dictionary containing encoding mappings for categorical features.
    :type encoding_config: dict
    :param allowed_columns: List of column names to keep. If None, keeps all columns.
    :type allowed_columns: list | None
    :return: Pandas dataframe containing the processed data.
    :rtype: DataFrame
    """
    clean_df = clean_data(raw_df, allowed_columns)
    non_numeric_summary = check_non_numeric_features(clean_df)

    original_length = len(clean_df)
    clean_example_raw_df = clean_data(example_raw_df, allowed_columns)
    df_and_example = pd.concat([clean_df, clean_example_raw_df], ignore_index=True)
    df_and_example_nomissing = handle_missing_values(df_and_example)

    if encoding_config is None:
        encoding_dict = {}
    else:
        encoding_dict = encoding_config.copy()
    encoding_dict.pop('class_target', None)
    df_and_example_norm = normalize_and_encode_features(df_and_example_nomissing, encoding_dict)

    if original_length == 1:
        norm_df = df_and_example_norm.iloc[[0]]
    else:
        norm_df = df_and_example_norm.iloc[0:original_length]

    return norm_df


def clean_data(df: pd.DataFrame, allowed_columns: list | None = None) -> pd.DataFrame:
    """
    Processes raw data to keep required columns, remove trailing whitespaces
    and make sure missing values are represented as NaN.

    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :param allowed_columns: List of column names to keep. If None, keeps all columns.
    :type allowed_columns: list | None
    :return: Pandas dataframe containing the processed data.
    :rtype: pd.DataFrame
    """
    df.columns = df.columns.str.strip()

    if allowed_columns is not None:
        present_columns = [col for col in df.columns if col in allowed_columns]
        df = df[present_columns]

    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)
    df.replace('NA', np.nan, inplace=True)

    return df


def check_target_proportion(
        df: pd.DataFrame,
        target: str = 'class_target',
        plot_title: str = 'Proportions of the Target Feature (RAI-R)',
        x_label: str = 'RAI-R (NO: Response, Yes: No response)',
        y_label: str = 'Proportion'
    ) -> Figure:
    """
    Creates a bar plot showing the proportion of the target feature. 
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :param target: Name of the column containing the target feature.
    :type target: str
    :param plot_title: Title of the plot.
    :type plot_title: str
    :param x_label: Label of the x axis.
    :type x_label: str
    :param y_label: Label of the y axis.
    :type y_label: str
    :return: The bar plot figure object.
    :rtype: matplotlib.figure.Figure
    """
    # Check the proportions of the target feature
    target_counts = df[target].value_counts(normalize=True)
    print(f"Proportions of the target feature:")
    print(target_counts)

    # Visualize the proportions using a bar plot
    sns.barplot(y=target_counts.values, hue=target_counts.index, palette='viridis')
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    fig = plt.gcf() # get current figure
    return fig

def check_non_numeric_features(df: pd.DataFrame) -> dict:
    """
    Checks the non-numeric values in the features of a data frame. 
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Dict containing, for each column (key), the non-numeric values.
    :rtype: dict
    """
    # Dictionary to store non-numeric entries for each column
    non_numeric_summary = {}

    # Loop over each column and attempt conversion to numeric to find non-numeric entries
    for column in df.columns:
        # Identify non-numeric entries by attempting to convert to numeric and checking for NaNs
        non_numeric_values = df[column].apply(pd.to_numeric, errors='coerce').isna()

        # Store non-numeric entries if they exist
        if non_numeric_values.any():
            non_numeric_summary[column] = df[column][non_numeric_values].unique()

    # Display the non-numeric values in each column
    for column, values in non_numeric_summary.items():
        print(f"Non-numeric entries in '{column}':", values)

    return non_numeric_summary


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks the percentage of missing values for each feature in the input data
    frame and imputates new values.

    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Pandas dataframe containing the new data frame with the imputated values.
    :rtype: pd.DataFrame
    """
    missing_percent_df = check_percent_missing_values(df)

    proc_df = df.copy()

    proc_df = imputate_numerical_values_with_median(proc_df)
    proc_df = imputate_categorical_values_with_mode(proc_df)

    missing_percent_df2 = check_percent_missing_values(proc_df)

    print("Dataset after handling missing values:")
    print(proc_df.head())

    return proc_df


def check_percent_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Checks the percentage of missing values in the features of a data frame.
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Pandas dataframe containing the percentage of missing values for each feature.
    :rtype: pd.DataFrame
    """
    # Checking for missing values
    missing_values_info = df.isnull().sum()
    total_rows = len(df)

    # Calculate the percentage of missing values
    missing_percent = (missing_values_info / total_rows * 100).round(2)

    # Combine count and percentage into a single DataFrame for clarity
    missing_percent_df = pd.DataFrame({
        'Missing Values': missing_values_info,
        '% Total Values': missing_percent
    })

    print("Missing Values Summary:")
    print(missing_percent_df)

    return missing_percent_df


def imputate_values_for_feature_with_reference(df: pd.DataFrame,
                                               feature_to_imputate: str,
                                               reference_features: list) -> pd.DataFrame:
    """
    Imputates values for a feature based on the values from other features.
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :param feature_to_imputate: Name of the column from the feature that we want
        to imputate values.
    :type feature_to_imputate: str
    :param reference_features: List of column names from features to be used as
        reference to imputate values. All features must be enconded with numeric
        values. It should contain the feature_to_imputate.
    :type reference_features: list
    :return: Pandas dataframe containing the data frame with the imputated values.
    :rtype: pd.DataFrame
    """
    print(f"Imputating values for feature {feature_to_imputate}...")
    # Check if feature_to_imputate in reference_features
    if feature_to_imputate not in reference_features:
        reference_features.append(feature_to_imputate)

    # Select reference columns
    imputation_df = df[reference_features]

    # Stop if non-numeric column found
    for column in imputation_df.columns:
        # Identify non-numeric entries by attempting to convert to numeric and checking for NaNs
        non_numeric_values = imputation_df[column].apply(pd.to_numeric, errors='coerce').isna()

        # Check if any of the values is not nan
        non_numeric_mask = non_numeric_values.isna() & imputation_df[column].notna()

        # Stop if non-numeric values other than nan are found
        if non_numeric_mask.any():
            print(f"Non-numeric column found: {column}. Imputation of values for {feature_to_imputate} is not possible!")
            return df

    # Initialize and apply the IterativeImputer
    imputer = IterativeImputer(max_iter=10, random_state=0)
    imputation_df[reference_features] = imputer.fit_transform(imputation_df[reference_features])

    # Copy the imputed column back to the original DataFrame
    df[feature_to_imputate] = imputation_df[feature_to_imputate]

    # Check if any missing values remain in 'BMI'
    remaining_missing_values = df[feature_to_imputate].isna().sum()
    print(f"Missing values remaining after imputation for {feature_to_imputate}: {remaining_missing_values}")

    return df


def imputate_numerical_values_with_median(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputates missing numerical values by assigning the median value of the feature.
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Pandas dataframe containing the data frame with the imputated values.
    :rtype: pd.DataFrame
    """
    numerical_columns = df.select_dtypes(include=['number']).columns
    for col in numerical_columns:
        df[col] = df[col].fillna(df[col].median())
    return df


def imputate_categorical_values_with_mode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Imputates missing categorical values by assigning the mode value of the
    feature (i.e., the value that appears more often).
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Pandas dataframe containing the data frame with the imputated values.
    :rtype: pd.DataFrame
    """
    categorical_columns = df.select_dtypes(exclude=['number']).columns
    for col in categorical_columns:
        df[col] = df[col].fillna(df[col].mode()[0])
    return df


def plot_distribution_numerical_features(df: pd.DataFrame) -> dict:
    """
    Creates two panels with the histograms and boxplots of the numerical
    features respectively.
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Dictionary containing the plot figure, column names, and distribution summary.
    :rtype: dict
    """
    # Get numerical columns
    numerical_columns = df.select_dtypes(include=['number']).columns.tolist()

    if not numerical_columns:
        print("No numerical columns found in the dataframe.")
        return {
            'histogram': None,
            'boxplot': None,
            'columns': [],
            'distribution_info': None
        }

    print("Numerical columns:", numerical_columns)

    # Get distribution summary
    distribution_info = df.describe()
    print("Numerical Variables Distribution Summary:")
    print(distribution_info)

    # Set up a grid for numerical columns
    num_cols = len(numerical_columns)
    num_rows_cat = math.ceil(num_cols / 3)

    # Create figures with exact number of subplots
    hist_fig, hist_axes = plt.subplots(num_rows_cat, 3, figsize=(15, 5*num_rows_cat), squeeze=False)
    box_fig, box_axes = plt.subplots(num_rows_cat, 3, figsize=(15, 5*num_rows_cat), squeeze=False)

    # Create subplots
    for i, col in enumerate(numerical_columns):
        row = i // 3
        col_idx = i % 3
        
        # Create histogram
        hist_axes[row, col_idx].hist(df[col].dropna(), bins=30, alpha=0.7, edgecolor='black', linewidth=0.5)
        hist_axes[row, col_idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        hist_axes[row, col_idx].set_xlabel(col, fontsize=10)
        hist_axes[row, col_idx].set_ylabel('Frequency', fontsize=10)
        hist_axes[row, col_idx].grid(True, alpha=0.3)

        # Add mean to the histogram
        mean_val = df[col].mean()
        hist_axes[row, col_idx].axvline(mean_val, color='red', linestyle='--', linewidth=2, 
                                  label=f'Mean: {mean_val:.2f}')
        hist_axes[row, col_idx].legend()

        # Create boxplot
        box_axes[row, col_idx].boxplot(df[col].dropna(), vert=True) # set vert=True for vertical orientation
        box_axes[row, col_idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        box_axes[row, col_idx].set_xlabel(col, fontsize=10)

    # Hide empty subplots
    for i in range(num_cols, num_rows_cat * 3):
        row = i // 3
        col_idx = i % 3
        hist_axes[row, col_idx].set_visible(False)
        box_axes[row, col_idx].set_visible(False)

    plt.tight_layout()

    return {
        'histogram': hist_fig,
        'boxplot': box_fig,
        'columns': numerical_columns,
        'distribution_info': distribution_info
    }


def plot_distribution_categorical_features(df: pd.DataFrame) -> dict:
    """
    Creates a panel with the barplots of the categorical features.
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Dictionary containing the plot figure and column names.
    :rtype: dict
    """
    # Get categorical columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns.tolist()

    if not categorical_columns:
        print("No categorical columns found in the dataframe.")
        return {'plot': None, 'columns': [], 'distribution_info': None}

    print("Categorical columns:", categorical_columns)

    # Set up a grid for categorical columns
    num_cols = len(categorical_columns)
    num_rows_cat = math.ceil(num_cols / 3)

    # Create figure with exact number of subplots
    fig, axes = plt.subplots(num_rows_cat, 3, figsize=(15, 5*num_rows_cat), squeeze=False)

    # Create subplots
    for i, col in enumerate(categorical_columns):
        row = i // 3
        col_idx = i % 3
        
        axes[row, col_idx].bar(df[col].value_counts().index, df[col].value_counts().values, edgecolor='black', linewidth=0.5)
        axes[row, col_idx].set_title(f'Distribution of {col}', fontsize=12, fontweight='bold')
        axes[row, col_idx].set_xlabel(col, fontsize=10)
        axes[row, col_idx].set_ylabel('Count', fontsize=10)
        axes[row, col_idx].tick_params(axis='x', rotation=45)
        axes[row, col_idx].grid(True, alpha=0.3)
    
    # Hide empty subplots
    for i in range(num_cols, num_rows_cat * 3):
        row = i // 3
        col_idx = i % 3
        axes[row, col_idx].set_visible(False)

    plt.tight_layout()
    
    return {
        'plot': fig,
        'columns': categorical_columns
    }


def detect_outliers_iqr(df: pd.DataFrame, columns: list) -> dict:
    """
    Detects outliers using the IQR method.

    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :param columns: List containing the numerical columns to analyze
    :type columns: list
    :return: Dictionary containing for each column (key) the dataframe with the outliers.
    :rtype: dict
    """
    outliers = {}
    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        # Define bounds for outliers
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        # Identify outliers
        outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
    return outliers


def detect_outliers_zscore(df: pd.DataFrame, columns: list, z_score_threshold: float = 3) -> dict:
    """
    Detects outliers using the z-score method.

    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :param columns: List containing the numerical columns to analyze
    :type columns: list
    :return: Dictionary containing for each column (key) the dataframe with the outliers.
    :rtype: dict
    """
    # Calculate z-scores for numerical columns
    z_scores = df[columns].apply(zscore)
    
    # Identify outliers with z-score > z_score_threshold or < -z_score_threshold
    outliers_mask = (z_scores > z_score_threshold) | (z_scores < -z_score_threshold)
    
    # Create dictionary with outlier rows for each column
    outlier_data = {}
    for col in columns:
        # Get rows where this column has outliers
        outlier_rows = df[outliers_mask[col]]
        outlier_data[col] = outlier_rows

    return outlier_data


def detect_outliers(df: pd.DataFrame) -> tuple:
    """
    Detects outliers using both IQR and z-score methods.

    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Dictionary containing the number of outliers for each column (key).
    :rtype: dict
    """
    # Get numerical columns
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

    # Detect outliers using IQR
    outliers_iqr = detect_outliers_iqr(df, numerical_cols)
    for col, outliers in outliers_iqr.items():
        print(f"{col} has {len(outliers)} outliers based on IQR.")

    # Identify outliers with z-score > 3 or < -3
    outliers_z = detect_outliers_zscore(df, numerical_cols, 3)

    # Display outliers using z-score
    for col in numerical_cols:
        print(f"{col} has {len(outliers_z)} outliers based on z-scores.")
    return (outliers_iqr, outliers_z)


def cramers_v(x: list | pd.Series, y: list | pd.Series) -> float:
    """
    Calculate Cramer's V for two lists of categorical variables.
    
    :param x: List of values for categorical variable x.
    :type x: list | pd.Series
    :param y:  List of values for categorical variable y.
    :type y: list | pd.Series
    :return: Float corresponding to Cramer's V value.
    :rtype: float
    """
    contingency_table = pd.crosstab(x, y)
    chi2 = chi2_contingency(contingency_table)[0]
    n = contingency_table.sum().sum()
    r, k = contingency_table.shape
    return np.sqrt(chi2 / (n * (min(r, k) - 1)))


def calculate_cramers_v_matrix(df: pd.DataFrame) -> dict:
    """
    Calculate matrix of Cramer's V values between all categorical features of a
    data frame.
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Dict containing the plot and the matrix.
    :rtype: dict
    """
    # Get categorical columns
    categorical_columns = df.select_dtypes(exclude=['number']).columns

    # Calculate Cramér's V matrix
    cramers_v_matrix = pd.DataFrame(index=categorical_columns, columns=categorical_columns, dtype=float)
    for col1 in categorical_columns:
        for col2 in categorical_columns:
            if col1 != col2:
                cramers_v_matrix.loc[col1, col2] = cramers_v(df[col1], df[col2])
            else:
                cramers_v_matrix.loc[col1, col2] = np.nan

    # Create a mask for the upper triangle
    mask = np.triu(np.ones_like(cramers_v_matrix, dtype=bool))

    # Visualize the lower triangle of the Cramér's V matrix
    sns.heatmap(cramers_v_matrix, mask=mask, annot=True, cmap='coolwarm', square=True, cbar_kws={'label': "Cramér's V"})
    plt.title("Cramér's V Matrix for Categorical Variables")
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.show()
    fig = plt.gcf() # get current figure
    return {'plot': fig,
            'matrix': cramers_v_matrix}


def normalize_numerical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize numerical features of a data frame using StandardScaler.
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :return: Pandas dataframe containing the normalized numerical features.
    :rtype: DataFrame
    """

    # Get numerical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Normalize numerical columns
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    return df


def create_rai_encoding_dict() -> dict:
    """
    Create dictionary of the encoding of categorical variables in RAI data.
    
    :return: Dictionary of the encoding.
    :rtype: dict
    """
    encoding_dict = {}
    encoding_dict['Gender'] = {'M': 0, 'F': 1}
    encoding_dict['Histology'] = {'OTC': 1,'PTC': 2,'FTC': 3,'PDTC': 4}
    encoding_dict['Tumor stage'] = {'T1a': 1, 'T1b': 2, 'T2': 3, 'T3a': 4, 'T3b': 5, 'T4a': 6, 'T4b': 7, 'Tx': -1}
    encoding_dict['Node stage'] = {'N0': 0, 'N1a': 1, 'N1b': 2, 'Nx': -1}
    encoding_dict['Metastases'] = {'Mx': 0, 'M1': 1}
    encoding_dict['ETE'] = {'NO': 0, 'YES': 1}
    encoding_dict['Multifocality'] = {'NO': 0, 'YES': 1}
    encoding_dict['Vascular invasion'] = {'NO': 0, 'YES': 1}
    encoding_dict['Resection'] = {'R0': 0,'R1': 1,'R2': 2}
    encoding_dict['ATA risk'] = {'Low': 1, 'Intermediate': 2, 'High': 3}
    encoding_dict['Goal of RAI'] = {'ABLATION': 1,'ADIUVANT': 2,'THERAPEUTIC': 3}
    encoding_dict['Persistence'] = {'Excellent Response': 1,'Biochemical': 2,'Indeterminate': 3,'Structural': 4,'still in determination': 5}
    encoding_dict['class_target'] = {'NO': 0, 'YES': 1}
    return encoding_dict


def encode_categorical_features(
        df: pd.DataFrame,
        encoding_dict: dict = create_rai_encoding_dict()
    ) -> pd.DataFrame:
    """
    Encode categorical features of a data frame using a dictionary.
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :param encoding_dict: Dictionary of the encoding.
    :type encoding_dict: dict
    :return: Pandas dataframe containing the encoded categorical features.
    :rtype: DataFrame
    """
    for feat, feat_dict in encoding_dict.items():
        df[feat] = df[feat].map(feat_dict)
    return df


def normalize_and_encode_features(
        df: pd.DataFrame,
        encoding_dict: dict = create_rai_encoding_dict()
    ) -> pd.DataFrame:
    """
    Normalize or encode the features of a data frame.
    
    :param df: Pandas dataframe containing the raw data.
    :type df: pd.DataFrame
    :param encoding_dict: Dictionary of the encoding for categorical features.
    :type encoding_dict: dict
    :return: Pandas dataframe containing the normalized/encoded features.
    :rtype: DataFrame
    """
    df = normalize_numerical_features(df)
    df = encode_categorical_features(df, encoding_dict)
    print("Dataset after standardization of numerical and categorical features:")
    print(df.head())
    return df
