from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections import register_projection
from matplotlib.projections.polar import PolarAxes
import matplotlib.pyplot as plt
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D
import logging
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
import sympy  # See https://github.com/knottwill/CoxKAN/blob/main/reprod/results.ipynb

logger = logging.getLogger(__name__)
_DELTA_XAI_FN_CACHE = {}


def split_data_with_known_target(input_df, target='class_target', test_split=0.2):
    """
    Splits the dataset into target (the label we want to classify) which is
    stored in the y variable, and features (the columns we use to predict the
    target) which is stored in the x variable. Then, splits each of them into
    train and test sets using the proportion indicated in test_split variable.
    :param input_df: pandas dataframe containing the features and target
    :param target: name of the target variable
    :param test_split: proportion of the data used for the test set
    :return: dataset divided into train or test and features (x) or target (y)
    """
    # Assign X to all features and y to the target class
    X, y = input_df.drop(columns=[target]), input_df[target]

    # Reformat column names so that there are no spaces
    new_cols = [col.replace(' ', '_') for col in X.columns]
    X.columns = new_cols

    # Split data into train and test
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=test_split, random_state=0)

    return x_train, x_test, y_train, y_test


def read_delta_xai_formula(formula_pkl_file: str, current_best_formula_file: str, simplify_formula: bool = True):
    """
    Reads the XAI output pickle file formula.pkl, which contains the formula of
    the XAI model, and simplifies it using the package sympy. Reads the file
    current_best_formula.txt and extracts the features names.
    :param formula_pkl_file: Path to the Pickle file where the XAI formula is
        stored.
    :param current_best_formula_file: Path to the file where the variable names
        are extracted.
    :return:
        - delta_formula: sympy object containing the formula
        - feature_mappings: Dict containing the mappings from formula variables to feature names
        - feature_order: List containing the feature names in the order of the formula
        - features_in_formula: List containing the feature names that are in the formula
    """
    with open(formula_pkl_file, 'rb') as f:
        formula = pd.read_pickle(f)

    # Show the formula saved
    logger.debug("Logit 0 formula: %s", formula[0])
    logger.debug("Logit 1 formula: %s", formula[1])

    # Simplify formula
    delta_formula = formula[1] - formula[0]
    if simplify_formula:
        delta_formula = sympy.simplify(delta_formula)
    logger.debug("Delta formula simplified: %s", delta_formula)

    # Read best formula file and extract variables
    with open(current_best_formula_file, 'r') as f:
        for line in f:
            if line.startswith('Variables:'):
                feature_order = eval(line.strip().split(': ')[1])
    feature_mappings = {str(idx+1) : feat_name for idx, feat_name in enumerate(feature_order)}
    logger.debug("Feature mappings: %s", '; '.join([f'x_{idx} = {feat_name}' for idx, feat_name in feature_mappings.items()]))

    # Get features in formula
    vars_in_formula = set(re.findall(r'x_(\d+)', str(delta_formula)))
    features_in_formula = [feature_mappings[str(var)] for var in sorted(list(map(int, vars_in_formula)))]

    return delta_formula, feature_mappings, feature_order, features_in_formula


def delta_xai(d_form, x_in: pd.DataFrame, feature_order: list | None = None):
    """
    Calculates probability associated to the classification of the target class
    based on the KAAM formula calculated using the default training set.
    :param d_form: sympy object containing the formula
    :param x_in: Pandas dataframe containing the features to be used for the
    classification
    :param feature_order: List containing the feature names that will be mapped
        in d_form, using the exact order. If None, the order will be extracted
        from x_in.
    :return: Pandas series containing the probability values
    """
    # If feature_order is provided, use it to map columns. If not, we will use
    # the order from x_in
    if feature_order is not None:
        # Verify that all feature names exist in the DataFrame
        missing_cols = set(feature_order) - set(x_in.columns)
        if missing_cols:
            raise ValueError(f"Missing columns in DataFrame: {missing_cols}")

        # Reorder DataFrame columns to match feature_order
        x_in = x_in[feature_order]

    x_d = x_in.to_numpy()
    n_features = x_d.shape[1]
    delta = np.zeros(
        (x_d.shape[0], n_features + 1))  # One input per covariate, one extra output for the constant term

    cache_key = (str(d_form), tuple(x_in.columns))
    if cache_key not in _DELTA_XAI_FN_CACHE:
        symbols = [sympy.Symbol(f"x_{idx + 1}") for idx in range(n_features)]
        zero_subs = {symbol: 0 for symbol in symbols}
        const = float(d_form.subs(zero_subs))
        partial_fns = []
        for symbol in symbols:
            partial_subs = {other_symbol: 0 for other_symbol in symbols if other_symbol != symbol}
            partial_expr = d_form.subs(partial_subs)
            partial_fns.append(sympy.lambdify(symbol, partial_expr, "numpy"))
        _DELTA_XAI_FN_CACHE[cache_key] = const, partial_fns

    const, partial_fns = _DELTA_XAI_FN_CACHE[cache_key]
    for j, partial_fn in enumerate(partial_fns):
        values = partial_fn(x_d[:, j])
        delta[:, j] = np.asarray(values, dtype=float)

    delta[:, -1] = const
    delta[:, :-1] -= const  # Subtract the constant term from all the other terms (only account for it once)
    delta = pd.DataFrame(delta, index=x_in.index, columns=x_in.columns.tolist() + ['const'])
    delta["pred_prob"] = 1. / (1 + np.exp(-delta.values.sum(axis=1)))
    return delta


def threshold_for_target_fnr(y_true: list, y_prob: list, target_fnr: float=0):
    """
    Find the highest threshold such that FNR <= target_fnr.
    :param y_true: True binary labels (0 or 1).
    :type y_true: list
    :param y_prob: Predicted probabilities for the positive class.
    :type y_prob: list
    :param target_fnr: Desired false negative rate (e.g., 0.05 for 5%).
    :type target_fnr: float
    :return:
        threshold: Decision threshold achieving the target FNR
        achieved_fnr: The actual FNR at that threshold
    """

    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    # Thresholds sorted from high → low
    thresholds = np.unique(y_prob)[::-1]

    positives = np.sum(y_true == 1)
    if positives == 0:
        return 0.5, 0.0

    best_threshold = None
    best_fnr = None

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)

        fn = np.sum((y_true == 1) & (y_pred == 0))
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fnr = fn / (fn + tp)

        if fnr <= target_fnr:
            best_threshold = t
            best_fnr = fnr
            break  # highest threshold satisfying the constraint

    if best_threshold is None:
        # Even threshold = 0 doesn't reach target FNR
        return 0.0, fnr

    return best_threshold, best_fnr


def analyze_patient(
        patient_id,
        df,
        delta_train,
        delta_test,
        x_train,
        y_train,
        x_test,
        features_to_plot=None,
        n_dists=5,
        max_plot_curves=10,
        show_closest_radial=True,
        show_average_radial=True,
        show_average_class0_radial=True,
        show_average_class1_radial=True,
        neg_class_label="Negative",
        pos_class_label="Positive",
        entity_label="Patient",
    ):

    logger.debug("Analyzing patient %s", patient_id)

    # Get feature values from patient selected and save them in delta_patient
    patient_index = df[df['ID'] == int(patient_id)].index.to_list()[0]
    if features_to_plot != None and len(features_to_plot) > 0:
        features_to_plot = [feat.replace(' ', '_') for feat in features_to_plot]
        features_to_plot = [x for x in features_to_plot if x in delta_test.columns.tolist() and x not in ["const", "pred_prob"]]
    else:
        features_to_plot = [x for x in delta_test.columns.tolist() if x not in ["const", "pred_prob"]]
    feature_names = [
            x for x in features_to_plot 
            if x in delta_test.columns.tolist() 
            and x in df.columns.tolist() 
            and x not in ["const", "pred_prob"]
    ]

    delta_train = delta_train[feature_names + ["const", "pred_prob"]]
    delta_test = delta_test[feature_names + ["const", "pred_prob"]]
    feature_idx = [delta_test.columns.get_loc(c) for c in feature_names if c in df]
    patient_info = pd.DataFrame(x_test.loc[patient_index].to_numpy().reshape([1, -1]), columns=x_test.columns.to_list()).drop(['class_target'], axis=1, errors='ignore')
    delta_patient = pd.DataFrame(delta_test.loc[patient_index].to_numpy().reshape([1, -1]), columns=delta_test.columns.tolist())
    pred_proba_patient = float(delta_patient["pred_prob"].to_list()[0])
    delta_patient = delta_patient.drop(['pred_prob'], axis=1, errors='ignore')

    # Get only the values of the training set
    delta_train_values = delta_train.drop(['pred_prob'], axis=1, errors='ignore').values
    delta_train_values_noconst = delta_train.drop(['pred_prob', 'const'], axis=1, errors='ignore').values

    # Get average training set values for each class
    delta_train_class0_values = delta_train[delta_train.index.isin(y_train[y_train.values == 0].index.to_list())].drop(['pred_prob'], axis=1, errors='ignore').values
    delta_train_class1_values = delta_train[delta_train.index.isin(y_train[y_train.values == 1].index.to_list())].drop(['pred_prob'], axis=1, errors='ignore').values
    delta_train_class0_values_noconst = delta_train[delta_train.index.isin(y_train[y_train.values == 0].index.to_list())].drop(['pred_prob', 'const'], axis=1, errors='ignore').values
    delta_train_class1_values_noconst = delta_train[delta_train.index.isin(y_train[y_train.values == 1].index.to_list())].drop(['pred_prob', 'const'], axis=1, errors='ignore').values

    # Find the n_dists closest patients in the training set
    dists = np.linalg.norm(delta_train_values - delta_patient.values, axis=1)
    idx_closest = np.argsort(dists)[:n_dists].tolist()
    idx_closest

    # RADAR PLOT
    n_feats = len(feature_names)
    if n_feats >= 3:  # We need at least 3 features to plot a proper radar plot
        theta = radar_factory(n_feats, frame='polygon')
        avg_proba = 1 / (1 + np.exp(-delta_train_values.mean(axis=0).sum())) * np.ones(len(feature_names))  # All average values averaged
        avg_proba_class0 = 1 / (1 + np.exp(-delta_train_class0_values.mean(axis=0).sum())) * np.ones(len(feature_names))
        avg_proba_class1 = 1 / (1 + np.exp(-delta_train_class1_values.mean(axis=0).sum())) * np.ones(len(feature_names))
        # avg_proba = 1 / (1 + np.exp(-delta_train_values_noconst.mean(axis=0)))
        # avg_proba_class0 = 1 / (1 + np.exp(-delta_train_class0_values_noconst.mean(axis=0)))
        # avg_proba_class1 = 1 / (1 + np.exp(-delta_train_class1_values_noconst.mean(axis=0)))
        title = f"{entity_label} {patient_id} (prob = {pred_proba_patient:.2f}, mean = {avg_proba[0]:.2f})"
        fig_radar, ax = plt.subplots(figsize=(7.8, 5.8), subplot_kw=dict(projection='radar'))
        fig_radar.subplots_adjust(left=0.32, right=0.96, top=0.86, bottom=0.08)
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, position=(0.5, 1.1), ha='center', weight='bold')
        ax.set_yticklabels([]) # Remove radial axis numbers

        # Plot the average of all train patients
        if show_average_radial:
            _ = ax.plot(theta, avg_proba, label='Average all patients', color='b')
            ax.fill(theta, avg_proba, alpha=0.1, color='b')

        # Plot the average for class 0
        if show_average_class0_radial:
            _ = ax.plot(theta, avg_proba_class0, label=f'Avg. {neg_class_label}', color='g')
            #ax.fill(theta, avg_proba_class0, alpha=0.1, color='g')

        # Plot the average for class 1
        if show_average_class1_radial:
            _ = ax.plot(theta, avg_proba_class1, label=f'Avg. {pos_class_label}', color='yellow')
            #ax.fill(theta, avg_proba_class1, alpha=0.1, color='yellow')

        # Prepare for individual patient plotting
        avg_delta = delta_train_values.mean(axis=0)[None, :]
        avg_matrix = np.repeat(avg_delta, delta_train_values.shape[1], axis=0)

        # Plot the closest patients
        if show_closest_radial:
            for j in range(n_dists):
                np.fill_diagonal(avg_matrix, delta_train_values[idx_closest[j]])
                pat_proba = 1 / (1 + np.exp(-avg_matrix.sum(axis=1)))
                _ = ax.plot(theta, pat_proba[feature_idx], label=f'Closest {j+1}', color='lightsalmon', alpha=0.5)
                #ax.fill(theta, pat_proba[feature_idx], alpha=0.1, color='lightsalmon')

        # Plot the current patient last, so that it can be seen better
        np.fill_diagonal(avg_matrix, delta_patient.values)
        pat_proba = 1 / (1 + np.exp(-avg_matrix.sum(axis=1)))
        _ = ax.plot(theta, pat_proba[feature_idx], label='Selected', color='r')
        ax.fill(theta, pat_proba[feature_idx], alpha=0.1, color='r')

        ax.set_varlabels([label.replace("_", " ") for label in feature_names])
        ax.legend(loc='center left', ncol=1, bbox_to_anchor=(-0.55, 0.5), frameon=True)

    # Curves plot: show only the ones that do matter!!
    # Revert again the order to have the right plot order
    feature_idx_inv = feature_idx[::-1]
    cols_vars = [delta_test.columns[i] for i in feature_idx_inv]

    n_feats = min(len(cols_vars), max_plot_curves)  # Number of features to show in the curves plot
    if n_feats > 0: # There is something to show
        dtrain_df = pd.DataFrame(delta_train_values, columns=feature_names+['const'])
        fig_curve, axs = plt.subplots(n_feats + 1, 1, figsize=(6, 2 * (n_feats + 1)))
        train_logits = delta_train_values.sum(axis=1)
        patient_logit = float(delta_patient.values.sum())
        x_min = min(float(train_logits.min()), patient_logit)
        x_max = max(float(train_logits.max()), patient_logit)
        padding = max((x_max - x_min) * 0.08, 0.1)
        x_vals = np.linspace(x_min - padding, x_max + padding, 300)
        theor_proba = 1 / (1 + np.exp(-x_vals))
        axs[0].plot(x_vals, theor_proba, 'b', alpha=0.2)
        axs[0].scatter(patient_logit, pred_proba_patient, color='r')
        axs[0].set_xlabel('Logit')
        axs[0].set_ylabel('Probability')
        axs[0].set_title(f'Patient results')

        for idj, feat_name in enumerate(cols_vars):
            if idj < n_feats: # Only plot the first n_feats features
                j = idj + 1  # The first plot is already used for the theoretical curve
                curve_df = pd.DataFrame(
                    {
                        "x": x_train[feat_name].values,
                        "y": dtrain_df[feat_name].values,
                    }
                ).dropna().sort_values("x")
                curve_df = curve_df.drop_duplicates("x", keep="first")
                if len(curve_df) > 1:
                    axs[j].plot(curve_df["x"].values, curve_df["y"].values, color='b')
                axs[j].scatter(curve_df["x"].values, curve_df["y"].values, color='b', alpha=0.1)
                for jj in range(n_dists):
                    axs[j].scatter(x_train.iloc[idx_closest[jj]][feat_name], dtrain_df.iloc[idx_closest[jj]][feat_name],
                                        color='lightsalmon', alpha=1)

                patient_x = float(patient_info[feat_name].iloc[0])
                patient_y = float(delta_patient[feat_name].iloc[0])
                axs[j].scatter(patient_x, patient_y, color='r')
                x_axis_values = curve_df["x"].tolist() + [patient_x]
                y_axis_values = curve_df["y"].tolist() + [patient_y]
                if x_axis_values:
                    x_axis_min = min(x_axis_values)
                    x_axis_max = max(x_axis_values)
                    x_padding = max((x_axis_max - x_axis_min) * 0.08, 0.05)
                    axs[j].set_xlim(x_axis_min - x_padding, x_axis_max + x_padding)
                if y_axis_values:
                    y_axis_min = min(y_axis_values)
                    y_axis_max = max(y_axis_values)
                    y_padding = max((y_axis_max - y_axis_min) * 0.08, 0.05)
                    axs[j].set_ylim(y_axis_min - y_padding, y_axis_max + y_padding)
                axs[j].set_ylabel(f"logit {feat_name.replace('_', ' ')}")

        return fig_radar, fig_curve


def analyze_patient_new(
        patient_id,
        df,
        delta_train,
        delta_test,
        x_train,
        y_train,
        features_to_plot=None,
        n_dists=3, # Reduced default for clarity
        max_plot_curves=10,
        show_closest_radial=True,
        show_average_radial=True,
        show_average_class0_radial=True,
        show_average_class1_radial=True
    ):

    logger.debug("Analyzing patient %s with enhanced visualization.", patient_id)

    # --- 1. DATA PREPARATION ---
    
    # Locate patient
    patient_ids = df['ID'].astype(int).tolist()
    if int(patient_id) not in patient_ids:
        logger.debug("Patient %s not found.", patient_id)
        return None, None
        
    patient_index = df[df['ID'] == int(patient_id)].index[0]
    
    # Filter Features
    if features_to_plot and len(features_to_plot) > 0:
        clean_feats = [feat.replace(' ', '_') for feat in features_to_plot]
        feature_names = [x for x in clean_feats if x in delta_test.columns and x not in ["const", "pred_prob"]]
    else:
        feature_names = [x for x in delta_test.columns if x not in ["const", "pred_prob"]]
        
    # Extract data subsets
    d_train = delta_train[feature_names].values
    d_test_patient = delta_test.loc[patient_index, feature_names].values.flatten()
    patient_prob = delta_test.iloc[patient_index]['pred_prob']
    
    # Probability
    pred_prob = delta_test.loc[patient_index, "pred_prob"]
    
    # Neighbor finding (Euclidean distance in SHAP/Delta space)
    dists = np.linalg.norm(d_train - d_test_patient, axis=1)
    idx_closest = np.argsort(dists)[:n_dists]

    # --- 2. RADAR PLOT ---
    
    # Setup Data for Radar
    # We use MinMax Scaling on the DELTA (contribution) values.
    # Min = Lowest contribution observed in training (Low Risk)
    # Max = Highest contribution observed in training (High Risk)
    
    mins = d_train.min(axis=0)
    maxs = d_train.max(axis=0)
    ranges = maxs - mins
    ranges[ranges == 0] = 1e-9 # Avoid division by zero
    
    def normalize(v):
        return (v - mins) / ranges

    # Prepare vectors to plot
    pat_norm = normalize(d_test_patient)
    
    # Averages
    class0_mask = (y_train == 0).values
    class1_mask = (y_train == 1).values
    
    avg_norm = normalize(d_train.mean(axis=0))
    avg_c0_norm = normalize(d_train[class0_mask].mean(axis=0))
    avg_c1_norm = normalize(d_train[class1_mask].mean(axis=0))
    
    # Plotting
    N = len(feature_names)
    theta = radar_factory(N, frame='polygon')
    
    fig_radar, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(projection='radar'))
    
    # Grid lines and labels
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8], labels=[], angle=0, color="grey", alpha=0.3)
    ax.set_varlabels([f.replace("_", " ") for f in feature_names])
    ax.tick_params(pad=15) # Move labels out slightly
    
    # 1. Plot Reference Populations
    if show_average_class0_radial:
        ax.plot(theta, avg_c0_norm, color='#2ca02c', linewidth=2, linestyle='--', label='Avg. negative')
        
    if show_average_class1_radial:
        ax.plot(theta, avg_c1_norm, color='#d62728', linewidth=2, linestyle='--', label='Avg. positive')
        
    if show_average_radial:
        ax.plot(theta, avg_norm, color='grey', linewidth=2, label='Population Average')

    # 2. Plot Closest Neighbors (lighter opacity)
    if show_closest_radial:
        for i, idx in enumerate(idx_closest):
            neighbor_vals = normalize(d_train[idx])
            ax.plot(theta, neighbor_vals, color='#ff7f0e', alpha=0.3, label='Similar Patients' if i == 0 else "")

    # 3. Plot Selected Patient (Thick, Filled)
    ax.plot(theta, pat_norm, color='#1f77b4', linewidth=2, label='Selected Patient')
    ax.fill(theta, pat_norm, color='#1f77b4', alpha=0.1)

    # Styling
    title = f"Patient {patient_id} (prob = {patient_prob:.2f})"
    ax.set_title(title, position=(0.5, 1.1), ha='center', weight='bold')
    
    # Improved Legend
    legend = ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize='small', frameon=False)
    
    
    # --- 3. CURVES PLOT ---
    
    # Select top features (sorted by absolute contribution for this patient)
    # This ensures we see the most relevant features first
    abs_contribution = np.abs(d_test_patient)
    top_indices = np.argsort(abs_contribution)[::-1][:max_plot_curves]
    top_features = [feature_names[i] for i in top_indices]
    
    n_curves = len(top_features) + 1
    
    # Use constrained_layout for automatic nice spacing
    fig_curve, axs = plt.subplots(n_curves, 1, figsize=(8, 3 * n_curves), constrained_layout=True)
    
    if n_curves == 1: axs = [axs] # Handle single plot case

    # -- A. Overall Probability Gauge (Top Plot) --
    ax_prob = axs[0]
    
    # Create a theoretical sigmoid background
    x_sigmoid = np.linspace(-6, 6, 100)
    y_sigmoid = 1 / (1 + np.exp(-x_sigmoid))
    
    # Plot population density of predictions
    all_probs = delta_test['pred_prob'].values
    ax_prob.hist(delta_train['pred_prob'], bins=30, density=False, alpha=0.15, color='grey', label='Population Dist.')
    
    # Plot patient marker
    ax_prob_twin = ax_prob.twinx() # Use twin axis for probability curve vs histogram count
    ax_prob_twin.plot([], []) # Dummy to align colors
    ax_prob_twin.set_ylim(0, 1.1)
    ax_prob_twin.set_yticks([0, 0.5, 1])
    ax_prob_twin.set_ylabel("Probability")
    
    # Current patient line
    ax_prob.axvline(pred_prob, color='#1f77b4', linewidth=3, linestyle='-', label=f'Patient: {pred_prob:.2f}')
    
    ax_prob.set_title("Overall Risk Prediction", weight='bold')
    ax_prob.set_xlabel("Predicted Probability")
    ax_prob.set_yticks([]) # Hide histogram counts
    ax_prob.legend(loc='upper left')


    # -- B. Feature Contribution Curves --
    
    for i, feat in enumerate(top_features):
        ax = axs[i+1]
        
        # Get data
        raw_vals = x_train[feat].values
        shap_vals = delta_train[feat].values
        
        # 1. Background Density (Histogram)
        # Shows where most patients lie for this feature
        ax_hist = ax.twinx()
        ax_hist.hist(raw_vals, bins=20, color='grey', alpha=0.1, density=True)
        ax_hist.set_yticks([]) # Hide density labels
        
        # 2. Relationship Curve (Smoothed or scatter)
        # We sort to draw a line
        sort_idx = np.argsort(raw_vals)
        ax.plot(raw_vals[sort_idx], shap_vals[sort_idx], color='black', alpha=0.4, linewidth=1, label='Risk Trend')
        
        # 3. Reference Points (Class Averages)
        c0_mean_x = x_train.loc[y_train==0, feat].mean()
        c0_mean_y = delta_train.loc[y_train==0, feat].mean()
        c1_mean_x = x_train.loc[y_train==1, feat].mean()
        c1_mean_y = delta_train.loc[y_train==1, feat].mean()
        
        ax.scatter(c0_mean_x, c0_mean_y, color='#2ca02c', s=100, marker='D', label='Avg Low Risk', zorder=3)
        ax.scatter(c1_mean_x, c1_mean_y, color='#d62728', s=100, marker='D', label='Avg High Risk', zorder=3)
        
        # 4. Patient & Neighbors
        pat_x = df.loc[patient_index, feat]
        pat_y = delta_test.loc[patient_index, feat]
        
        # Neighbors
        for n_idx in idx_closest:
            n_x = x_train.iloc[n_idx][feat]
            n_y = delta_train.iloc[n_idx][feat]
            ax.scatter(n_x, n_y, color='#ff7f0e', alpha=0.6, s=30)
            
        # Patient (Large Dot)
        ax.scatter(pat_x, pat_y, color='#1f77b4', s=150, edgecolors='white', linewidth=2, label='Patient', zorder=5)
        
        # Labels
        ax.set_title(f"Feature: {feat.replace('_', ' ')}")
        ax.set_xlabel(f"Value ({feat})")
        ax.set_ylabel("Risk Contribution")
        ax.grid(True, alpha=0.2)
        
        if i == 0: # Only legend on first feature plot to save space
            ax.legend(loc='best', fontsize='small')

    return fig_radar, fig_curve

def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` Axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding Axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                # spine_type must be 'left'/'right'/'top'/'bottom'/'circle'.
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                # unit_regular_polygon gives a polygon of radius 1 centered at
                # (0, 0) but we want a polygon of radius 0.5 centered at (0.5,
                # 0.5) in axes coordinates.
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta
