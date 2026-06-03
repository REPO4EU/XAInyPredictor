from datetime import datetime
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

# Local imports
from XAInyPredictor.modules.xai import analyze_patient, threshold_for_target_fnr


@module.ui
def prediction_ui(config=None):
    if config is None:
        config = {}

    titles = config.get("titles", {})
    labels = config.get("labels", {})
    help_texts = config.get("help_texts", {})

    pos_class_label = labels.get("positive_class_label", "Positive")
    neg_class_label = labels.get("negative_class_label", "Negative")
    prob_col = labels.get("probability_column", "Stratification Score")
    class_col = labels.get("class_column", "Patient Group")

    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("patient_selector_ui"),
            ui.div(
                ui.input_numeric(
                    id="fnr_threshold",
                    label= ui.div(
                        "Allowed false-negative rate: ",
                        ui.popover(
                            ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                            ui.tags.div(
                                ui.tags.b("Understanding this threshold:"),
                                ui.tags.p(help_texts.get("fnr_threshold", "This controls the safety margin of the model.")),
                                ui.tags.ul(
                                    ui.tags.li(ui.tags.b("0% False Negative Ratio:"), " " + help_texts.get("fnr_zero", "We refuse to miss any true positive patients.")),
                                    ui.tags.li(ui.tags.b("Higher False Negative Ratio:"), " " + help_texts.get("fnr_higher", "We accept missing some positive patients for higher specificity.")),
                                ),
                                style="width: 250px;"
                            ),
                            placement="right"
                        )
                    ),
                    value=0, min=0, max=100, step=1
                ),
            ),
            ui.hr(),
            ui.h5("Visualization Settings"),
            ui.input_select(
                id="view_mode",
                label="Select View:",
                choices={
                    "radar": titles.get("feature_analysis_short", "Profile Comparison"),
                    "curve": titles.get("distance_analysis_short", "Feature Curves")
                },
                selected="radar",
            ),
            ui.output_ui("features_to_plot_ui"),
            ui.panel_conditional(
                "input.view_mode == 'radar'",
                ui.input_checkbox_group(
                    id="radar_plot_elements",
                    label="Radar Plot Elements:",
                    choices={
                        "closest": "Closest patients",
                        "average": "Average all patients",
                        "average_0": f"Avg. {neg_class_label}",
                        "average_1": f"Avg. {pos_class_label}",
                    },
                    selected=["closest", "average", "average_0", "average_1"],
                ),
            ),
        ),
        ui.page_fluid(
            ui.div(
                ui.div("Stratification Analysis", class_="stratification-tabset-title"),
                ui.navset_tab(
                ui.nav_panel(
                    "Patient",
                    ui.div(
                        ui.download_button("download_report_package", "Download report package", class_="btn-primary btn-sm report-package-download"),
                        ui.download_button("download_stratification_results", "Download stratification results", class_="btn-default btn-sm"),
                        ui.download_button("download_closest_patients", "Download closest patients", class_="btn-default btn-sm"),
                        ui.download_button("download_cohort_summary", "Download cohort summary", class_="btn-default btn-sm"),
                        class_="stratification-downloads",
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Selected Patient Stratification"),
                            ui.output_ui("stratification_summary"),
                            ui.output_ui("stratification_interpretation"),
                            class_="selected-patient-stratification-card",
                        ),
                        col_widths=12,
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    titles.get("prediction_results", "Prediction results "),
                                    ui.popover(
                                        ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                                        ui.tags.div(
                                            ui.tags.b("Stratification results:"),
                                            ui.tags.p("The table shows the model-based patient stratification:"),
                                            ui.tags.ul(
                                                ui.tags.li(ui.tags.b(f"{prob_col}:"), f" Indicates the score used to assign the patient group."),
                                                ui.tags.li(ui.tags.b(f"{class_col}:"), f" Assigns the patient to {pos_class_label} or {neg_class_label}."),
                                            ),
                                            style="width: 250px;"
                                        ),
                                        placement="right"
                                    )
                                )
                            ),
                            ui.output_data_frame("results_table_output"),
                            height="240px",
                        ),
                        col_widths=12,
                    ),
                    ui.output_ui("dynamic_plot_container"),
                    value="patient",
                ),
                ui.nav_panel(
                    "Cohort",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Cohort Stratification Summary"),
                            ui.output_ui("cohort_stratification_summary"),
                            height="180px",
                        ),
                        col_widths=12,
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    "Cohort Score Distribution ",
                                    ui.popover(
                                        ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                                        ui.tags.div(
                                            ui.tags.b("Cohort score distribution:"),
                                            ui.tags.p("Patients are ordered by stratification score. The dashed line marks the current decision threshold and the selected patient is highlighted."),
                                            style="width: 250px;"
                                        ),
                                        placement="right"
                                    )
                                )
                            ),
                            ui.output_plot("cohort_score_distribution_plot", height="420px", width="100%"),
                            full_screen=True,
                        ),
                        col_widths=12,
                    ),
                    value="cohort",
                ),
                ui.nav_panel(
                    "Model",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header("Model Card"),
                            ui.output_ui("model_card"),
                            class_="model-card",
                        ),
                        col_widths=12,
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    "Global Feature Importance ",
                                    ui.popover(
                                        ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                                        ui.tags.div(
                                            ui.tags.b("Global feature importance:"),
                                            ui.tags.p("Importance is estimated from the variance of each feature contribution in the model explanation matrix."),
                                            style="width: 250px;"
                                        ),
                                        placement="right"
                                    )
                                )
                            ),
                            ui.output_plot("global_feature_importance_plot", height="460px", width="100%"),
                            full_screen=True,
                        ),
                        col_widths=12,
                    ),
                    value="model",
                ),
                ui.nav_panel(
                    "Reference Patients",
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(
                                ui.div(
                                    "Closest Reference Patients ",
                                    ui.popover(
                                        ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                                        ui.tags.div(
                                            ui.tags.b("Closest reference patients:"),
                                            ui.tags.p("Reference patients are ranked by distance in the model contribution space, using the same feature effects that drive the explanation plots."),
                                            style="width: 260px;"
                                        ),
                                        placement="right"
                                    )
                                )
                            ),
                            ui.output_ui("closest_reference_patients_narrative"),
                            ui.output_data_frame("closest_reference_patients_table"),
                            height="420px",
                            full_screen=True,
                        ),
                        col_widths=12,
                    ),
                    value="reference",
                ),
                    id="stratification_tabs",
                    selected="patient",
                ),
                class_="stratification-tabset",
            )
        ),
    )


@module.server
def server(input: Inputs, output: Outputs, session: Session, global_input_data, patient_selected_id, model_data, delta_test_reactive, x_test_reactive, prob_threshold, config_init, config_reactive=None):

    @reactive.Calc
    def current_labels():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        labels = current_cfg.get("labels", {})
        return {
            "positive_class": current_cfg.get("positive_class", "YES"),
            "negative_class": current_cfg.get("negative_class", "NO"),
            "positive_class_label": labels.get("positive_class_label", "Positive"),
            "negative_class_label": labels.get("negative_class_label", "Negative"),
            "probability_column": labels.get("probability_column", "Stratification Score"),
            "class_column": labels.get("class_column", "Patient Group")
        }

    @reactive.Calc
    def current_config():
        return config_reactive.get() if config_reactive else (config_init or {})

    @reactive.Calc
    def stratification_results_df():
        df = global_input_data.get()
        delta_test = delta_test_reactive.get()
        prob_thr = prob_threshold.get()
        lbls = current_labels()

        if df is None or delta_test is None or df.empty or delta_test.empty:
            return None
        if "ID" not in df.columns or "pred_prob" not in delta_test.columns:
            return None

        threshold = float(prob_thr) if prob_thr is not None else 0
        res_df = pd.concat(
            [
                df[["ID"]].reset_index(drop=True),
                delta_test[["pred_prob"]].reset_index(drop=True),
            ],
            axis=1,
        )
        res_df = res_df.rename(columns={"pred_prob": lbls["probability_column"]})
        res_df[lbls["class_column"]] = np.where(
            res_df[lbls["probability_column"]] >= threshold,
            lbls["positive_class_label"],
            lbls["negative_class_label"],
        )
        return res_df.sort_values(by=["ID"]).reset_index(drop=True)

    def _feature_display_name(feature_name: str) -> str:
        return str(feature_name).replace("_", " ")

    def _contribution_columns(delta_df):
        if delta_df is None:
            return []
        return [col for col in delta_df.columns if col not in ["const", "pred_prob"]]

    @reactive.Calc
    def closest_reference_patients_df():
        raw_df = global_input_data.get()
        delta_test = delta_test_reactive.get()
        md = model_data.get()
        sel_id = patient_selected_id.get()
        lbls = current_labels()
        threshold = float(prob_threshold.get()) if prob_threshold.get() is not None else 0

        if (
            raw_df is None
            or delta_test is None
            or md is None
            or sel_id is None
            or raw_df.empty
            or delta_test.empty
        ):
            return None

        delta_train = md.get("D_TRAIN")
        x_train_raw = md.get("X_TRAIN_RAW")
        if delta_train is None or x_train_raw is None or delta_train.empty or x_train_raw.empty:
            return None

        patient_rows = raw_df.index[raw_df["ID"] == int(sel_id)].tolist()
        if not patient_rows:
            return None
        patient_pos = patient_rows[0]

        feature_cols = [
            col
            for col in _contribution_columns(delta_train)
            if col in delta_test.columns
        ]
        if not feature_cols or patient_pos >= len(delta_test):
            return None

        train_matrix = delta_train[feature_cols].to_numpy(dtype=float)
        patient_vector = delta_test.iloc[patient_pos][feature_cols].to_numpy(dtype=float)
        distances = np.linalg.norm(train_matrix - patient_vector, axis=1)
        closest_positions = np.argsort(distances)[:5]

        feature_candidates = list(input.features_to_plot() or [])
        if not feature_candidates:
            feature_candidates = [_feature_display_name(col) for col in feature_cols[:4]]
        feature_candidates = feature_candidates[:4]

        rows = []
        x_train_display = x_train_raw.copy()
        x_train_display.columns = [col.replace("_", " ") for col in x_train_display.columns]

        for pos in closest_positions:
            train_row = x_train_display.iloc[int(pos)]
            score = float(delta_train.iloc[int(pos)]["pred_prob"])
            group = lbls["positive_class_label"] if score >= threshold else lbls["negative_class_label"]
            row = {
                "Reference ID": int(train_row["ID"]) if "ID" in train_row else int(pos) + 1,
                "Distance": round(float(distances[int(pos)]), 3),
                lbls["probability_column"]: round(score, 3),
                lbls["class_column"]: group,
            }
            for feature in feature_candidates:
                display_feature = _feature_display_name(feature)
                if display_feature in train_row.index:
                    row[display_feature] = train_row[display_feature]
            rows.append(row)

        return pd.DataFrame(rows)

    @reactive.Calc
    def cohort_summary_df():
        res_df = stratification_results_df()
        lbls = current_labels()

        if res_df is None or res_df.empty:
            return None

        prob_col = lbls["probability_column"]
        class_col = lbls["class_column"]
        total = len(res_df)
        rows = [
            {"Metric": "Patients", "Value": total},
            {"Metric": "Mean score", "Value": round(float(res_df[prob_col].mean()), 3)},
            {"Metric": "Decision threshold", "Value": round(float(prob_threshold.get() or 0), 3)},
        ]
        for group, count in res_df[class_col].value_counts().items():
            rows.append({"Metric": f"{group} count", "Value": int(count)})
            rows.append({"Metric": f"{group} percentage", "Value": round(float(count) / total * 100, 1)})
        return pd.DataFrame(rows)

    def _export_metadata():
        cfg = current_config()
        lbls = current_labels()
        input_df = global_input_data.get()
        patient_count = 0 if input_df is None else len(input_df)
        threshold = float(prob_threshold.get()) if prob_threshold.get() is not None else 0

        return {
            "Use Case": cfg.get("name", "Unknown use case"),
            "Model Objective": cfg.get("description", "Patient stratification model"),
            "Target": cfg.get("target_column", "target"),
            "Decision Threshold": round(threshold, 3),
            "Negative Group": lbls["negative_class_label"],
            "Positive Group": lbls["positive_class_label"],
            "Patient Count": patient_count,
            "Exported At": datetime.now().isoformat(timespec="seconds"),
            "Output Type": "Research prototype; clinical workflow utility should be validated with collaborators",
        }

    def _attach_export_metadata(df):
        metadata = _export_metadata()
        export_df = df.copy() if df is not None else pd.DataFrame()
        for key, value in reversed(list(metadata.items())):
            export_df.insert(0, key, value)
        return export_df

    def _cohort_summary_export_df():
        summary_df = cohort_summary_df()
        if summary_df is None:
            summary_df = pd.DataFrame()
        metadata_df = pd.DataFrame(
            [{"Metric": key, "Value": value} for key, value in _export_metadata().items()]
        )
        if not summary_df.empty:
            return pd.concat([metadata_df, pd.DataFrame([{"Metric": "", "Value": ""}]), summary_df], ignore_index=True)
        return metadata_df

    def _metadata_export_df():
        return pd.DataFrame(
            [{"Field": key, "Value": value} for key, value in _export_metadata().items()]
        )

    def _report_readme_text():
        metadata = _export_metadata()
        return (
            "XAInyPredictor report package\n"
            "============================\n\n"
            f"Use case: {metadata.get('Use Case', 'Unknown use case')}\n"
            f"Exported at: {metadata.get('Exported At', '')}\n\n"
            "Files included:\n"
            "- metadata.csv: use case, threshold, patient groups, export timestamp, and prototype context.\n"
            "- stratification_results.csv: patient-level stratification score and assigned patient group.\n"
            "- cohort_stratification_summary.csv: cohort-level counts, score summary, and metadata.\n"
            "- closest_reference_patients.csv: closest reference patients for the selected patient when available.\n\n"
            "Prototype context:\n"
            "This package supports patient stratification research. Clinical workflow utility and interpretation should be validated with clinical collaborators.\n"
        )

    def _run_patient_analysis(data, feats, opts=None):
        if not data:
            return None, None

        opts = opts or []
        lbls = current_labels()
        try:
            result = analyze_patient(
                patient_id=data["patient_id"],
                df=data["df"],
                delta_train=data["delta_train"],
                delta_test=data["delta_test"],
                x_train=data["x_train"],
                y_train=data["y_train"],
                x_test=data["x_test"],
                features_to_plot=feats,
                n_dists=3,
                show_closest_radial="closest" in opts,
                show_average_radial="average" in opts,
                show_average_class0_radial="average_0" in opts,
                show_average_class1_radial="average_1" in opts,
                neg_class_label=lbls["negative_class_label"],
                pos_class_label=lbls["positive_class_label"],
            )
        except Exception:
            return None, None

        if not result:
            return None, None

        return result

    @reactive.Effect
    def _update_sidebar_labels():
        lbls = current_labels()
        neg = lbls["negative_class_label"]
        pos = lbls["positive_class_label"]
        
        ui.update_checkbox_group(
            "radar_plot_elements",
            choices={
                "closest": "Closest patients",
                "average": "Average all patients",
                "average_0": f"Avg. {neg}",
                "average_1": f"Avg. {pos}",
            },
            selected=list(input.radar_plot_elements()) if input.radar_plot_elements() else ["closest", "average", "average_0", "average_1"]
        )

    @reactive.Effect
    def _sync_selection_with_active_data():
        df = global_input_data.get()
        if df is None or df.empty or "ID" not in df.columns:
            patient_selected_id.set(None)
            return

        all_ids = sorted(df["ID"].astype(int).tolist())
        current_selection = patient_selected_id.get()
        if current_selection is None or int(current_selection) not in all_ids:
            patient_selected_id.set(all_ids[0])
            ui.update_selectize("local_patient_select", selected=all_ids[0])

    @output
    @render.ui
    def patient_selector_ui():
        """
        Dynamic UI for Patient Selection. The function listens to changes in
        global_input_data variable and according to this, it creates a dropdown
        to select patients.
        """
        df = global_input_data.get()
        if df is None or df.empty or 'ID' not in df.columns:
            return ui.p("Add or upload patients in Data Input to generate stratification outputs.")
        
        # Get list of patient IDs
        all_ids = sorted(df['ID'].astype(int).tolist())

        # Check current selection from global state
        current_selection = patient_selected_id.get()

        # Default to first if no current selection
        if current_selection is None or int(current_selection) not in all_ids:
            selected_val = all_ids[0]
        else:
            selected_val = int(current_selection)

        return ui.input_selectize(
            id="local_patient_select",
            label="Select patient:",
            choices=all_ids,
            selected=selected_val,
            options={
                "create": False,  # Don't allow creating new options
                "allowEmptyOption": False,
                "placeholder": "Search by patient ID...",
                "maxItems": 1
            }
        )

    @output
    @render.ui
    def features_to_plot_ui():
        """
        Dynamic UI to define the features to plot. The UI gets the default features
        to plot from the argument features_to_plot, which obtains them from the
        features that appear in the formula (discarding the ones that do not appear).
        """
        md = model_data.get()
        feature_names = md.get("FEATURE_ORDER_DISPLAY", []) if md else []
        features_to_plot = md.get("FEATS_IN_FORMULA", []) if md else []
        return ui.input_selectize(
            id="features_to_plot",
            label="Select features to view:",
            choices=feature_names,
            selected=features_to_plot,
            multiple=True,
        )

    @output
    @render.ui
    def dynamic_plot_container():
        mode = input.view_mode()
        selected_features = input.features_to_plot()

        lbls = current_labels()
        neg = lbls["negative_class_label"]
        pos = lbls["positive_class_label"]

        # Calculate dynamic height for Curve plot
        # Base height + (pixels per feature * number of features)
        n_feats = len(selected_features) if selected_features else 0
        curve_height_px = 300 + (n_feats * 250) 
        
        # Define UI cards
        radar_card = ui.card(
            ui.card_header(
                ui.div(
                    "Patient Profile Comparison ",
                    ui.popover(
                        ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                        ui.tags.div(
                            ui.tags.b("Understanding the radar plot:"),
                            ui.tags.p("The radar plot compares the values of the features from a selected patient (red) with three different distributions:"),
                            ui.tags.ul(
                                ui.tags.li(ui.tags.b("Average all patients:"), " The average values from all patients in the model (blue)."),
                                ui.tags.li(ui.tags.b(f"Avg. {neg}:"), f" The average values from all {neg} patients in the model (green)."),
                                ui.tags.li(ui.tags.b(f"Avg. {pos}:"), f" The average values from all {pos} patients in the model (yellow)."),
                            ),
                            ui.tags.p("Comparing these values helps contextualize why the selected patient falls into a given stratification group."),
                            style="width: 250px;"
                        ),
                        placement="right"
                    )
                )
            ),
            # Height is fixed or auto, but width adapts
            ui.output_plot("radar_plot", height="650px", width="100%"),
            full_screen=True
        )
        
        curve_card = ui.card(
            ui.card_header(
                ui.div(
                    "Feature Context Curves ",
                    ui.popover(
                        ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                        ui.tags.div(
                            ui.tags.b("Understanding the curves plot:"),
                            ui.tags.p("The curves plot displays the distribution of values from a specific feature across the patients in the model, ordered from lowest to highest (blue dots). It highlights in red the patient selected, put in context the feature values of the patient in comparison with the values from the rest of patients."),
                            style="width: 250px;"
                        ),
                        placement="right"
                    )
                )
            ),
            # DYNAMIC HEIGHT APPLIED HERE
            ui.output_plot("curve_plot", height=f"{curve_height_px}px", width="100%"),
            full_screen=True
        )

        if mode == "radar":
            return radar_card
        elif mode == "curve":
            return curve_card
        else:
            return ui.p("Incorrect view selection!")

    @reactive.Effect
    @reactive.event(input.local_patient_select)
    def _sync_selection():
        """
        Synchronize selected patient in data_exploration.py with the global variable.
        """
        # Update the global reactive value (as integer if possible)
        val = input.local_patient_select()
        
        # Handle empty string case - don't update if no valid selection
        if not val or val == '' or val == 'None':
            return
        
        try:
            new_value = int(val)
            patient_selected_id.set(new_value)
        except ValueError:
            # If conversion fails, don't update and optionally show notification
            # ui.notification_show("Invalid patient ID selected", type="error")
            return

    @reactive.Effect
    @reactive.event(input.fnr_threshold)
    def _calculate_probability_threshold():
        """
        Calculate probability threshold when the FNR threshold changes.
        """
        fnr_val = input.fnr_threshold()
        delta_test = delta_test_reactive.get()
        md = model_data.get()
        y_test = md.get("Y_TEST") if md else None

        if y_test is None or delta_test is None:
            return

        try:
            target_fnr = float(fnr_val) / 100
        except ValueError:
            return

        # Encode y_test to 1/0
        lbls = current_labels()
        pos_class = lbls["positive_class"]
        y_test_encoded = (y_test == pos_class).astype(int)

        # Calculate probability threshold
        y_test_encoded.sort_index(inplace=True)
        delta_test.sort_index(inplace=True)
        threshold, fnr = threshold_for_target_fnr(
            y_test_encoded.to_numpy(),
            delta_test['pred_prob'].to_numpy(),
            target_fnr=target_fnr
        )

        prob_threshold.set(threshold)

    # --- Calculation ---

    @reactive.Calc
    def get_patient_data_context():
        """
        Gathers all the dataframes and IDs needed for analysis.
        Returns a dictionary of data or None if invalid.
        """
        raw_df = global_input_data.get()
        delta_test = delta_test_reactive.get()
        x_test = x_test_reactive.get()
        patient_id = patient_selected_id.get()

        md = model_data.get()
        if md is None:
            return None

        delta_train = md.get("D_TRAIN")
        x_train = md.get("X_TRAIN")
        y_train = md.get("Y_TRAIN")

        # Validation checks
        if any((x is None) or (isinstance(x, pd.DataFrame) and x.empty) for x in [raw_df, delta_train, delta_test, x_train, y_train, patient_id]):
            return None

        all_ids = sorted(raw_df['ID'].astype(int).tolist())
        if patient_id == None or int(patient_id) not in all_ids:
            return None

        # Standardize columns
        df = raw_df.copy()
        df.columns = [col.replace(' ', '_') for col in df.columns]

        # Encode y_train to 1/0
        lbls = current_labels()
        pos_class = lbls["positive_class"]
        y_train_encoded = (y_train == pos_class).astype(int)

        return {
            "patient_id": patient_id,
            "df": df,
            "delta_train": delta_train,
            "delta_test": delta_test,
            "x_train": x_train,
            "y_train": y_train_encoded,
            "x_test": x_test,
        }

    # --- Outputs ---

    @output
    @render.ui
    def stratification_summary():
        sel_id = patient_selected_id.get()
        prob_thr = prob_threshold.get()
        lbls = current_labels()
        res_df = stratification_results_df()

        prob_col = lbls["probability_column"]
        class_col = lbls["class_column"]

        if res_df is None or res_df.empty or sel_id is None:
            return ui.div("Add or upload patients to review individual stratification outputs.", class_="stratification-summary-empty")

        patient_row = res_df[res_df["ID"] == sel_id]
        if patient_row.empty:
            return ui.div("Selected patient not found.", class_="stratification-summary-empty")

        score = float(patient_row[prob_col].iloc[0])
        threshold = float(prob_thr) if prob_thr is not None else 0
        group = str(patient_row[class_col].iloc[0])
        group_class = "stratification-positive" if group == lbls["positive_class_label"] else "stratification-negative"

        return ui.div(
            ui.div(
                ui.div("Patient", class_="stratification-summary-label"),
                ui.div(f"{int(sel_id)}", class_="stratification-summary-value"),
                class_="stratification-summary-item",
            ),
            ui.div(
                ui.div(prob_col, class_="stratification-summary-label"),
                ui.div(f"{score:.3f}", class_="stratification-summary-value"),
                class_="stratification-summary-item",
            ),
            ui.div(
                ui.div("Decision Threshold", class_="stratification-summary-label"),
                ui.div(f"{threshold:.3f}", class_="stratification-summary-value"),
                class_="stratification-summary-item",
            ),
            ui.div(
                ui.div(class_col, class_="stratification-summary-label"),
                ui.div(group, class_=f"stratification-summary-value {group_class}"),
                class_="stratification-summary-item",
            ),
            class_="stratification-summary",
        )

    @output
    @render.ui
    def stratification_interpretation():
        sel_id = patient_selected_id.get()
        prob_thr = prob_threshold.get()
        lbls = current_labels()
        res_df = stratification_results_df()

        if res_df is None or res_df.empty or sel_id is None:
            return None

        prob_col = lbls["probability_column"]
        class_col = lbls["class_column"]
        patient_row = res_df[res_df["ID"] == sel_id]
        if patient_row.empty:
            return None

        score = float(patient_row[prob_col].iloc[0])
        threshold = float(prob_thr) if prob_thr is not None else 0
        group = str(patient_row[class_col].iloc[0])
        direction = "above or equal to" if score >= threshold else "below"
        positive_label = lbls["positive_class_label"]
        negative_label = lbls["negative_class_label"]

        return ui.div(
            ui.p(
                f"This patient is assigned to {group} because the stratification score ({score:.3f}) is {direction} the decision threshold ({threshold:.3f}).",
                class_="stratification-interpretation-main",
            ),
            ui.div(
                ui.div(
                    ui.tags.b(f"Below threshold: "),
                    f"assigned to {negative_label}.",
                    class_="stratification-rule stratification-rule-negative",
                ),
                ui.div(
                    ui.tags.b(f"At or above threshold: "),
                    f"assigned to {positive_label}.",
                    class_="stratification-rule stratification-rule-positive",
                ),
                class_="stratification-rules",
            ),
            ui.p(
                "Research prototype output: this stratification supports cohort-level clinical decision support research and should be interpreted with clinical collaborators during utility validation.",
                class_="stratification-disclaimer",
            ),
            class_="stratification-interpretation",
        )

    @output
    @render.ui
    def cohort_stratification_summary():
        res_df = stratification_results_df()
        lbls = current_labels()

        if res_df is None or res_df.empty:
            return ui.div("Add or upload patients to summarize cohort-level stratification.", class_="stratification-summary-empty")

        prob_col = lbls["probability_column"]
        class_col = lbls["class_column"]
        group_counts = res_df[class_col].value_counts()
        total = len(res_df)
        positive_label = lbls["positive_class_label"]
        negative_label = lbls["negative_class_label"]

        positive_count = int(group_counts.get(positive_label, 0))
        negative_count = int(group_counts.get(negative_label, 0))
        mean_score = float(res_df[prob_col].mean())
        positive_pct = (positive_count / total * 100) if total else 0

        return ui.div(
            ui.div(
                ui.div("Patients", class_="stratification-summary-label"),
                ui.div(str(total), class_="stratification-summary-value"),
                class_="stratification-summary-item",
            ),
            ui.div(
                ui.div("Mean score", class_="stratification-summary-label"),
                ui.div(f"{mean_score:.3f}", class_="stratification-summary-value"),
                class_="stratification-summary-item",
            ),
            ui.div(
                ui.div(negative_label, class_="stratification-summary-label"),
                ui.div(str(negative_count), class_="stratification-summary-value stratification-negative"),
                class_="stratification-summary-item",
            ),
            ui.div(
                ui.div(positive_label, class_="stratification-summary-label"),
                ui.div(f"{positive_count} ({positive_pct:.0f}%)", class_="stratification-summary-value stratification-positive"),
                class_="stratification-summary-item",
            ),
            class_="stratification-summary",
        )

    @output
    @render.ui
    def model_card():
        cfg = current_config()
        md = model_data.get()
        lbls = current_labels()
        features = cfg.get("features", [])
        metadata = cfg.get("model_metadata", {})
        threshold = float(prob_threshold.get()) if prob_threshold.get() is not None else 0
        reference_n = 0
        if md is not None and md.get("X_TRAIN_RAW") is not None:
            reference_n = len(md.get("X_TRAIN_RAW"))

        feature_names = [
            feature.get("display_name", feature.get("name", "Feature"))
            for feature in features
        ]
        feature_preview = ", ".join(feature_names[:6])
        if len(feature_names) > 6:
            feature_preview += f", and {len(feature_names) - 6} more"

        return ui.div(
            ui.div(
                ui.div(
                    ui.div("Use case", class_="model-card-label"),
                    ui.div(cfg.get("name", "Patient stratification model"), class_="model-card-value"),
                    class_="model-card-item",
                ),
                ui.div(
                    ui.div("Target", class_="model-card-label"),
                    ui.div(cfg.get("target_column", "Target"), class_="model-card-value"),
                    class_="model-card-item",
                ),
                ui.div(
                    ui.div("Decision threshold", class_="model-card-label"),
                    ui.div(f"{threshold:.3f}", class_="model-card-value"),
                    class_="model-card-item",
                ),
                ui.div(
                    ui.div("Reference cohort", class_="model-card-label"),
                    ui.div(str(reference_n), class_="model-card-value"),
                    class_="model-card-item",
                ),
                ui.div(
                    ui.div("Input variables", class_="model-card-label"),
                    ui.div(str(len(features)), class_="model-card-value"),
                    class_="model-card-item",
                ),
                ui.div(
                    ui.div("Patient groups", class_="model-card-label"),
                    ui.div(f"{lbls['negative_class_label']} / {lbls['positive_class_label']}", class_="model-card-value"),
                    class_="model-card-item model-card-groups",
                ),
                ui.div(
                    ui.div("Model version", class_="model-card-label"),
                    ui.div(metadata.get("model_version", "Prototype"), class_="model-card-value"),
                    class_="model-card-item",
                ),
                ui.div(
                    ui.div("Training snapshot", class_="model-card-label"),
                    ui.div(metadata.get("training_date", "Not specified"), class_="model-card-value"),
                    class_="model-card-item",
                ),
                class_="model-card-grid",
            ),
            ui.div(
                ui.div("Model objective", class_="model-card-label"),
                ui.p(cfg.get("description", "Patient stratification for cohort-level clinical decision support research."), class_="model-card-copy"),
                class_="model-card-note",
            ),
            ui.div(
                ui.div("Reference cohort source", class_="model-card-label"),
                ui.p(metadata.get("cohort_source", "Reference cohort source not specified."), class_="model-card-copy"),
                class_="model-card-note",
            ),
            ui.div(
                ui.div("Intended use", class_="model-card-label"),
                ui.p(metadata.get("intended_use", "Patient stratification research and exploratory decision support."), class_="model-card-copy"),
                class_="model-card-note",
            ),
            ui.div(
                ui.div("Variables used", class_="model-card-label"),
                ui.p(feature_preview or "No input variables configured.", class_="model-card-copy"),
                class_="model-card-note",
            ),
            ui.div(
                ui.div("Validation status", class_="model-card-label"),
                ui.p(metadata.get("validation_status", "Clinical utility should be validated with domain experts."), class_="model-card-copy"),
                class_="model-card-note model-card-validation",
            ),
            ui.div(
                ui.div("Limitations", class_="model-card-label"),
                ui.p(metadata.get("limitations", "Research prototype for patient stratification; outputs require expert review and validation."), class_="model-card-copy"),
                class_="model-card-note model-card-warning",
            ),
        )

    @output
    @render.plot
    def cohort_score_distribution_plot():
        res_df = stratification_results_df()
        sel_id = patient_selected_id.get()
        threshold = float(prob_threshold.get()) if prob_threshold.get() is not None else 0
        lbls = current_labels()

        if res_df is None or res_df.empty:
            return _empty_plot("No cohort scores available")

        prob_col = lbls["probability_column"]
        class_col = lbls["class_column"]
        sorted_df = res_df.sort_values(prob_col, ascending=False).reset_index(drop=True)
        colors = np.where(
            sorted_df[class_col] == lbls["positive_class_label"],
            "#dc3545",
            "#198754",
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(np.arange(len(sorted_df)), sorted_df[prob_col], color=colors, alpha=0.82, width=0.85)
        ax.axhline(threshold, color="#1f2d3d", linestyle="--", linewidth=1.5, label=f"Threshold {threshold:.3f}")

        if sel_id is not None and sel_id in sorted_df["ID"].values:
            selected_idx = int(sorted_df.index[sorted_df["ID"] == sel_id][0])
            selected_score = float(sorted_df.loc[selected_idx, prob_col])
            ax.scatter(selected_idx, selected_score, color="#ffc107", edgecolor="#1f2d3d", s=110, zorder=5, label=f"Patient {int(sel_id)}")

        ax.set_title("Cohort stratification scores", fontweight="bold")
        ax.set_xlabel("Patients ordered by score")
        ax.set_ylabel(prob_col)
        ax.set_ylim(0, max(1.0, float(sorted_df[prob_col].max()) * 1.08))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    @output
    @render.plot
    def global_feature_importance_plot():
        md = model_data.get()
        delta_test = delta_test_reactive.get()

        if md is None:
            return _empty_plot("No model data available")

        delta_train = md.get("D_TRAIN")
        if delta_train is None or delta_train.empty:
            return _empty_plot("No reference explanation matrix available")

        train_cols = _contribution_columns(delta_train)
        if not train_cols:
            return _empty_plot("No feature contributions available")

        train_importance = delta_train[train_cols].var(axis=0).sort_values(ascending=False)
        top_features = train_importance.head(10).index.tolist()

        fig, ax = plt.subplots(figsize=(9, 4.8))
        y_pos = np.arange(len(top_features))
        ax.barh(
            y_pos,
            train_importance.loc[top_features].to_numpy(),
            color="#007bff",
            alpha=0.78,
            label="Reference",
        )

        if delta_test is not None and not delta_test.empty:
            current_cols = [col for col in top_features if col in delta_test.columns]
            if current_cols:
                current_importance = delta_test[current_cols].var(axis=0).reindex(top_features).fillna(0)
                ax.scatter(
                    current_importance.to_numpy(),
                    y_pos,
                    color="#fd7e14",
                    s=42,
                    label="Current cohort",
                    zorder=4,
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([_feature_display_name(col) for col in top_features])
        ax.invert_yaxis()
        ax.set_xlabel("Importance")
        ax.set_title("Global feature importance", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    @output
    @render.data_frame
    def closest_reference_patients_table():
        closest_df = closest_reference_patients_df()
        if closest_df is None or closest_df.empty:
            return render.DataGrid(
                pd.DataFrame({"Message": ["Reference comparison requires a selected patient and an available reference cohort."]}),
                width="100%",
                selection_mode="none",
            )

        return render.DataGrid(
            closest_df,
            width="100%",
            filters=False,
            selection_mode="none",
        )

    @output
    @render.ui
    def closest_reference_patients_narrative():
        closest_df = closest_reference_patients_df()
        lbls = current_labels()

        if closest_df is None or closest_df.empty:
            return ui.div("Reference comparison requires a selected patient and an available reference cohort.", class_="closest-patients-narrative")

        class_col = lbls["class_column"]
        if class_col not in closest_df.columns:
            return None

        counts = closest_df[class_col].value_counts()
        top_group = str(counts.index[0])
        top_count = int(counts.iloc[0])
        total = len(closest_df)
        score_col = lbls["probability_column"]
        mean_score = float(closest_df[score_col].mean()) if score_col in closest_df.columns else None

        score_text = f" Their mean stratification score is {mean_score:.3f}." if mean_score is not None else ""
        return ui.div(
            f"The selected patient is most similar to reference patients mostly assigned to {top_group} ({top_count}/{total}).{score_text}",
            class_="closest-patients-narrative",
        )

    @render.download(filename="stratification_results.csv")
    def download_stratification_results():
        res_df = stratification_results_df()
        if res_df is None:
            res_df = pd.DataFrame()
        yield _attach_export_metadata(res_df).to_csv(index=False)

    @render.download(filename="xainypredictor_report_package.zip")
    def download_report_package():
        res_df = stratification_results_df()
        closest_df = closest_reference_patients_df()
        if res_df is None:
            res_df = pd.DataFrame()
        if closest_df is None:
            closest_df = pd.DataFrame()

        buffer = BytesIO()
        with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as report_zip:
            report_zip.writestr("README.txt", _report_readme_text())
            report_zip.writestr("metadata.csv", _metadata_export_df().to_csv(index=False))
            report_zip.writestr("stratification_results.csv", _attach_export_metadata(res_df).to_csv(index=False))
            report_zip.writestr("closest_reference_patients.csv", _attach_export_metadata(closest_df).to_csv(index=False))
            report_zip.writestr("cohort_stratification_summary.csv", _cohort_summary_export_df().to_csv(index=False))

        buffer.seek(0)
        yield buffer.getvalue()

    @render.download(filename="closest_reference_patients.csv")
    def download_closest_patients():
        closest_df = closest_reference_patients_df()
        if closest_df is None:
            closest_df = pd.DataFrame()
        yield _attach_export_metadata(closest_df).to_csv(index=False)

    @render.download(filename="cohort_stratification_summary.csv")
    def download_cohort_summary():
        yield _cohort_summary_export_df().to_csv(index=False)

    @output
    @render.plot
    def radar_plot():
        data = get_patient_data_context()
        if not data: return _empty_plot("Radar data unavailable")
        
        opts = list(input.radar_plot_elements() or [])
        feats = list(input.features_to_plot() or [])

        if len(feats) < 3:
            return _empty_plot("Please, select at least 3 features to view")

        fig_radar, _ = _run_patient_analysis(data, feats, opts)
        return fig_radar if fig_radar else _empty_plot("Radar data unavailable")

    @output
    @render.plot
    def curve_plot():
        data = get_patient_data_context()
        if not data: return _empty_plot("Curve data unavailable")
        
        feats = list(input.features_to_plot() or [])

        if len(feats) < 3:
            return _empty_plot("Please, select at least 3 features to view")

        _, fig_curve = _run_patient_analysis(data, feats)
        return fig_curve if fig_curve else _empty_plot("Curve data unavailable")

    def _empty_plot(msg):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        ax.set_axis_off()
        return fig

    @output
    @render.data_frame
    def results_table_output():
        sel_id = patient_selected_id.get()
        res_df = stratification_results_df()

        # Fetch current labels reactively
        lbls = current_labels()
        pos_class_label = lbls["positive_class_label"]
        neg_class_label = lbls["negative_class_label"]
        prob_col = lbls["probability_column"]
        class_col = lbls["class_column"]

        if res_df is None or res_df.empty:
            return render.DataGrid(
                pd.DataFrame({"Message": ["Add or upload patients to generate stratification results."]}),
                width="100%"
            )

        columns_to_show = "ID", prob_col, class_col
        final_cols = [col for col in columns_to_show if col in res_df.columns]
        res_df = res_df[final_cols].reset_index(drop=True)

        res_df[prob_col] = res_df[prob_col].round(3)

        table_styles = None
        if sel_id:
            if (res_df[res_df['ID'] == sel_id].empty):
                return render.DataGrid(
                    pd.DataFrame({"Message": ["Selected ID not found."]}),
                    width="100%"
                )
            sel_id_row = int(res_df[res_df['ID'] == sel_id].index[0])
            pos_rows = res_df[res_df[class_col] == pos_class_label].index.to_list()
            neg_rows = res_df[res_df[class_col] == neg_class_label].index.to_list()
            table_styles = [
                {
                    "rows": [sel_id_row],
                    "cols": list(range(len(res_df.columns))),
                    "style": {
                        "background-color": "#fff3cd",
                        "font-weight": "bold",
                    },
                },
                {
                    "rows": pos_rows,
                    "cols": list(range(len(res_df.columns))),
                    "style": {
                        "color": "#dc3545",
                        "font-weight": "bold",
                    },
                },
                {
                    "rows": neg_rows,
                    "cols": list(range(len(res_df.columns))),
                    "style": {
                        "color": "#198754",
                        "font-weight": "bold",
                    },
                }
            ]

        return render.DataGrid(
            res_df if res_df is not None else pd.DataFrame(),
            width="100%",
            filters=False,
            styles=table_styles
        )
