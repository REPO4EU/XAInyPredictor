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
    entity = labels.get("entity", {})
    text = labels.get("text", {})

    pos_class_label = labels.get("positive_class_label", "Positive")
    neg_class_label = labels.get("negative_class_label", "Negative")
    singular_title = entity.get("singular_title", "Patient")
    plural = entity.get("plural", "patients")
    set_title = entity.get("set_title", "Cohort")
    reference_title = entity.get("reference_title", "Reference Patients")

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
        ),
        ui.page_fluid(
            ui.div(
                ui.div(ui.output_ui("analysis_title_header"), class_="stratification-tabset-title"),
                ui.navset_tab(
                ui.nav_panel(
                    singular_title,
                    ui.div(
                        ui.tags.details(
                            ui.tags.summary(text.get("download_menu", "Download outputs")),
                            ui.div(
                                ui.download_button("download_report_package", text.get("download_report_package", "Report package"), class_="btn-primary btn-sm report-package-download"),
                                ui.download_button("download_stratification_results", text.get("download_results", "Stratification results"), class_="btn-default btn-sm"),
                                ui.download_button("download_closest_patients", text.get("download_closest", "Closest reference candidates"), class_="btn-default btn-sm"),
                                ui.download_button("download_cohort_summary", text.get("download_summary", "Cohort summary"), class_="btn-default btn-sm"),
                                class_="stratification-download-menu",
                            ),
                            class_="stratification-download-details",
                        ),
                        class_="stratification-downloads",
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(ui.output_ui("selected_output_header")),
                            ui.output_ui("stratification_summary"),
                            ui.output_ui("stratification_interpretation"),
                            class_="selected-patient-stratification-card",
                        ),
                        col_widths=12,
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(ui.output_ui("results_table_header")),
                            ui.output_data_frame("results_table_output"),
                            height="240px",
                        ),
                        col_widths=12,
                    ),
                    ui.card(
                        ui.card_header("Profile controls"),
                        ui.div(
                            ui.div(
                                ui.div(
                                    ui.input_select(
                                        id="view_mode",
                                        label="Select View:",
                                        choices={
                                            "radar": titles.get("feature_analysis_short", "Profile Comparison"),
                                            "curve": titles.get("distance_analysis_short", "Feature Curves")
                                        },
                                        selected="radar",
                                    ),
                                    class_="patient-visual-setting patient-visual-setting-view",
                                ),
                                ui.div(
                                    ui.input_action_button("btn_select_default_features", "Top features", class_="btn-default btn-sm feature-control-button"),
                                    ui.input_action_button("btn_clear_features", "Clear", class_="btn-default btn-sm feature-control-button"),
                                    class_="feature-control-row",
                                ),
                                class_="patient-visual-controls-top",
                            ),
                            ui.div(
                                ui.div(
                                    ui.output_ui("features_to_plot_ui"),
                                    class_="feature-selector-inline",
                                ),
                                class_="patient-visual-setting patient-visual-setting-features",
                            ),
                            ui.div(
                                ui.panel_conditional(
                                    "input.view_mode == 'radar'",
                                    ui.div(
                                        ui.input_checkbox_group(
                                            id="radar_plot_elements",
                                            label="Radar Plot Elements:",
                                            choices={
                                                "closest": text.get("radar_closest", f"Closest {plural}"),
                                                "average": text.get("radar_average", f"Average all {plural}"),
                                                "average_0": f"Avg. {neg_class_label}",
                                                "average_1": f"Avg. {pos_class_label}",
                                            },
                                            selected=["closest", "average", "average_0", "average_1"],
                                        ),
                                        class_="patient-visual-setting patient-visual-setting-radar",
                                    ),
                                ),
                                class_="patient-visual-radar-wrap",
                            ),
                            class_="patient-visual-settings-grid",
                        ),
                        class_="patient-visual-settings-card",
                    ),
                    ui.output_ui("dynamic_plot_container"),
                    value="patient",
                ),
                ui.nav_panel(
                    set_title,
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(ui.output_ui("set_summary_header")),
                            ui.output_ui("cohort_stratification_summary"),
                            height="180px",
                        ),
                        col_widths=12,
                    ),
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(ui.output_ui("set_distribution_header")),
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
                    reference_title,
                    ui.layout_columns(
                        ui.card(
                            ui.card_header(ui.output_ui("closest_reference_header")),
                            ui.output_ui("closest_reference_patients_narrative"),
                            ui.output_ui("closest_reference_patients_table"),
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
    selection_revision = reactive.Value(0)

    @reactive.Calc
    def current_labels():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        labels = current_cfg.get("labels", {})
        entity = labels.get("entity", {})
        positive_color = labels.get("positive_class_color", "#dc3545")
        negative_color = labels.get("negative_class_color", "#198754")
        return {
            "positive_class": current_cfg.get("positive_class", "YES"),
            "negative_class": current_cfg.get("negative_class", "NO"),
            "positive_class_label": labels.get("positive_class_label", "Positive"),
            "negative_class_label": labels.get("negative_class_label", "Negative"),
            "positive_class_color": positive_color,
            "negative_class_color": negative_color,
            "positive_class_bg": labels.get("positive_class_bg", f"{positive_color}22"),
            "negative_class_bg": labels.get("negative_class_bg", f"{negative_color}22"),
            "probability_column": labels.get("probability_column", "Stratification Score"),
            "class_column": labels.get("class_column", "Patient Group"),
            "singular": entity.get("singular", "patient"),
            "plural": entity.get("plural", "patients"),
            "singular_title": entity.get("singular_title", "Patient"),
            "plural_title": entity.get("plural_title", "Patients"),
            "set_lower": entity.get("set_lower", "cohort"),
            "set_title": entity.get("set_title", "Cohort"),
            "reference_plural": entity.get("reference_plural", "reference patients"),
            "reference_title": entity.get("reference_title", "Reference Patients"),
            "text": labels.get("text", {}),
        }

    @reactive.Calc
    def current_config():
        return config_reactive.get() if config_reactive else (config_init or {})

    @reactive.Calc
    def stratification_results_df():
        selection_revision.get()
        df = global_input_data.get()
        delta_test = delta_test_reactive.get()
        prob_thr = prob_threshold.get()
        lbls = current_labels()

        if df is None or delta_test is None or df.empty or delta_test.empty:
            return None
        if "ID" not in df.columns or "pred_prob" not in delta_test.columns:
            return None
        if len(df) != len(delta_test):
            return None

        threshold = float(prob_thr) if prob_thr is not None else 0
        res_df = pd.concat(
            [
                df[["ID"]].reset_index(drop=True),
                delta_test[["pred_prob"]].reset_index(drop=True),
            ],
            axis=1,
        )
        res_df["ID"] = res_df["ID"].astype(int)
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

    def _selected_entity_id():
        sel_id = patient_selected_id.get()
        if sel_id is None or sel_id == "" or sel_id == "None":
            return None
        try:
            return int(sel_id)
        except (TypeError, ValueError):
            return None

    @reactive.Calc
    def closest_reference_patients_df():
        selection_revision.get()
        raw_df = global_input_data.get()
        delta_test = delta_test_reactive.get()
        md = model_data.get()
        sel_id = _selected_entity_id()
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
        if len(raw_df) != len(delta_test):
            return None

        raw_df = raw_df.reset_index(drop=True)
        delta_test = delta_test.reset_index(drop=True)

        delta_train = md.get("D_TRAIN")
        if delta_train is None or delta_train.empty:
            return None

        patient_rows = raw_df.index[raw_df["ID"].astype(int) == sel_id].tolist()
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

        rows = []
        for pos in closest_positions:
            score = float(delta_train.iloc[int(pos)]["pred_prob"])
            group = lbls["positive_class_label"] if score >= threshold else lbls["negative_class_label"]
            row = {
                "Reference Rank": len(rows) + 1,
                "Distance": round(float(distances[int(pos)]), 3),
                lbls["probability_column"]: round(score, 3),
                lbls["class_column"]: group,
            }
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
            {"Metric": lbls["plural_title"], "Value": total},
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
        item_count = 0 if input_df is None else len(input_df)
        threshold = float(prob_threshold.get()) if prob_threshold.get() is not None else 0

        return {
            "Use Case": cfg.get("name", "Unknown use case"),
            "Model Objective": cfg.get("description", "Patient stratification model"),
            "Target": cfg.get("target_column", "target"),
            "Decision Threshold": round(threshold, 3),
            "Negative Group": lbls["negative_class_label"],
            "Positive Group": lbls["positive_class_label"],
            f"{lbls['singular_title']} Count": item_count,
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
        lbls = current_labels()
        text = lbls["text"]
        closest_reference_file = f"closest_reference_{_slug_for_filename(lbls['plural'])}.csv"
        set_summary_file = f"{_slug_for_filename(lbls['set_lower'])}_stratification_summary.csv"
        return (
            "XAInyPredictor report package\n"
            "============================\n\n"
            f"Use case: {metadata.get('Use Case', 'Unknown use case')}\n"
            f"Exported at: {metadata.get('Exported At', '')}\n\n"
            "Files included:\n"
            f"- metadata.csv: use case, threshold, {lbls['class_column'].lower()}, export timestamp, and prototype context.\n"
            f"- stratification_results.csv: {lbls['singular']}-level stratification score and assigned group.\n"
            f"- {set_summary_file}: {lbls['set_lower']}-level counts, score summary, and metadata.\n"
            f"- {closest_reference_file}: anonymized closest-reference ranks, distances, scores, and classes for the selected {lbls['singular']} when available.\n\n"
            "Prototype context:\n" +
            text.get(
                "report_context",
                "This package supports patient stratification research. Clinical workflow utility and interpretation should be validated with clinical collaborators.",
            ) + "\n"
        )

    def _slug_for_filename(value):
        slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value))
        return "_".join(part for part in slug.split("_") if part)

    def _closest_reference_filename():
        lbls = current_labels()
        return f"closest_reference_{_slug_for_filename(lbls['plural'])}.csv"

    def _set_summary_filename():
        lbls = current_labels()
        return f"{_slug_for_filename(lbls['set_lower'])}_stratification_summary.csv"

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
                entity_label=lbls["singular_title"],
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
                "closest": lbls["text"].get("radar_closest", f"Closest {lbls['plural']}"),
                "average": lbls["text"].get("radar_average", f"Average all {lbls['plural']}"),
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
        current_selection = _selected_entity_id()
        if current_selection is None or int(current_selection) not in all_ids:
            patient_selected_id.set(all_ids[0])
            ui.update_selectize("local_patient_select", selected=all_ids[0])

    @output(suspend_when_hidden=False)
    @render.ui
    def analysis_title_header():
        lbls = current_labels()
        return lbls["text"].get("analysis_title", "Stratification Analysis")

    @output(suspend_when_hidden=False)
    @render.ui
    def selected_output_header():
        lbls = current_labels()
        return lbls["text"].get(
            "selected_output_title",
            f"Selected {lbls['singular_title']} Stratification",
        )

    @output(suspend_when_hidden=False)
    @render.ui
    def results_table_header():
        cfg = current_config()
        titles = cfg.get("titles", {})
        lbls = current_labels()
        text = lbls["text"]
        prob_col = lbls["probability_column"]
        class_col = lbls["class_column"]
        return ui.div(
            titles.get("prediction_results", "Prediction results "),
            ui.popover(
                ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                ui.tags.div(
                    ui.tags.b(text.get("results_table_help_title", "Stratification results:")),
                    ui.tags.p(text.get("results_table_help", f"The table shows model-based {lbls['singular']} stratification:")),
                    ui.tags.ul(
                        ui.tags.li(ui.tags.b(f"{prob_col}:"), text.get("score_help", f" Indicates the score used to assign the {lbls['singular']} class.")),
                        ui.tags.li(ui.tags.b(f"{class_col}:"), text.get("class_help", f" Assigns the {lbls['singular']} to {lbls['positive_class_label']} or {lbls['negative_class_label']}.")),
                    ),
                    style="width: 250px;",
                ),
                placement="right",
            ),
        )

    @output(suspend_when_hidden=False)
    @render.ui
    def set_summary_header():
        lbls = current_labels()
        return lbls["text"].get("set_summary_title", f"{lbls['set_title']} Stratification Summary")

    @output(suspend_when_hidden=False)
    @render.ui
    def set_distribution_header():
        lbls = current_labels()
        text = lbls["text"]
        return ui.div(
            text.get("set_distribution_title", f"{lbls['set_title']} Score Distribution "),
            ui.popover(
                ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                ui.tags.div(
                    ui.tags.b(text.get("set_distribution_help_title", f"{lbls['set_title']} score distribution:")),
                    ui.tags.p(
                        text.get(
                            "set_distribution_help",
                            f"{lbls['plural_title']} are ordered by stratification score. The dashed line marks the current decision threshold and the selected {lbls['singular']} is highlighted.",
                        )
                    ),
                    style="width: 250px;",
                ),
                placement="right",
            ),
        )

    @output(suspend_when_hidden=False)
    @render.ui
    def closest_reference_header():
        lbls = current_labels()
        text = lbls["text"]
        return ui.div(
            text.get("closest_reference_title", f"Closest {lbls['reference_title']} "),
            ui.popover(
                ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                ui.tags.div(
                    ui.tags.b(text.get("closest_reference_help_title", f"Closest {lbls['reference_plural']}:")),
                    ui.tags.p(
                        text.get(
                            "closest_reference_help",
                            f"{lbls['reference_title']} are ranked by distance in the model contribution space, using the same feature effects that drive the explanation plots.",
                        )
                    ),
                    style="width: 260px;",
                ),
                placement="right",
            ),
        )

    @output(suspend_when_hidden=False)
    @render.ui
    def patient_selector_ui():
        """
        Dynamic UI for Patient Selection. The function listens to changes in
        global_input_data variable and according to this, it creates a dropdown
        to select patients.
        """
        lbls = current_labels()
        df = global_input_data.get()
        if df is None or df.empty or 'ID' not in df.columns:
            return ui.p(lbls["text"].get("empty_results_message", f"Add or upload {lbls['plural']} in Data Input to generate stratification outputs."))
        
        # Get list of patient IDs
        all_ids = sorted(df['ID'].astype(int).tolist())

        # Check current selection from global state
        current_selection = _selected_entity_id()

        # Default to first if no current selection
        if current_selection is None or int(current_selection) not in all_ids:
            selected_val = all_ids[0]
        else:
            selected_val = int(current_selection)

        return ui.input_selectize(
            id="local_patient_select",
            label=lbls["text"].get("select_entity_label", f"Select {lbls['singular']}:"),
            choices=all_ids,
            selected=selected_val,
            options={
                "create": False,  # Don't allow creating new options
                "allowEmptyOption": False,
                "placeholder": lbls["text"].get("search_entity_placeholder", f"Search by {lbls['singular']} ID..."),
                "maxItems": 1
            }
        )

    @output(suspend_when_hidden=False)
    @render.ui
    def features_to_plot_ui():
        """
        Dynamic UI to define the features to plot. The UI gets the default features
        to plot from the argument features_to_plot, which obtains them from the
        features that appear in the formula (discarding the ones that do not appear).
        """
        md = model_data.get()
        cfg = current_config()
        feature_names = md.get("FEATURE_ORDER_DISPLAY", []) if md else []
        features_to_plot = md.get("FEATS_IN_FORMULA", []) if md else []
        max_default_features = cfg.get("default_selected_features")
        if max_default_features:
            features_to_plot = features_to_plot[:int(max_default_features)]
        return ui.input_selectize(
            id="features_to_plot",
            label="Select features to view:",
            choices=feature_names,
            selected=features_to_plot,
            multiple=True,
            options={
                "plugins": ["remove_button"],
                "maxOptions": 100,
                "placeholder": "Choose features",
            },
        )

    @reactive.Effect
    @reactive.event(input.btn_select_default_features)
    def _select_default_features():
        md = model_data.get()
        cfg = current_config()
        if not md:
            return
        features_to_plot = md.get("FEATS_IN_FORMULA", [])
        max_default_features = cfg.get("default_selected_features", 5)
        features_to_plot = features_to_plot[:int(max_default_features)]
        ui.update_selectize("features_to_plot", selected=features_to_plot)

    @reactive.Effect
    @reactive.event(input.btn_clear_features)
    def _clear_features():
        ui.update_selectize("features_to_plot", selected=[])

    @output(suspend_when_hidden=False)
    @render.ui
    def dynamic_plot_container():
        mode = input.view_mode()
        selected_features = input.features_to_plot()
        cfg = current_config()

        lbls = current_labels()
        neg = lbls["negative_class_label"]
        pos = lbls["positive_class_label"]

        # Calculate dynamic height for Curve plot
        # Base height + (pixels per feature * number of features)
        n_feats = len(selected_features) if selected_features else 0
        max_curve_features = int(cfg.get("max_curve_features", 8))
        curve_height_px = 300 + (min(n_feats, max_curve_features) * 210)
        
        # Define UI cards
        radar_card = ui.card(
            ui.card_header(
                ui.div(
                    lbls["text"].get("profile_comparison_title", "Patient Profile Comparison "),
                    ui.popover(
                        ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                        ui.tags.div(
                            ui.tags.b("Understanding the radar plot:"),
                            ui.tags.p(lbls["text"].get("radar_help_intro", f"The radar plot compares the values of the features from a selected {lbls['singular']} (red) with three different distributions:")),
                            ui.tags.ul(
                                ui.tags.li(ui.tags.b(f"Average all {lbls['plural']}:"), f" The average values from all {lbls['plural']} in the model (blue)."),
                                ui.tags.li(ui.tags.b(f"Avg. {neg}:"), f" The average values from all {neg} {lbls['plural']} in the model (green)."),
                                ui.tags.li(ui.tags.b(f"Avg. {pos}:"), f" The average values from all {pos} {lbls['plural']} in the model (yellow)."),
                            ),
                            ui.tags.p(lbls["text"].get("radar_help_outro", f"Comparing these values helps contextualize why the selected {lbls['singular']} falls into a given stratification group.")),
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
                            ui.tags.p(lbls["text"].get("curves_help", f"The curves plot displays the distribution of values from a specific feature across the {lbls['plural']} in the model, ordered from lowest to highest (blue dots). It highlights in red the selected {lbls['singular']}, putting its feature values in context.")),
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
            with reactive.isolate():
                selection_revision.set(selection_revision.get() + 1)
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

        if (
            y_test is None
            or delta_test is None
            or delta_test.empty
            or "pred_prob" not in delta_test.columns
        ):
            return
        if len(y_test) != len(delta_test):
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
        selection_revision.get()
        """
        Gathers all the dataframes and IDs needed for analysis.
        Returns a dictionary of data or None if invalid.
        """
        raw_df = global_input_data.get()
        delta_test = delta_test_reactive.get()
        x_test = x_test_reactive.get()
        patient_id = _selected_entity_id()

        md = model_data.get()
        if md is None:
            return None

        delta_train = md.get("D_TRAIN")
        x_train = md.get("X_TRAIN")
        y_train = md.get("Y_TRAIN")

        # Validation checks
        if any((x is None) or (isinstance(x, pd.DataFrame) and x.empty) for x in [raw_df, delta_train, delta_test, x_train, y_train, patient_id]):
            return None
        if len(raw_df) != len(delta_test) or len(raw_df) != len(x_test):
            return None

        raw_df = raw_df.reset_index(drop=True)
        delta_test = delta_test.reset_index(drop=True)
        x_test = x_test.reset_index(drop=True)

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

    @output(suspend_when_hidden=False)
    @render.ui
    def stratification_summary():
        try:
            sel_id = _selected_entity_id()
            prob_thr = prob_threshold.get()
            lbls = current_labels()
            res_df = stratification_results_df()

            prob_col = lbls["probability_column"]
            class_col = lbls["class_column"]

            if res_df is None or res_df.empty or sel_id is None:
                return ui.div(lbls["text"].get("empty_individual_summary", f"Add or upload {lbls['plural']} to review individual stratification outputs."), class_="stratification-summary-empty")

            patient_row = res_df[res_df["ID"].astype(int) == sel_id]
            if patient_row.empty:
                return ui.div(lbls["text"].get("selected_entity_not_found", f"Selected {lbls['singular']} not found."), class_="stratification-summary-empty")

            score = float(patient_row[prob_col].iloc[0])
            threshold = float(prob_thr) if prob_thr is not None else 0
            group = str(patient_row[class_col].iloc[0])
            group_color = lbls["positive_class_color"] if group == lbls["positive_class_label"] else lbls["negative_class_color"]

            return ui.div(
                ui.div(
                    ui.div(lbls["singular_title"], class_="stratification-summary-label"),
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
                    ui.div(group, class_="stratification-summary-value", style=f"color: {group_color};"),
                    class_="stratification-summary-item",
                ),
                class_="stratification-summary",
            )
        except Exception as exc:
            return ui.div(f"Unable to update selected stratification: {exc}", class_="stratification-summary-empty")

    @output(suspend_when_hidden=False)
    @render.ui
    def stratification_interpretation():
        try:
            sel_id = _selected_entity_id()
            prob_thr = prob_threshold.get()
            lbls = current_labels()
            res_df = stratification_results_df()

            if res_df is None or res_df.empty or sel_id is None:
                return None

            prob_col = lbls["probability_column"]
            class_col = lbls["class_column"]
            patient_row = res_df[res_df["ID"].astype(int) == sel_id]
            if patient_row.empty:
                return None

            score = float(patient_row[prob_col].iloc[0])
            threshold = float(prob_thr) if prob_thr is not None else 0
            group = str(patient_row[class_col].iloc[0])
            direction = "above or equal to" if score >= threshold else "below"
            positive_label = lbls["positive_class_label"]
            negative_label = lbls["negative_class_label"]
            interpretation = lbls["text"].get(
                "stratification_interpretation",
                f"This {lbls['singular']} is assigned to {group} because the stratification score ({score:.3f}) is {direction} the decision threshold ({threshold:.3f}).",
            )

            return ui.div(
                ui.p(
                    interpretation.format(
                        entity=lbls["singular"],
                        entity_title=lbls["singular_title"],
                        group=group,
                        score=f"{score:.3f}",
                        threshold=f"{threshold:.3f}",
                        direction=direction,
                    ),
                    class_="stratification-interpretation-main",
                ),
                ui.div(
                    ui.div(
                        ui.tags.b(f"Below threshold: "),
                        f"assigned to {negative_label}.",
                        class_="stratification-rule stratification-rule-negative",
                        style=f"background: {lbls['negative_class_bg']}; color: {lbls['negative_class_color']};",
                    ),
                    ui.div(
                        ui.tags.b(f"At or above threshold: "),
                        f"assigned to {positive_label}.",
                        class_="stratification-rule stratification-rule-positive",
                        style=f"background: {lbls['positive_class_bg']}; color: {lbls['positive_class_color']};",
                    ),
                    class_="stratification-rules",
                ),
                ui.p(
                    lbls["text"].get(
                        "stratification_disclaimer",
                        "Research prototype output: this stratification supports cohort-level clinical decision support research and should be interpreted with clinical collaborators during utility validation.",
                    ),
                    class_="stratification-disclaimer",
                ),
                class_="stratification-interpretation",
            )
        except Exception as exc:
            return ui.div(f"Unable to update selected stratification explanation: {exc}", class_="stratification-summary-empty")

    @output(suspend_when_hidden=False)
    @render.ui
    def cohort_stratification_summary():
        res_df = stratification_results_df()
        lbls = current_labels()

        if res_df is None or res_df.empty:
            return ui.div(lbls["text"].get("empty_set_summary", f"Add or upload {lbls['plural']} to summarize {lbls['set_lower']}-level stratification."), class_="stratification-summary-empty")

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
                ui.div(lbls["plural_title"], class_="stratification-summary-label"),
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
                ui.div(str(negative_count), class_="stratification-summary-value", style=f"color: {lbls['negative_class_color']};"),
                class_="stratification-summary-item",
            ),
            ui.div(
                ui.div(positive_label, class_="stratification-summary-label"),
                ui.div(f"{positive_count} ({positive_pct:.0f}%)", class_="stratification-summary-value", style=f"color: {lbls['positive_class_color']};"),
                class_="stratification-summary-item",
            ),
            class_="stratification-summary",
        )

    @output(suspend_when_hidden=False)
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
                    ui.div(lbls["text"].get("reference_set_label", "Reference cohort"), class_="model-card-label"),
                    ui.div(str(reference_n), class_="model-card-value"),
                    class_="model-card-item",
                ),
                ui.div(
                    ui.div("Input variables", class_="model-card-label"),
                    ui.div(str(len(features)), class_="model-card-value"),
                    class_="model-card-item",
                ),
                ui.div(
                    ui.div(lbls["text"].get("groups_label", "Patient groups"), class_="model-card-label"),
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
                ui.div(lbls["text"].get("reference_source_label", "Reference cohort source"), class_="model-card-label"),
                ui.p(metadata.get("cohort_source", lbls["text"].get("reference_source_missing", "Reference cohort source not specified.")), class_="model-card-copy"),
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

    @output(suspend_when_hidden=False)
    @render.plot
    def cohort_score_distribution_plot():
        if input.stratification_tabs() != "cohort":
            return _empty_plot("Open the set tab to view this plot")

        res_df = stratification_results_df()
        sel_id = _selected_entity_id()
        threshold = float(prob_threshold.get()) if prob_threshold.get() is not None else 0
        lbls = current_labels()

        if res_df is None or res_df.empty:
            return _empty_plot(lbls["text"].get("no_set_scores", f"No {lbls['set_lower']} scores available"))

        prob_col = lbls["probability_column"]
        class_col = lbls["class_column"]
        sorted_df = res_df.sort_values(prob_col, ascending=False).reset_index(drop=True)
        colors = np.where(
            sorted_df[class_col] == lbls["positive_class_label"],
            lbls["positive_class_color"],
            lbls["negative_class_color"],
        )

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(np.arange(len(sorted_df)), sorted_df[prob_col], color=colors, alpha=0.82, width=0.85)
        ax.axhline(threshold, color="#1f2d3d", linestyle="--", linewidth=1.5, label=f"Threshold {threshold:.3f}")

        if sel_id is not None and sel_id in sorted_df["ID"].astype(int).values:
            selected_idx = int(sorted_df.index[sorted_df["ID"].astype(int) == sel_id][0])
            selected_score = float(sorted_df.loc[selected_idx, prob_col])
            ax.scatter(selected_idx, selected_score, color="#ffc107", edgecolor="#1f2d3d", s=110, zorder=5, label=f"{lbls['singular_title']} {int(sel_id)}")

        ax.set_title(lbls["text"].get("set_scores_plot_title", "Cohort stratification scores"), fontweight="bold")
        ax.set_xlabel(lbls["text"].get("set_scores_plot_xlabel", f"{lbls['plural_title']} ordered by score"))
        ax.set_ylabel(prob_col)
        ax.set_ylim(0, max(1.0, float(sorted_df[prob_col].max()) * 1.08))
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        return fig

    @output(suspend_when_hidden=False)
    @render.plot
    def global_feature_importance_plot():
        if input.stratification_tabs() != "model":
            return _empty_plot("Open the model tab to view this plot")

        md = model_data.get()
        delta_test = delta_test_reactive.get()
        lbls = current_labels()

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
                    label=lbls["text"].get("current_set_label", "Current cohort"),
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

    @output(suspend_when_hidden=False)
    @render.ui
    def closest_reference_patients_table():
        closest_df = closest_reference_patients_df()
        if closest_df is None or closest_df.empty:
            return None

        return ui.tags.div(
            ui.HTML(closest_df.to_html(index=False, classes="reference-candidates-table", border=0)),
            class_="reference-candidates-table-wrap",
        )

    @output(suspend_when_hidden=False)
    @render.ui
    def closest_reference_patients_narrative():
        closest_df = closest_reference_patients_df()
        lbls = current_labels()

        if closest_df is None or closest_df.empty:
            return ui.div(lbls["text"].get("reference_comparison_empty", f"Reference comparison requires a selected {lbls['singular']} and an available reference {lbls['set_lower']}."), class_="closest-patients-narrative")

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
            ui.div(
                lbls["text"].get(
                    "closest_reference_narrative",
                    f"The selected {lbls['singular']} is most similar to {lbls['reference_plural']} mostly assigned to {top_group} ({top_count}/{total}).{score_text}",
                ).format(
                    entity=lbls["singular"],
                    reference_plural=lbls["reference_plural"],
                    top_group=top_group,
                    top_count=top_count,
                    total=total,
                    score_text=score_text,
                ),
                class_="closest-patients-narrative",
            ),
            ui.div(
                lbls["text"].get(
                    "closest_reference_privacy_note",
                    "Reference records come from the model reference population, not from the current input set. If the reference population contains private or sensitive records, expose only anonymized summaries or distances rather than row-level reference data.",
                ),
                class_="reference-privacy-note",
            ),
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
            report_zip.writestr(_closest_reference_filename(), _attach_export_metadata(closest_df).to_csv(index=False))
            report_zip.writestr(_set_summary_filename(), _cohort_summary_export_df().to_csv(index=False))

        buffer.seek(0)
        yield buffer.getvalue()

    @render.download(filename=_closest_reference_filename)
    def download_closest_patients():
        closest_df = closest_reference_patients_df()
        if closest_df is None:
            closest_df = pd.DataFrame()
        yield _attach_export_metadata(closest_df).to_csv(index=False)

    @render.download(filename=_set_summary_filename)
    def download_cohort_summary():
        yield _cohort_summary_export_df().to_csv(index=False)

    @output(suspend_when_hidden=False)
    @render.plot
    def radar_plot():
        if input.stratification_tabs() != "patient" or input.view_mode() != "radar":
            return _empty_plot("Open profile comparison to view this plot")

        data = get_patient_data_context()
        if not data: return _empty_plot("Radar data unavailable")
        
        opts = list(input.radar_plot_elements() or [])
        feats = list(input.features_to_plot() or [])

        if len(feats) < 3:
            return _empty_plot("Please, select at least 3 features to view")

        fig_radar, _ = _run_patient_analysis(data, feats, opts)
        return fig_radar if fig_radar else _empty_plot("Radar data unavailable")

    @output(suspend_when_hidden=False)
    @render.plot
    def curve_plot():
        if input.stratification_tabs() != "patient" or input.view_mode() != "curve":
            return _empty_plot("Open feature curves to view this plot")

        data = get_patient_data_context()
        if not data: return _empty_plot("Curve data unavailable")
        
        cfg = current_config()
        feats = list(input.features_to_plot() or [])
        max_curve_features = int(cfg.get("max_curve_features", 8))
        if len(feats) > max_curve_features:
            feats = feats[:max_curve_features]

        if len(feats) < 3:
            return _empty_plot("Please, select at least 3 features to view")

        _, fig_curve = _run_patient_analysis(data, feats)
        return fig_curve if fig_curve else _empty_plot("Curve data unavailable")

    def _empty_plot(msg):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        ax.set_axis_off()
        return fig

    @output(suspend_when_hidden=False)
    @render.data_frame
    def results_table_output():
        sel_id = _selected_entity_id()
        res_df = stratification_results_df()

        # Fetch current labels reactively
        lbls = current_labels()
        pos_class_label = lbls["positive_class_label"]
        neg_class_label = lbls["negative_class_label"]
        prob_col = lbls["probability_column"]
        class_col = lbls["class_column"]

        if res_df is None or res_df.empty:
            return render.DataGrid(
                pd.DataFrame({"Message": [lbls["text"].get("empty_results_message", f"Add or upload {lbls['plural']} to generate stratification results.")]}),
                width="100%"
            )

        columns_to_show = "ID", prob_col, class_col
        final_cols = [col for col in columns_to_show if col in res_df.columns]
        res_df = res_df[final_cols].reset_index(drop=True)

        res_df[prob_col] = res_df[prob_col].round(3)

        table_styles = None
        if sel_id is not None:
            if (res_df[res_df['ID'].astype(int) == sel_id].empty):
                return render.DataGrid(
                    pd.DataFrame({"Message": ["Selected ID not found."]}),
                    width="100%"
                )
            sel_id_row = int(res_df[res_df['ID'].astype(int) == sel_id].index[0])
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
                        "color": lbls["positive_class_color"],
                        "font-weight": "bold",
                    },
                },
                {
                    "rows": neg_rows,
                    "cols": list(range(len(res_df.columns))),
                    "style": {
                        "color": lbls["negative_class_color"],
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
