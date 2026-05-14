import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui

# Local imports
from XAInyPredictor.modules.analyzer import analyze_patient, threshold_for_target_fnr


@module.ui
def prediction_ui(config=None):
    if config is None:
        config = {}

    titles = config.get("titles", {})
    labels = config.get("labels", {})
    help_texts = config.get("help_texts", {})

    pos_class = labels.get("positive_class", "Positive")
    neg_class = labels.get("negative_class", "Negative")
    prob_col = labels.get("probability_column", "Probability")
    class_col = labels.get("class_column", "Class")

    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("patient_selector_ui"),
            ui.div(
                ui.input_numeric(
                    id="fnr_threshold",
                    label= ui.div(
                        "Min. % false negatives: ",
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
                    "radar": titles.get("feature_analysis", "Feature Analysis (Radar)"),
                    "curve": titles.get("distance_analysis", "Distance Analysis (Curve)")
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
                        "average_0": f"Avg. {neg_class}",
                        "average_1": f"Avg. {pos_class}",
                    },
                    selected=["closest", "average", "average_0", "average_1"],
                ),
            ),
        ),
        ui.page_fluid(
            ui.layout_columns(
                ui.card(
                    ui.card_header(
                        ui.div(
                            titles.get("prediction_results", "Prediction results "),
                            ui.popover(
                                ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                                ui.tags.div(
                                    ui.tags.b("Prediction results:"),
                                    ui.tags.p("The table shows the predictions of the model:"),
                                    ui.tags.ul(
                                        ui.tags.li(ui.tags.b(f"{prob_col}:"), f" Indicates the probability. The higher, the more likely."),
                                        ui.tags.li(ui.tags.b(f"{class_col}:"), f" Classifies as {pos_class} or {neg_class}."),
                                    ),
                                    style="width: 250px;"
                                ),
                                placement="right"
                            )
                        )
                    ),
                    ui.output_data_frame("results_table_output"),
                    height="300px",
                ),
                col_widths=12
            ),
            ui.output_ui("dynamic_plot_container")
        ),
    )


@module.server
def server(input: Inputs, output: Outputs, session: Session, global_input_data, patient_selected_id, model_data, delta_test_reactive, x_test_reactive, prob_threshold, config_init, config_reactive=None):
    cfg = config_init if config_init else {}

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
            return ui.p("No valid data loaded.")
        
        # Get list of patient IDs
        all_ids = sorted(df['ID'].astype(int).tolist())

        # Check current selection from global state
        current_selection = patient_selected_id.get()

        # Default to first if no current selection
        if current_selection == None or int(current_selection) not in all_ids:
            selected_val = all_ids[0]
        else:
            selected_val = int(current_selection)
        print(f"Patient selected (run analysis): {selected_val}")

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
        
        # Calculate dynamic height for Curve plot
        # Base height + (pixels per feature * number of features)
        n_feats = len(selected_features) if selected_features else 0
        curve_height_px = 300 + (n_feats * 250) 
        
        # Define UI cards
        radar_card = ui.card(
            ui.card_header(
                ui.div(
                    "Feature Analysis (Radar) ",
                    ui.popover(
                        ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                        ui.tags.div(
                            ui.tags.b("Understanding the radar plot:"),
                            ui.tags.p("The radar plot compares the values of the features from a selected patient with three different distributions:"),
                            ui.tags.ul(
                                ui.tags.li(ui.tags.b("Average all patients:"), " The average values from all patients in the model."),
                                ui.tags.li(ui.tags.b("Avg. RAI-R negative:"), " The average values from all RAI-R negative patients in the model (green)."),
                                ui.tags.li(ui.tags.b("Avg. RAI-R positive:"), " The average values from all RAI-R positive patients in the model (red)."),
                            ),
                            ui.tags.p("Comparing these values help to evaluate if a patient has been correctly classified or not."),
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
                    "Distance Analysis (Curves) ",
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

        # Calculate probability threshold
        y_test.sort_index(inplace=True)
        delta_test.sort_index(inplace=True)
        threshold, fnr = threshold_for_target_fnr(
            y_test.to_numpy(),
            delta_test['pred_prob'].to_numpy(),
            target_fnr=target_fnr
        )

        prob_threshold.set(threshold)
        print(f"Updated Threshold: {threshold:.3f} (Target FNR: {target_fnr})")

    # --- Calculation ---

    @reactive.Calc
    def get_patient_data_context():
        """
        Gathers all the dataframes and IDs needed for analysis.
        Returns a dictionary of data or None if invalid.
        """
        df = global_input_data.get()
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
        if any((x is None) or (isinstance(x, pd.DataFrame) and x.empty) for x in [df, delta_train, delta_test, x_train, y_train, patient_id]):
            return None

        all_ids = sorted(df['ID'].astype(int).tolist())
        if patient_id == None or int(patient_id) not in all_ids:
            return None

        # Standardize columns
        df.columns = [col.replace(' ', '_') for col in df.columns]

        return {
            "patient_id": patient_id,
            "df": df,
            "delta_train": delta_train,
            "delta_test": delta_test,
            "x_train": x_train,
            "y_train": y_train,
            "x_test": x_test,
        }

    # --- Outputs ---

    @output
    @render.plot
    def radar_plot():
        data = get_patient_data_context()
        if not data: return _empty_plot("Radar data unavailable")
        
        opts = input.radar_plot_elements()
        feats = input.features_to_plot()

        if len(feats) < 3:
            return _empty_plot("Please, select at least 3 features to view")

        # We call the plotting function INSIDE render.plot
        # This ensures it redraws correctly on resize.
        fig_radar, _ = analyze_patient(
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
        )
        return fig_radar if fig_radar else _empty_plot("Radar data unavailable")

    @output
    @render.plot
    def curve_plot():
        data = get_patient_data_context()
        if not data: return _empty_plot("Curve data unavailable")
        
        feats = input.features_to_plot()

        if len(feats) < 3:
            return _empty_plot("Please, select at least 3 features to view")

        # Retrieve the second figure (Curve) from the function
        # Note: Ideally you'd split analyze_patient into two functions to save performance,
        # but calling it again here is the safest way to ensure the plot exists for this context.
        _, fig_curve = analyze_patient(
            patient_id=data["patient_id"],
            df=data["df"],
            delta_train=data["delta_train"],
            delta_test=data["delta_test"],
            x_train=data["x_train"],
            y_train=data["y_train"],
            x_test=data["x_test"],
            features_to_plot=feats,
            n_dists=3,
            # We don't care about radar options here
        )
        return fig_curve if fig_curve else _empty_plot("Curve data unavailable")

    def _empty_plot(msg):
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, msg, ha="center", va="center")
        ax.set_axis_off()
        return fig

    @output
    @render.data_frame
    def results_table_output():
        df = global_input_data.get()
        delta_test = delta_test_reactive.get()
        sel_id = patient_selected_id.get()
        prob_thr = prob_threshold.get()

        cfg = config_init if config_init else {}
        labels = cfg.get("labels", {})
        pos_class = labels.get("positive_class", "YES")
        neg_class = labels.get("negative_class", "NO")
        prob_col = labels.get("probability_column", "Probability")
        class_col = labels.get("class_column", "Class")

        if df is None or delta_test is None or df.empty or delta_test.empty:
            return render.DataGrid(
                pd.DataFrame({"Message": ["No data available."]}),
                width="100%"
            )

        if prob_thr is None:
            prob_thr = 0

        res_df = df.copy()
        if "pred_prob" in delta_test.columns:
            res_df = pd.concat([res_df, delta_test[["pred_prob"]]], axis=1)
            res_df = res_df.rename(columns={"pred_prob": prob_col}).sort_values(by=['ID'])
            res_df[class_col] = np.where(res_df[prob_col] >= prob_thr, pos_class, neg_class)

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
            pos_rows = res_df[res_df[class_col] == pos_class].index.to_list()
            neg_rows = res_df[res_df[class_col] == neg_class].index.to_list()
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
