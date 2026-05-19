from os import path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from shiny import Inputs, Outputs, Session, module, reactive, render, req, ui


@module.ui
def data_exploration_ui(config=None):
    if config is None:
        config = {}

    return ui.layout_sidebar(
        ui.sidebar(
            ui.output_ui("feature_dropdown"),
            ui.input_select(
                id="reference_data",
                label= ui.div(
                    "Reference population: ",
                    ui.popover(
                        ui.span(ui.tags.i(class_="glyphicon glyphicon-info-sign"), "", style="color: #007bc2; cursor: pointer; font-size: 0.9em;"),
                        ui.tags.div(
                            ui.tags.b("Reference population:"),
                            ui.tags.p("This selects the population that you want to show in the distribution."),
                            ui.tags.ul(
                                ui.tags.li(ui.tags.b("Input data:"), " It will show the distribution of the input data."),
                                ui.tags.li(ui.tags.b("Model:"), " It will show the distribution of the model data."),
                            ),
                            style="width: 250px;"
                        ),
                        placement="right"
                    )
                ),
                choices=['Input data', 'Model'],
                selected='Input data',
            ),
            ui.output_ui("patient_selector_ui"),
            ui.p("Select a patient to highlight their value in the distribution.", style="font-size: 0.8em; color: gray;"),
        ),
        ui.page_fluid(
            ui.layout_columns(
                ui.card(
                    ui.card_header(
                        ui.div(
                            "Feature Distribution Analysis",
                            ui.tooltip(
                                ui.span(ui.tags.i(class_="glyphicon glyphicon-question-sign"), style="font-size: 0.8em; color: gray; margin-left: 5px;"),
                                "Blue bars: Reference population distribution. Red dashed line: Selected patient's value.",
                            )
                        )
                    ),
                    ui.output_plot("plot_feature_distribution"),
                    full_screen=True
                ),
                ui.card(
                    ui.card_header("Feature Statistics"),
                    ui.output_ui("feature_stats_ui"),
                    full_screen=True
                ),
                col_widths=(9, 3),
            ),
        ),
    )


@module.server
def server(input: Inputs, output: Outputs, session: Session, global_input_data, model_data, patient_selected_id, config_init=None, config_reactive=None):

    display_to_name = reactive.Value({})

    @reactive.Effect
    def _rebuild_feature_list():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        features = current_cfg.get("features", []) if current_cfg else []
        display_to_name.set({f["display_name"]: f["name"] for f in features})

    @output
    @render.ui
    def feature_dropdown():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        features = current_cfg.get("features", []) if current_cfg else []
        choices = [(f["display_name"], f["display_name"]) for f in features]
        if not choices:
            return ui.p("No features configured", style="font-size: 0.85em; color: gray;")
        return ui.input_select(
            id="feature_to_plot",
            label="Feature to plot:",
            choices=dict(choices),
            selected=features[0].get("display_name") if features else None,
        )

    @reactive.Effect
    def _ensure_valid_patient():
        """
        Whenever data changes, ensure the selected patient is valid.
        If it's None or not in the new dataset, default to the first ID.
        """
        df = global_input_data.get()
        if df is None or df.empty or 'ID' not in df.columns:
            return
        
        current_id = patient_selected_id.get()
        all_ids = sorted(df['ID'].astype(int).tolist())
        
        if current_id is None or int(current_id) not in all_ids:
            patient_selected_id.set(all_ids[0])

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
        print(f"Patient selected (inspect): {selected_val}")

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

    @output
    @render.ui
    def feature_stats_ui():
        """
        Display statistics for the selected feature
        """
        reference_data = input.reference_data()
        if reference_data == 'Input data':
            df = global_input_data.get()
        else:
            md = model_data.get()
            df = md.get("X_TRAIN_RAW") if md else None

        if df is None or df.empty:
            return ui.p("No data available")
        
        # Ensure column names have the proper format for the UI
        df.columns = [col.replace('_', ' ') for col in df.columns]

        display_name = str(input.feature_to_plot())
        col_name = display_to_name.get().get(display_name, display_name)
        # Also try with space replacement as fallback
        if col_name not in df.columns:
            col_name_alt = col_name.replace(' ', '_')
            if col_name_alt in [c.replace(' ', '_') for c in df.columns]:
                col_name = col_name_alt

        feature = col_name
        if feature not in df.columns:
            return ui.p("Feature not found")
        
        # Get the feature data
        feature_data = df[feature].dropna()
        
        if pd.api.types.is_numeric_dtype(feature_data):
            stats = {
                'Mean': f"{feature_data.mean():.2f}",
                'Median': f"{feature_data.median():.2f}",
                'Std Dev': f"{feature_data.std():.2f}",
                'Min': f"{feature_data.min():.2f}",
                'Max': f"{feature_data.max():.2f}",
                'Count': f"{len(feature_data)}",
                'Missing': f"{df[feature].isnull().sum()}"
            }
        else:
            stats = {
                'Unique Values': f"{feature_data.nunique()}",
                'Most Common': f"{feature_data.mode().iloc[0] if not feature_data.mode().empty else 'N/A'}",
                'Count': f"{len(feature_data)}",
                'Missing': f"{df[feature].isnull().sum()}"
            }
        
        # Create statistics table
        stat_items = []
        for key, value in stats.items():
            stat_items.append(ui.p(ui.tags.b(key + ": "), value))
        
        return ui.div(*stat_items, style="font-size: 0.9em;")

    @output
    @render.plot
    def plot_feature_distribution():
        """
        Plot distribution of a feature within the input dataset and highlight
        the value from a specific patient.
        """
        df = global_input_data.get()
        if df is None or df.empty:
            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig
        
        # Ensure column names have the proper format for the UI
        df.columns = [col.replace('_', ' ') for col in df.columns]
        display_name = str(input.feature_to_plot())
        col_name = display_to_name.get().get(display_name, display_name)
        if col_name not in df.columns:
            col_name_alt = col_name.replace(' ', '_')
            if col_name_alt in [c.replace(' ', '_') for c in df.columns]:
                col_name = col_name_alt

        feature = col_name
        patient_id = patient_selected_id.get()

        if feature not in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"Feature '{feature}' not found", ha="center", va="center", fontsize=16)
            return fig

        # Filter for specific patient
        patient_row = df[df['ID'] == patient_id]
        if patient_row.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"Patient {patient_id} not found in data!", ha="center", va="center", fontsize=16)
            return fig
        patient_val = patient_row[feature].values[0]

        # Get model data if necessary
        reference_data = input.reference_data()
        if reference_data == 'Model':
            md = model_data.get()
            df = md.get("X_TRAIN_RAW") if md else None
            if df is not None:
                df = pd.concat([df, patient_row])

        # Create figure
        fig, ax = plt.subplots()

        # Handle categorical vs numerical
        if pd.api.types.is_numeric_dtype(df[feature]):
            # Create histogram for numerical data
            ax.hist(df[feature].dropna(), bins=30, color='#4682B4', alpha=0.5, edgecolor='black', linewidth=0.5)
            ax.set_title(f"Distribution of {feature}\nPatient {patient_id} value: {patient_val}", 
                        fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Frequency', fontsize=12)

            # Add vertical line for patient value
            ax.axvline(patient_val, color='red', linewidth=3, linestyle='--',
                       label=f'Patient {patient_id} ({patient_val})')

            # Add statistics text
            mean_val = df[feature].mean()
            ax.axvline(mean_val, color='blue', linewidth=2, linestyle='-', 
                     label=f'Mean ({mean_val:.2f})')

        else:
            # Create bar plot for categorical data
            value_counts = df[feature].value_counts()
            bars = ax.bar(value_counts.index, value_counts.values,
                          color='#4682B4', alpha=0.5, edgecolor='black', linewidth=0.5)
            ax.set_title(f"Distribution of {feature}\nPatient {patient_id} value: {patient_val}", 
                    fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel(feature, fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.tick_params(axis='x', rotation=45)

            # Highlight patient's value
            if patient_val in value_counts.index:
                patient_idx = list(value_counts.index).index(patient_val)
                bars[patient_idx].set_color('red')
                bars[patient_idx].set_alpha(1.0)
                bars[patient_idx].set_edgecolor('black')
                bars[patient_idx].set_linewidth(2)

        ax.grid(True, alpha=0.3)
        ax.legend()
        return fig
