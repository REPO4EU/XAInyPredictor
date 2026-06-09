import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui


@module.ui
def data_exploration_ui(config=None):
    if config is None:
        config = {}
    labels = config.get("labels", {})
    entity = labels.get("entity", {})
    text = labels.get("text", {})
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
            ui.output_ui("context_selection_hint"),
            class_="context-sidebar",
        ),
        ui.page_fluid(
            ui.layout_columns(
                ui.card(
                    ui.card_header(ui.output_ui("context_card_header")),
                    ui.output_ui("context_plot_context"),
                    ui.output_plot("plot_feature_distribution"),
                    full_screen=True,
                    class_="cohort-context-plot-card",
                ),
                ui.card(
                    ui.card_header("Feature summary"),
                    ui.output_ui("feature_stats_ui"),
                    full_screen=True,
                    class_="feature-summary-card",
                ),
                col_widths=(9, 3),
            ),
            class_="cohort-context-page",
        ),
    )


@module.server
def server(input: Inputs, output: Outputs, session: Session, global_input_data, model_data, patient_selected_id, config_init=None, config_reactive=None):

    display_to_name = reactive.Value({})

    def _text_labels():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        labels = current_cfg.get("labels", {})
        entity = labels.get("entity", {})
        return {
            "singular": entity.get("singular", "patient"),
            "singular_title": entity.get("singular_title", "Patient"),
            "plural": entity.get("plural", "patients"),
            "set_lower": entity.get("set_lower", "cohort"),
            "set_title": entity.get("set_title", "Cohort"),
            "select_label": labels.get("text", {}).get("select_entity_label", "Select patient:"),
            "search_placeholder": labels.get("text", {}).get("search_entity_placeholder", "Search by patient ID..."),
            "context_selection_hint": labels.get("text", {}).get(
                "context_selection_hint",
                f"Highlighted {entity.get('singular', 'patient')} updates the plot.",
            ),
            "context_card_title": labels.get("text", {}).get(
                "context_card_title",
                f"Feature distribution in current {entity.get('set_lower', 'cohort')}",
            ),
            "context_card_tooltip": labels.get("text", {}).get(
                "context_card_tooltip",
                f"Blue bars: Reference population distribution. Red dashed line: Selected {entity.get('singular', 'patient')}'s value.",
            ),
        }

    def _feature_label(feature: dict) -> str:
        return feature.get("display_name") or feature.get("label") or feature.get("name", "Feature")

    def _is_plot_feature(feature: dict) -> bool:
        if feature.get("plot") is False:
            return False
        if feature.get("role") == "identifier":
            return False
        return True

    def _normalize_columns(df):
        if df is None:
            return None
        df = df.copy()
        df.columns = [col.replace('_', ' ') for col in df.columns]
        return df

    def _selected_feature_name(df=None):
        display_name = str(input.feature_to_plot())
        col_name = display_to_name.get().get(display_name, display_name)
        if df is not None and col_name not in df.columns:
            underscored = col_name.replace(' ', '_')
            for column in df.columns:
                if column.replace(' ', '_') == underscored:
                    return column
        return col_name

    def _selected_feature_label():
        return str(input.feature_to_plot())

    def _reference_dataframe():
        reference_data = input.reference_data()
        if reference_data == 'Input data':
            df = global_input_data.get()
        else:
            md = model_data.get()
            df = md.get("X_TRAIN_RAW") if md else None
        return _normalize_columns(df)

    def _selected_patient_row():
        df = _normalize_columns(global_input_data.get())
        patient_id = patient_selected_id.get()
        if df is None or df.empty or 'ID' not in df.columns or patient_id is None:
            return None
        row = df[df['ID'] == patient_id]
        return None if row.empty else row

    def _format_value(value):
        if pd.isna(value):
            return "Missing"
        if isinstance(value, (int, float, np.integer, np.floating)):
            if float(value).is_integer():
                return str(int(value))
            return f"{float(value):.2f}"
        return str(value)

    @reactive.Effect
    def _rebuild_feature_list():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        features = current_cfg.get("features", []) if current_cfg else []
        features = [feature for feature in features if _is_plot_feature(feature)]
        display_to_name.set({_feature_label(f): f["name"] for f in features})

    @output
    @render.ui
    def feature_dropdown():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        features = current_cfg.get("features", []) if current_cfg else []
        features = [feature for feature in features if _is_plot_feature(feature)]
        choices = [(_feature_label(f), _feature_label(f)) for f in features]
        if not choices:
            return ui.p("No features configured", style="font-size: 0.85em; color: gray;")
        return ui.input_select(
            id="feature_to_plot",
            label="Feature to plot:",
            choices=dict(choices),
            selected=_feature_label(features[0]) if features else None,
        )

    @output
    @render.ui
    def context_selection_hint():
        return ui.p(_text_labels()["context_selection_hint"], class_="sidebar-help-text")

    @output
    @render.ui
    def context_card_header():
        labels = _text_labels()
        return ui.div(
            labels["context_card_title"],
            ui.tooltip(
                ui.span(ui.tags.i(class_="glyphicon glyphicon-question-sign"), class_="card-help-icon"),
                labels["context_card_tooltip"],
            ),
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
        return ui.input_selectize(
            id="local_patient_select",
            label=_text_labels()["select_label"],
            choices=all_ids,
            selected=selected_val,
            options={
                "create": False,  # Don't allow creating new options
                "allowEmptyOption": False,
                "placeholder": _text_labels()["search_placeholder"],
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
        df = _reference_dataframe()

        if df is None or df.empty:
            return ui.div("No data available", class_="feature-summary-empty")

        feature = _selected_feature_name(df)

        if feature not in df.columns:
            return ui.div("Feature not found", class_="feature-summary-empty")
        
        feature_data = df[feature].dropna()
        missing = int(df[feature].isnull().sum())

        def metric(label, value, class_name=""):
            return ui.div(
                ui.div(label, class_="feature-metric-label"),
                ui.div(value, class_="feature-metric-value"),
                class_=f"feature-metric {class_name}".strip(),
            )
        
        if pd.api.types.is_numeric_dtype(feature_data):
            if feature_data.empty:
                return ui.div("No non-missing values", class_="feature-summary-empty")

            return ui.div(
                metric("Count", str(len(feature_data)), "feature-metric-primary"),
                metric("Missing", str(missing)),
                metric("Mean", f"{feature_data.mean():.2f}"),
                metric("Median", f"{feature_data.median():.2f}"),
                metric("Range", f"{feature_data.min():.2f} - {feature_data.max():.2f}"),
                metric("Std dev", f"{feature_data.std():.2f}" if len(feature_data) > 1 else "N/A"),
                class_="feature-metric-grid",
            )
        else:
            unique_count = feature_data.nunique()
            mode = feature_data.mode().iloc[0] if not feature_data.mode().empty else "N/A"
            return ui.div(
                metric("Count", str(len(feature_data)), "feature-metric-primary"),
                metric("Missing", str(missing)),
                metric("Unique", str(unique_count)),
                metric("Most common", str(mode), "feature-metric-wide"),
                class_="feature-metric-grid",
            )

    @output
    @render.ui
    def context_plot_context():
        df = _reference_dataframe()
        patient_row = _selected_patient_row()
        labels = _text_labels()

        if df is None or df.empty:
            return None

        feature = _selected_feature_name(df)
        feature_label = _selected_feature_label()
        patient_id = patient_selected_id.get()
        patient_value = None
        if patient_row is not None and feature in patient_row.columns:
            patient_value = patient_row[feature].values[0]

        chips = [
            ui.span(f"Reference: {input.reference_data()}", class_="context-chip"),
            ui.span(f"n={len(df)}", class_="context-chip"),
        ]
        if patient_id is not None:
            chips.append(ui.span(f"Highlighted {labels['singular']}: {patient_id}", class_="context-chip context-chip-highlight"))
        if patient_value is not None:
            chips.append(ui.span(f"Value: {_format_value(patient_value)}", class_="context-chip context-chip-highlight"))

        notices = []
        if len(df) < 10:
            notices.append(
                ui.div(
                    f"Small {labels['set_lower']}: distribution is based on {len(df)} records.",
                    class_="context-small-sample context-small-sample-info",
                )
            )

        return ui.div(ui.div(*chips, class_="context-chip-row"), *notices, class_="context-plot-context")

    @output
    @render.plot
    def plot_feature_distribution():
        """
        Plot distribution of a feature within the input dataset and highlight
        the value from a specific patient.
        """
        input_df = _normalize_columns(global_input_data.get())
        df = _reference_dataframe()
        if input_df is None or input_df.empty or df is None or df.empty:
            fig, ax = plt.subplots()
            ax.set_axis_off()
            ax.text(0.5, 0.5, "No data available", ha="center", va="center")
            return fig

        feature = _selected_feature_name(df)
        feature_label = _selected_feature_label()
        patient_id = patient_selected_id.get()

        if feature not in df.columns:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"Feature '{feature}' not found", ha="center", va="center", fontsize=16)
            return fig

        # Filter for specific patient
        patient_row = input_df[input_df['ID'] == patient_id]
        if patient_row.empty:
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.set_axis_off()
            ax.text(0.5, 0.5, f"{_text_labels()['singular_title']} {patient_id} not found in data!", ha="center", va="center", fontsize=16)
            return fig
        patient_val = patient_row[feature].values[0]
        plot_series = df[feature].dropna()

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 5.6))

        # Handle categorical vs numerical
        if pd.api.types.is_numeric_dtype(plot_series):
            numeric_values = pd.to_numeric(plot_series, errors="coerce").dropna()
            if len(numeric_values) < 10:
                y = [1] * len(numeric_values)
                ax.scatter(numeric_values, y, color='#4682B4', alpha=0.75, s=80, edgecolor='black', linewidth=0.4, label="Reference records")
                ax.set_yticks([])
                ax.set_ylabel("")
            else:
                bins = min(30, max(6, int(len(numeric_values) ** 0.5)))
                ax.hist(numeric_values, bins=bins, color='#4682B4', alpha=0.5, edgecolor='black', linewidth=0.5)
                ax.set_ylabel('Frequency', fontsize=12)

            ax.set_title(f"Distribution of {feature_label}", fontsize=14, fontweight='bold', pad=16)
            ax.set_xlabel(feature_label, fontsize=12)

            # Add vertical line for patient value
            ax.axvline(patient_val, color='red', linewidth=3, linestyle='--',
                       label=f"{_text_labels()['singular_title']} {patient_id} ({_format_value(patient_val)})")

            # Add statistics text
            mean_val = numeric_values.mean()
            ax.axvline(mean_val, color='blue', linewidth=2, linestyle='-', 
                     label=f'Mean ({mean_val:.2f})')

        else:
            value_counts = plot_series.astype(str).value_counts()
            unique_ratio = value_counts.size / max(len(plot_series), 1)
            if value_counts.size == 1:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.58,
                    f"All records share the same {feature_label} value",
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                )
                ax.text(
                    0.5,
                    0.43,
                    f"Value: {_format_value(patient_val)}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#405261",
                )
                return fig

            if value_counts.size > 12 or unique_ratio > 0.8:
                ax.set_axis_off()
                ax.text(
                    0.5,
                    0.58,
                    f"{feature_label} has mostly unique values",
                    ha="center",
                    va="center",
                    fontsize=15,
                    fontweight="bold",
                )
                ax.text(
                    0.5,
                    0.43,
                    f"Selected {_text_labels()['singular']} value: {_format_value(patient_val)}",
                    ha="center",
                    va="center",
                    fontsize=12,
                    color="#405261",
                )
                return fig

            patient_val_str = str(patient_val)
            bar_colors = ['#9fbed6'] * len(value_counts)
            edge_colors = ['#6889a5'] * len(value_counts)
            line_widths = [0.8] * len(value_counts)
            if patient_val_str in value_counts.index:
                patient_idx = list(value_counts.index).index(patient_val_str)
                bar_colors[patient_idx] = '#bfd7ec'
                edge_colors[patient_idx] = '#d62728'
                line_widths[patient_idx] = 2.5

            ax.barh(
                value_counts.index,
                value_counts.values,
                color=bar_colors,
                alpha=0.95,
                edgecolor=edge_colors,
                linewidth=line_widths,
            )
            ax.set_title(f"Distribution of {feature_label}", fontsize=14, fontweight='bold', pad=16)
            ax.set_xlabel('Count', fontsize=12)
            ax.set_ylabel(feature_label, fontsize=12)
            ax.invert_yaxis()

            if patient_val_str in value_counts.index:
                ax.barh([], [], color='#bfd7ec', edgecolor='#d62728', linewidth=2.5, label=f"Selected value: {_format_value(patient_val)}")

        ax.grid(True, alpha=0.3)
        handles, _ = ax.get_legend_handles_labels()
        if handles:
            ax.legend()
        fig.tight_layout()
        return fig
