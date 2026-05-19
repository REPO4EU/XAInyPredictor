import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from XAInyPredictor.modules.data_processing import clean_data


def build_form_fields(config: dict) -> tuple:
    """Build form fields dynamically from config.
    Returns (columns, labels_dict) where columns is a list of ui.div for layout,
    and labels_dict maps input_id to display_label for server-side access.
    """
    features = config.get("features", [])
    labels = config.get("labels", {})

    if not features:
        return None, {}

    columns = []
    labels_dict = {}
    current_column = []
    column_count = 0

    for idx, feature in enumerate(features):
        feat_name = feature.get("name", "")
        display_name = feature.get("display_name", feat_name)
        input_type = feature.get("input_type", "numeric")
        values = feature.get("values", [])
        display_values = feature.get("display_values", {})
        default = feature.get("default", "")
        help_text = feature.get("help_text", "")
        min_val = feature.get("min", 0)
        max_val = feature.get("max", 100)
        step = feature.get("step", 1)

        input_id = f"in_{feat_name.replace(' ', '_').replace('(', '').replace(')', '')}"
        labels_dict[input_id] = feat_name

        field_ui = None

        if input_type == "numeric":
            field_ui = ui.input_numeric(
                input_id,
                display_name,
                value=default if default else min_val,
                min=min_val,
                max=max_val,
                step=step
            )
        elif input_type == "select" and values:
            choices = {}
            for v in values:
                display = display_values.get(v, v)
                choices[v] = display
            field_ui = ui.input_select(input_id, display_name, choices=choices, selected=default)

        if help_text and field_ui:
            field_ui = ui.div(
                field_ui,
                ui.tooltip(
                    ui.span(ui.tags.i(class_="glyphicon glyphicon-question-sign"), style="cursor: pointer; color: #007bc2; margin-left: 5px;"),
                    help_text,
                    placement="right"
                ),
                style="display: flex; align-items: center;"
            )

        current_column.append(ui.div(field_ui, style="margin-bottom: 10px;"))

        if (idx + 1) % 4 == 0 or idx == len(features) - 1:
            if current_column:
                col_ui = ui.div(
                    *current_column,
                    style="padding: 10px;"
                )
                columns.append(col_ui)
                current_column = []
                column_count += 1

    return columns, labels_dict


@module.ui
def data_input_ui(config=None):
    if config is None:
        config = {}

    labels = config.get("labels", {})
    titles = config.get("titles", {})

    form_columns, labels_dict = build_form_fields(config)

    return ui.layout_sidebar(
        ui.sidebar(
            ui.h4(labels.get("data_source", "Data Source")),
            ui.input_radio_buttons(
                "input_method",
                labels.get("select_method", "Select Method:"),
                {"form": labels.get("manual_entry", "Manual Entry"), "file": labels.get("upload_file", "Upload File"), "example": labels.get("example_cohort", "Example Cohort")}
            ),
            ui.panel_conditional(
                "input.input_method == 'file'",
                ui.input_file(
                    "input_dataset_file",
                    "Upload File (TSV, CSV, Excel)",
                    accept=[".tsv", ".csv", ".xlsx"],
                    multiple=False,
                ),
                ui.p("Ensure columns match the template.", style="font-size: 0.8em; color: gray;"),
            ),
        ),

        ui.panel_conditional(
            "input.input_method == 'form'",
            ui.card(
                ui.card_header(ui.tags.b("➕ " + (labels.get("form", {}).get("add_patient_button", "New Patient Entry")))),
                ui.output_ui("form_fields"),
                ui.card_footer(
                    ui.input_action_button("btn_add_form", labels.get("form", {}).get("add_patient_button", "Add Patient to Cohort"), class_="btn-primary", width="100%")
                )
            ),
            ui.br()
        ),

        ui.card(
            ui.card_header(ui.tags.b(labels.get("form", {}).get("cohort_title", "Current Patient Cohort"))),
            ui.output_data_frame("out_patient_table"),
            ui.panel_conditional(
                "input.input_method == 'form'",
                ui.input_action_button("btn_delete_selected", labels.get("form", {}).get("delete_selected_button", "Delete Selected"), class_="btn-danger btn-sm", width="100%")
            ),
            full_screen=True
        )
    )


@module.server
def server(input: Inputs, output: Outputs, session: Session, model_data, config_init, config_reactive=None):
    labels_dict = reactive.Value({})
    feature_cols = reactive.Value([])
    form_df = reactive.Value(pd.DataFrame())
    id_counter = reactive.Value(1)

    @reactive.Effect
    def _rebuild_fields():
        current_cfg = config_reactive.get() if config_reactive else config_init
        features = current_cfg.get("features", []) if current_cfg else []

        new_labels = {}
        new_cols = []
        if features:
            new_cols = [f["name"] for f in features]
            for f in features:
                feat_name = f.get("name", "")
                input_id = f"in_{feat_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                new_labels[input_id] = feat_name

        labels_dict.set(new_labels)
        feature_cols.set(new_cols)
        form_df.set(pd.DataFrame(columns=['ID'] + new_cols))

        # Force the radio button back to form on use-case change
        ui.update_radio_buttons("input_method", selected="form")

    output_data = reactive.Value(None)
    output_is_custom = reactive.Value(False)

    def _ensure_id(df):
        if df is None: return None
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))
        else:
            cols = ['ID'] + [c for c in df.columns if c != 'ID']
            df = df[cols]
        return df

    @reactive.Effect
    @reactive.event(input.btn_add_form)
    def _add_patient():
        current_df = form_df.get()
        new_id = id_counter()
        current_labels = labels_dict.get()

        row_data = {'ID': new_id}
        for input_id, feat_name in current_labels.items():
            val = getattr(input, input_id, None)()
            if val is not None:
                if isinstance(val, str):
                    row_data[feat_name] = val
                else:
                    row_data[feat_name] = float(val)

        new_row = pd.DataFrame([row_data])
        updated_df = pd.concat([current_df, new_row], ignore_index=True)
        form_df.set(updated_df)
        id_counter.set(new_id + 1)
        ui.notification_show("Patient added.", type="message")

    @reactive.Effect
    @reactive.event(input.btn_delete_selected)
    def _delete_patient():
        selected_rows = input.out_patient_table_selected_rows()
        if not selected_rows:
            ui.notification_show("No rows selected.", type="warning")
            return

        current_df = form_df.get()
        updated_df = current_df.drop(index=list(selected_rows)).reset_index(drop=True)
        form_df.set(updated_df)
        ui.notification_show(f"Deleted {len(selected_rows)} patient(s).", type="message")

    @reactive.Effect
    def _update_output_pipeline():
        method = str(input.input_method())
        raw_df = None
        is_custom = False

        md = model_data.get()
        example_raw_data = md.get("X_TEST_RAW")

        if method == "example":
            if example_raw_data is not None:
                raw_df = example_raw_data.copy()
            is_custom = False

        elif method == "form":
            raw_df = form_df.get()
            is_custom = True

        elif method == "file":
            file_info = input.input_dataset_file()
            is_custom = True
            if not file_info:
                output_data.set(None)
                output_is_custom.set(is_custom)
                return

            file_path = file_info[0]["datapath"]
            try:
                if file_path.endswith(".csv"):
                    df = pd.read_csv(file_path)
                elif file_path.endswith(".tsv"):
                    df = pd.read_csv(file_path, sep="\t")
                elif file_path.endswith(".xlsx"):
                    df = pd.read_excel(file_path)
                else:
                    raise ValueError("Unsupported file format!")
                raw_df = clean_data(df)
            except Exception as e:
                ui.notification_show(f"Error reading file: {e}", type="error")
                raw_df = None

        if raw_df is not None and not raw_df.empty:
            clean_df = _ensure_id(raw_df)
            output_data.set(clean_df)
        else:
            output_data.set(None)

        output_is_custom.set(is_custom)

    @output
    @render.ui
    def form_fields():
        current_cfg = config_reactive.get() if config_reactive else config_init
        form_columns, _ = build_form_fields(current_cfg or {})
        if form_columns:
            return ui.layout_columns(
                *form_columns, col_widths=tuple([4] * len(form_columns))
            )
        return ui.p("No features configured.")

    @output
    @render.data_frame
    def out_patient_table():
        display_df = output_data.get()

        if display_df is None or display_df.empty:
            return render.DataGrid(
                pd.DataFrame({"Message": ["No data uploaded."]}),
                width="100%",
                selection_mode="none"
            )

        display_df = display_df.copy()
        display_df.columns = [c.replace('_', ' ') for c in display_df.columns]
        display_df = _ensure_id(display_df)

        return render.DataGrid(
            display_df,
            width="100%",
            selection_mode="rows"
        )

    return {"data": output_data, "is_custom": output_is_custom}