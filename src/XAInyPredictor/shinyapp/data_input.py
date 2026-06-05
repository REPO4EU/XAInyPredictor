import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from XAInyPredictor.modules.data_processing import clean_data


def _decimal_text(value):
    if value in (None, ""):
        return ""
    return str(value).replace(",", ".")


def build_form_fields(config: dict) -> tuple:
    """Build form fields dynamically from config.
    Returns (columns, labels_dict) where columns is a list of ui.div for layout,
    and labels_dict maps input_id to display_label for server-side access.
    """
    features = config.get("features", [])
    labels = config.get("labels", {})

    if not features:
        return None, {}

    sections = {}
    section_order = []
    labels_dict = {}

    def _section_for_feature(feature: dict) -> str:
        name = f"{feature.get('name', '')} {feature.get('display_name', '')} {feature.get('label', '')}".lower()
        if any(token in name for token in ["peptide", "hla", "allele", "pseudo"]):
            return "Candidate identity"
        if any(token in name for token in ["molecular", "gravy", "instability", "isoelectric", "half", "charge"]):
            return "Molecular properties"
        if any(token in name for token in ["mhcflurry", "score", "affinity", "presentation", "processing"]):
            return "Presentation features"
        if any(token in name for token in ["age", "gender", "bmi"]):
            return "Patient profile"
        if any(token in name for token in ["tumor", "histology", "node", "metastases", "extension", "multifocality", "vascular", "resection"]):
            return "Tumor characteristics"
        if any(token in name for token in ["ata", "rai", "treatment"]):
            return "Treatment context"
        return "Input variables"

    for idx, feature in enumerate(features):
        feat_name = feature.get("name", "")
        display_name = feature.get("display_name") or feature.get("label") or feat_name
        input_type = feature.get("input_type") or feature.get("type", "numeric")
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
            default_value = default if default not in (None, "") else min_val
            field_ui = ui.div(
                ui.input_text(
                    input_id,
                    display_name,
                    value=_decimal_text(default_value),
                    placeholder=_decimal_text(default_value),
                    autocomplete="off",
                ),
                class_="manual-decimal-input",
            )
        elif input_type == "text":
            field_ui = ui.input_text(input_id, display_name, value=str(default or ""))
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
                    ui.span(ui.tags.i(class_="glyphicon glyphicon-question-sign"), class_="form-help-icon"),
                    help_text,
                    placement="right"
                ),
                class_="form-field-with-help",
            )

        section_name = _section_for_feature(feature)
        if section_name not in sections:
            sections[section_name] = []
            section_order.append(section_name)
        sections[section_name].append(ui.div(field_ui, class_="manual-entry-field"))

    section_cards = [
        ui.div(
            ui.div(section_name, class_="manual-entry-section-title"),
            ui.div(*sections[section_name], class_="manual-entry-section-fields"),
            class_="manual-entry-section",
        )
        for section_name in section_order
    ]

    return section_cards, labels_dict


@module.ui
def data_input_ui(config=None):
    if config is None:
        config = {}

    labels = config.get("labels", {})
    return ui.page_fluid(
        ui.div(
            ui.card(
                ui.card_header("Selected Use Case"),
                ui.output_ui("use_case_summary"),
                class_="use-case-summary-card",
            ),
            ui.card(
                ui.card_header(ui.output_ui("cohort_table_header")),
                ui.output_ui("current_set_body"),
                ui.output_ui("input_validation_ui"),
                ui.output_ui("input_validation_success_ui"),
                ui.card_footer(
                    ui.div(
                        ui.output_ui("apply_data_ui"),
                        ui.div(
                            ui.output_ui("delete_selected_action"),
                            ui.output_ui("reset_cohort_action"),
                            class_="cohort-secondary-actions",
                        ),
                        class_="cohort-actions",
                    )
                ),
                class_="current-set-card",
                full_screen=True,
            ),
            ui.div(
                ui.output_ui("input_method_header"),
                ui.navset_tab(
                    ui.nav_panel(
                        labels.get("manual_entry", "Manual Entry"),
                        ui.card(
                            ui.card_header(ui.output_ui("manual_entry_header")),
                            ui.output_ui("form_fields"),
                            ui.card_footer(ui.output_ui("manual_entry_action")),
                        ),
                        value="form",
                    ),
                    ui.nav_panel(
                        labels.get("upload_file", "Upload File"),
                        ui.card(
                            ui.card_header(ui.output_ui("upload_entry_header")),
                            ui.div(
                                ui.div(
                                    ui.input_file(
                                        "input_dataset_file",
                                        "Upload File (TSV, CSV, Excel)",
                                        accept=[".tsv", ".csv", ".xlsx"],
                                        multiple=False,
                                    ),
                                    ui.p("CSV, TSV, or Excel file with the current use case columns.", class_="upload-template-hint"),
                                    class_="upload-file-picker",
                                ),
                                ui.div(
                                    ui.download_button(
                                        "download_input_template",
                                        "Template",
                                        class_="btn-default btn-sm input-template-download",
                                    ),
                                    ui.download_button(
                                        "download_data_dictionary",
                                        "Data dictionary",
                                        class_="btn-default btn-sm input-template-download",
                                    ),
                                    class_="upload-resource-actions",
                                ),
                                class_="upload-file-panel",
                            ),
                            ui.div(
                                ui.output_ui("upload_file_requirements"),
                                ui.input_action_button("btn_load_file", "Load file into current set", class_="btn-primary"),
                                class_="upload-file-actions",
                            ),
                        ),
                        value="file",
                    ),
                    ui.nav_panel(
                        labels.get("example_cohort", "Example Cohort"),
                        ui.card(
                            ui.card_header(ui.output_ui("example_entry_header")),
                            ui.div(
                                ui.div(
                                    ui.div("Configured sample dataset", class_="example-set-title"),
                                    ui.p(
                                        "Load a ready-to-use dataset for the selected use case.",
                                        class_="upload-template-hint",
                                    ),
                                    class_="example-set-copy",
                                ),
                                ui.output_ui("example_set_summary"),
                                class_="example-set-panel",
                            ),
                            ui.div(
                                ui.input_action_button("btn_load_example", "Load example into current set", class_="btn-primary"),
                                class_="example-set-actions",
                            ),
                        ),
                        value="example",
                    ),
                    id="input_method",
                    selected="form",
                ),
                class_="data-input-method-tabs",
            ),
            class_="data-input-page",
        ),
    )


@module.server
def server(input: Inputs, output: Outputs, session: Session, model_data, config_init, config_reactive=None):
    labels_dict = reactive.Value({})
    input_types = reactive.Value({})
    feature_cols = reactive.Value([])
    form_df = reactive.Value(pd.DataFrame())
    id_counter = reactive.Value(1)
    validation_errors = reactive.Value([])
    validation_success = reactive.Value(None)
    pending_data = reactive.Value(None)
    pending_is_custom = reactive.Value(False)
    pending_signature = reactive.Value(None)
    confirmed_signature = reactive.Value(None)
    confirmed_ready = reactive.Value(False)
    output_data = reactive.Value(None)
    output_is_custom = reactive.Value(False)
    working_source = reactive.Value("form")

    def _current_text_labels():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        labels = current_cfg.get("labels", {})
        entity = labels.get("entity", {})
        form_labels = labels.get("form", {})
        return {
            "singular": entity.get("singular", "patient"),
            "plural": entity.get("plural", "patients"),
            "set_lower": entity.get("set_lower", "cohort"),
            "set_title": entity.get("set_title", "Cohort"),
            "manual_source": form_labels.get("manual_source", "manual cohort"),
            "uploaded_source": form_labels.get("uploaded_source", "uploaded file"),
            "example_source": form_labels.get("example_source", "example cohort"),
            "added_message": form_labels.get("added_message", "Patient added."),
            "deleted_message": form_labels.get("deleted_message", "Deleted {count} patient(s)."),
            "reset_message": form_labels.get("reset_message", "Cohort reset."),
            "groups_label": labels.get("text", {}).get("groups_label", "Patient groups"),
            "add_button": form_labels.get("add_patient_button", "Add Patient to Cohort"),
            "delete_button": form_labels.get("delete_selected_button", "Delete Selected"),
            "reset_button": form_labels.get("reset_cohort_button", "Reset Cohort"),
            "cohort_title": form_labels.get(
                "cohort_title",
                f"Current {entity.get('set_title', 'Patient Cohort')}",
            ),
            "no_data_message": form_labels.get("no_data_message", "No data uploaded."),
        }

    def _input_method_choices():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        labels = current_cfg.get("labels", {})
        return {
            "form": labels.get("manual_entry", "Manual Entry"),
            "file": labels.get("upload_file", "Upload File"),
            "example": labels.get("example_cohort", "Example Cohort"),
        }

    @reactive.Effect
    def _rebuild_fields():
        current_cfg = config_reactive.get() if config_reactive else config_init
        features = current_cfg.get("features", []) if current_cfg else []

        new_labels = {}
        new_input_types = {}
        new_cols = []
        if features:
            new_cols = [f["name"] for f in features]
            for f in features:
                feat_name = f.get("name", "")
                input_id = f"in_{feat_name.replace(' ', '_').replace('(', '').replace(')', '')}"
                new_labels[input_id] = feat_name
                new_input_types[feat_name] = f.get("input_type") or f.get("type", "numeric")

        labels_dict.set(new_labels)
        input_types.set(new_input_types)
        feature_cols.set(new_cols)
        form_df.set(pd.DataFrame(columns=['ID'] + new_cols))
        id_counter.set(1)
        validation_errors.set([])
        validation_success.set(None)
        pending_data.set(None)
        pending_is_custom.set(False)
        pending_signature.set(None)
        confirmed_signature.set(None)
        confirmed_ready.set(False)

        working_source.set("form")

        # Force the tabs back to manual entry on use-case change.
        ui.update_navset("input_method", selected="form")

    def _ensure_id(df):
        if df is None: return None
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))
        else:
            cols = ['ID'] + [c for c in df.columns if c != 'ID']
            df = df[cols]
        return df

    def _data_signature(df, is_custom):
        if df is None or df.empty:
            return None
        normalized = df.copy().reset_index(drop=True)
        normalized = normalized.reindex(sorted(normalized.columns), axis=1)
        row_hash = pd.util.hash_pandas_object(normalized.astype(str), index=False).sum()
        return f"{bool(is_custom)}:{len(normalized)}:{list(normalized.columns)}:{int(row_hash)}"

    def _format_values(values, max_items=5):
        clean_values = [str(v) for v in values if pd.notna(v)]
        preview = clean_values[:max_items]
        suffix = "" if len(clean_values) <= max_items else f" and {len(clean_values) - max_items} more"
        return ", ".join(preview) + suffix

    def _template_values(feature, n_rows=5):
        if "default" in feature:
            default = feature.get("default")
        else:
            default = ""

        if feature.get("input_type") == "select":
            values = feature.get("values", [])
            if not values:
                return [default] * n_rows
            start_idx = values.index(default) if default in values else 0
            return [values[(start_idx + i) % len(values)] for i in range(n_rows)]

        min_val = feature.get("min")
        max_val = feature.get("max")
        step = feature.get("step", 1)

        if min_val is None or max_val is None:
            return [default] * n_rows

        min_float = float(_decimal_text(min_val))
        max_float = float(_decimal_text(max_val))
        step_float = float(_decimal_text(step))
        base_value = float(_decimal_text(default)) if default != "" else (min_float + max_float) / 2
        midpoint = (min_float + max_float) / 2
        numeric_values = [
            base_value,
            min_val,
            max_val,
            midpoint,
            min(max(base_value + step_float, min_float), max_float),
        ]

        return [round(value, 2) if isinstance(value, float) and not value.is_integer() else int(value) if isinstance(value, float) else value for value in numeric_values[:n_rows]]

    def _template_dataframe(config):
        features = config.get("features", []) if config else []
        n_rows = 5
        rows = [{"ID": idx + 1} for idx in range(n_rows)]
        for feature in features:
            feat_name = feature.get("name")
            if feat_name:
                values = _template_values(feature, n_rows)
                for idx, row in enumerate(rows):
                    row[feat_name] = values[idx]
        return pd.DataFrame(rows)

    def _data_dictionary_dataframe(config):
        rows = []
        features = config.get("features", []) if config else []
        for feature in features:
            allowed_values = feature.get("values", [])
            rows.append(
                {
                    "Variable": feature.get("name", ""),
                    "Display Name": feature.get("display_name") or feature.get("label") or feature.get("name", ""),
                    "Type": feature.get("type", ""),
                    "Input Type": feature.get("input_type", ""),
                    "Required": "YES",
                    "Allowed Values": ", ".join(map(str, allowed_values)),
                    "Min": feature.get("min", ""),
                    "Max": feature.get("max", ""),
                    "Default": feature.get("default", ""),
                    "Description": feature.get("help_text", ""),
                }
            )
        return pd.DataFrame(rows)

    def _validation_success_message(df, config, method):
        features = config.get("features", []) if config else []
        required_cols = [feature.get("name") for feature in features if feature.get("name")]
        matched_cols = [col for col in required_cols if col in df.columns]
        text_labels = _current_text_labels()
        source = {
            "form": text_labels["manual_source"],
            "file": text_labels["uploaded_source"],
            "example": text_labels["example_source"],
        }.get(method, text_labels["set_lower"])
        return f"{len(df)} {text_labels['singular']}(s) loaded from {source}. {len(matched_cols)}/{len(required_cols)} required columns matched. No validation issues found."

    def _validate_input_data(df, config):
        if df is None or df.empty or not config:
            return []

        errors = []
        features = config.get("features", [])
        required_cols = [feature.get("name") for feature in features if feature.get("name")]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            errors.append(f"Missing required column(s): {', '.join(missing_cols)}.")

        for feature in features:
            feat_name = feature.get("name")
            if not feat_name or feat_name not in df.columns:
                continue

            display_name = feature.get("display_name") or feature.get("label") or feat_name
            input_type = feature.get("input_type") or feature.get("type", "numeric")
            series = df[feat_name]
            non_missing = series[series.notna()]

            if non_missing.empty:
                continue

            if input_type == "numeric":
                numeric_series = pd.to_numeric(non_missing.astype(str).str.replace(",", ".", regex=False), errors="coerce")
                invalid_values = non_missing[numeric_series.isna()].drop_duplicates().tolist()
                if invalid_values:
                    errors.append(f"{display_name} must be numeric. Invalid value(s): {_format_values(invalid_values)}.")
                    continue

                min_val = feature.get("min")
                max_val = feature.get("max")
                range_mask = pd.Series(False, index=numeric_series.index)
                if min_val is not None:
                    range_mask = range_mask | (numeric_series < min_val)
                if max_val is not None:
                    range_mask = range_mask | (numeric_series > max_val)

                if range_mask.any():
                    bad_values = non_missing[range_mask].drop_duplicates().tolist()
                    if min_val is not None and max_val is not None:
                        range_text = f"between {min_val} and {max_val}"
                    elif min_val is not None:
                        range_text = f"greater than or equal to {min_val}"
                    else:
                        range_text = f"less than or equal to {max_val}"
                    errors.append(f"{display_name} must be {range_text}. Out-of-range value(s): {_format_values(bad_values)}.")

            elif input_type == "select":
                allowed_values = feature.get("values", [])
                if not allowed_values:
                    continue

                invalid_values = sorted(set(non_missing.astype(str)) - set(map(str, allowed_values)))
                if invalid_values:
                    errors.append(
                        f"{display_name} has unexpected categor{('y' if len(invalid_values) == 1 else 'ies')}: "
                        f"{_format_values(invalid_values)}. Allowed: {', '.join(map(str, allowed_values))}."
                    )
            elif input_type == "text":
                continue

        return errors

    @reactive.Effect
    @reactive.event(input.btn_add_form)
    def _add_patient():
        current_df = form_df.get()
        new_id = id_counter()
        current_labels = labels_dict.get()
        current_input_types = input_types.get()

        row_data = {'ID': new_id}
        for input_id, feat_name in current_labels.items():
            val = getattr(input, input_id, None)()
            if val is not None:
                if current_input_types.get(feat_name) == "numeric":
                    row_data[feat_name] = float(_decimal_text(val))
                elif isinstance(val, str):
                    row_data[feat_name] = val
                else:
                    row_data[feat_name] = float(val)

        new_row = pd.DataFrame([row_data])
        updated_df = pd.concat([current_df, new_row], ignore_index=True)
        form_df.set(updated_df)
        working_source.set("form")
        id_counter.set(new_id + 1)
        ui.notification_show(_current_text_labels()["added_message"], type="message")

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
        working_source.set("form")
        ui.notification_show(_current_text_labels()["deleted_message"].format(count=len(selected_rows)), type="message")

    @reactive.Effect
    @reactive.event(input.btn_reset_cohort)
    def _reset_cohort():
        current_cols = feature_cols.get()
        form_df.set(pd.DataFrame(columns=['ID'] + current_cols))
        id_counter.set(1)
        validation_errors.set([])
        validation_success.set(None)
        pending_data.set(None)
        pending_is_custom.set(False)
        pending_signature.set(None)
        confirmed_signature.set(None)
        confirmed_ready.set(False)
        output_data.set(None)
        output_is_custom.set(False)
        working_source.set("form")
        ui.update_navset("input_method", selected="form")
        ui.notification_show(_current_text_labels()["reset_message"], type="message")

    @reactive.Effect
    @reactive.event(input.btn_load_file)
    def _load_file_into_set():
        file_info = input.input_dataset_file()
        if not file_info:
            ui.notification_show("Select a file before loading it into the current set.", type="warning")
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
            form_df.set(_ensure_id(clean_data(df)))
            working_source.set("file")
            ui.notification_show("File loaded into the current set.", type="message")
        except Exception as e:
            ui.notification_show(f"Error reading file: {e}", type="error")
            validation_errors.set([f"Could not read uploaded file: {e}"])
            validation_success.set(None)

    @reactive.Effect
    @reactive.event(input.btn_load_example)
    def _load_example_into_set():
        md = model_data.get()
        example_raw_data = md.get("X_TEST_RAW") if md else None
        if example_raw_data is None or example_raw_data.empty:
            ui.notification_show("No example data available for this use case.", type="warning")
            return

        form_df.set(_ensure_id(example_raw_data.copy()))
        working_source.set("example")
        ui.notification_show(f"Example {_current_text_labels()['set_lower']} loaded.", type="message")

    @reactive.Effect
    def _update_output_pipeline():
        method = working_source.get()
        raw_df = form_df.get()
        is_custom = method != "example"
        current_cfg = config_reactive.get() if config_reactive else config_init

        if raw_df is not None and not raw_df.empty:
            clean_df = _ensure_id(raw_df)
            errors = _validate_input_data(clean_df, current_cfg)
            validation_errors.set(errors)
            if errors:
                validation_success.set(None)
                pending_data.set(None)
                pending_is_custom.set(is_custom)
                pending_signature.set(None)
                confirmed_ready.set(False)
                output_data.set(None)
                output_is_custom.set(False)
                ui.notification_show("Input validation failed. See details in Data Input.", type="error")
                return
            validation_success.set(_validation_success_message(clean_df, current_cfg, method))
            pending_data.set(clean_df)
            pending_is_custom.set(is_custom)
            current_signature = _data_signature(clean_df, is_custom)
            pending_signature.set(current_signature)
            if current_signature != confirmed_signature.get():
                confirmed_ready.set(False)
                output_data.set(None)
                output_is_custom.set(False)
        else:
            validation_errors.set([])
            validation_success.set(None)
            pending_data.set(None)
            pending_is_custom.set(is_custom)
            pending_signature.set(None)
            confirmed_ready.set(False)
            output_data.set(None)
            output_is_custom.set(False)

    @reactive.Effect
    @reactive.event(input.btn_use_data)
    def _use_pending_data():
        clean_df = pending_data.get()
        if clean_df is None or clean_df.empty:
            ui.notification_show("No valid data available to use.", type="warning")
            return

        output_data.set(clean_df.copy())
        output_is_custom.set(bool(pending_is_custom.get()))
        confirmed_signature.set(pending_signature.get())
        confirmed_ready.set(True)
        ui.notification_show(f"{_current_text_labels()['set_title']} confirmed for analysis.", type="message")

    @output
    @render.ui
    def input_validation_ui():
        errors = validation_errors.get()
        if not errors:
            return None

        return ui.div(
            ui.div("Input validation failed", class_="input-validation-title"),
            ui.tags.ul(*[ui.tags.li(error) for error in errors]),
            class_="input-validation-alert",
        )

    @output
    @render.ui
    def input_validation_success_ui():
        message = validation_success.get()
        if not message:
            return None

        return ui.div(
            ui.div("Input validation passed", class_="input-validation-title"),
            ui.p(message, class_="input-validation-success-copy"),
            class_="input-validation-success",
        )

    @output
    @render.ui
    def apply_data_ui():
        clean_df = pending_data.get()
        if clean_df is None or clean_df.empty or validation_errors.get():
            return None

        labels = _current_text_labels()
        is_confirmed = confirmed_ready.get() and pending_signature.get() == confirmed_signature.get()
        if is_confirmed:
            return ui.div(
                ui.div(
                    ui.div(f"{labels['set_title']} confirmed", class_="confirm-data-title"),
                    ui.p(
                        f"{len(clean_df)} {labels['plural']} are locked for Candidate Context and Stratification Support.",
                        class_="apply-data-hint",
                    ),
                    class_="confirm-data-copy",
                ),
                class_="apply-data-panel apply-data-panel-confirmed",
            )

        return ui.div(
            ui.input_action_button(
                "btn_use_data",
                f"Confirm {labels['set_lower']}",
                class_="btn-primary",
            ),
            ui.p(
                "Candidate Context and Stratification Support will unlock after confirmation.",
                class_="apply-data-hint",
            ),
            class_="apply-data-panel",
        )

    @render.download(filename="input_template.csv")
    def download_input_template():
        current_cfg = config_reactive.get() if config_reactive else config_init
        yield _template_dataframe(current_cfg or {}).to_csv(index=False)

    @render.download(filename="data_dictionary.csv")
    def download_data_dictionary():
        current_cfg = config_reactive.get() if config_reactive else config_init
        yield _data_dictionary_dataframe(current_cfg or {}).to_csv(index=False)

    @output
    @render.ui
    def manual_entry_header():
        return ui.tags.b("+ " + _current_text_labels()["add_button"])

    @output
    @render.ui
    def upload_entry_header():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        labels = current_cfg.get("labels", {})
        return ui.tags.b(labels.get("upload_file", "Upload File"))

    @output
    @render.ui
    def upload_file_requirements():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        features = current_cfg.get("features", [])
        return ui.div(
            ui.span(str(len(features)), class_="upload-requirement-count"),
            ui.span(" required variables"),
            class_="upload-requirement-pill",
        )

    @output
    @render.ui
    def example_entry_header():
        current_cfg = config_reactive.get() if config_reactive else (config_init or {})
        labels = current_cfg.get("labels", {})
        return ui.tags.b(labels.get("example_cohort", "Example Cohort"))

    @output
    @render.ui
    def example_set_summary():
        md = model_data.get()
        example_raw_data = md.get("X_TEST_RAW") if md else None
        row_count = 0 if example_raw_data is None else len(example_raw_data)
        labels = _current_text_labels()
        return ui.div(
            ui.span(str(row_count), class_="upload-requirement-count"),
            ui.span(f" example {labels['plural']}"),
            class_="upload-requirement-pill",
        )

    @output
    @render.ui
    def manual_entry_action():
        return ui.input_action_button(
            "btn_add_form",
            _current_text_labels()["add_button"],
            class_="btn-primary manual-entry-submit",
        )

    @output
    @render.ui
    def delete_selected_action():
        df = pending_data.get()
        if df is None or df.empty:
            df = output_data.get()
        if df is None or df.empty:
            return None

        return ui.input_action_button(
            "btn_delete_selected",
            _current_text_labels()["delete_button"],
            class_="btn-danger btn-sm cohort-action-button",
        )

    @output
    @render.ui
    def reset_cohort_action():
        return ui.input_action_button(
            "btn_reset_cohort",
            _current_text_labels()["reset_button"],
            class_="btn-default btn-sm cohort-action-button",
        )

    @output
    @render.ui
    def cohort_table_header():
        df = pending_data.get()
        if df is None or df.empty:
            df = form_df.get()
        row_count = 0 if df is None else len(df)
        is_confirmed = (
            confirmed_ready.get()
            and pending_signature.get() is not None
            and pending_signature.get() == confirmed_signature.get()
        )

        if row_count == 0:
            status_label = "Empty"
            status_class = "set-status-empty"
        elif is_confirmed:
            status_label = "Confirmed"
            status_class = "set-status-confirmed"
        else:
            status_label = "Confirmation required"
            status_class = "set-status-pending"

        labels = _current_text_labels()
        return ui.div(
            ui.div(
                ui.tags.b(labels["cohort_title"]),
                ui.span(f"{row_count} {labels['plural']}", class_="set-row-count"),
                class_="cohort-title-group",
            ),
            ui.span(status_label, class_=f"set-status-pill {status_class}"),
            class_="cohort-table-title",
        )

    @output
    @render.ui
    def input_method_header():
        labels = _current_text_labels()
        return ui.div(
            ui.div("Add or load records", class_="data-input-method-title"),
            ui.div(f"Working {labels['set_lower']}", class_="data-input-method-subtitle"),
            class_="data-input-method-header",
        )

    @output
    @render.ui
    def use_case_summary():
        current_cfg = config_reactive.get() if config_reactive else config_init
        if not current_cfg:
            return ui.p("No use case selected.")

        features = current_cfg.get("features", [])
        labels = current_cfg.get("labels", {})
        positive_label = labels.get("positive_class_label", current_cfg.get("positive_class", "Positive"))
        negative_label = labels.get("negative_class_label", current_cfg.get("negative_class", "Negative"))

        return ui.div(
            ui.div(
                ui.div("Model objective", class_="use-case-summary-label"),
                ui.div(current_cfg.get("description", "Patient stratification model"), class_="use-case-summary-value"),
                class_="use-case-summary-item use-case-summary-wide",
            ),
            ui.div(
                ui.div(_current_text_labels()["groups_label"], class_="use-case-summary-label"),
                ui.div(f"{negative_label} / {positive_label}", class_="use-case-summary-value"),
                class_="use-case-summary-item",
            ),
            ui.div(
                ui.div("Input variables", class_="use-case-summary-label"),
                ui.div(str(len(features)), class_="use-case-summary-value"),
                class_="use-case-summary-item",
            ),
            class_="use-case-summary",
        )

    @output
    @render.ui
    def form_fields():
        current_cfg = config_reactive.get() if config_reactive else config_init
        form_sections, _ = build_form_fields(current_cfg or {})
        if form_sections:
            return ui.div(*form_sections, class_="manual-entry-grid")
        return ui.p("No features configured.")

    @output
    @render.ui
    def current_set_body():
        display_df = pending_data.get()
        if display_df is None or display_df.empty:
            display_df = output_data.get()

        if display_df is None or display_df.empty:
            labels = _current_text_labels()
            return ui.div(
                ui.div("No records yet", class_="empty-set-title"),
                ui.div(
                    f"Add manually, upload a file, or load the example {labels['set_lower']}.",
                    class_="empty-set-copy",
                ),
                class_="empty-set-state",
            )

        return ui.output_data_frame("out_patient_table")

    @output
    @render.data_frame
    def out_patient_table():
        display_df = pending_data.get()
        if display_df is None or display_df.empty:
            display_df = output_data.get()

        if display_df is None or display_df.empty:
            return render.DataGrid(pd.DataFrame(), width="100%", height="1px", selection_mode="none")

        display_df = display_df.copy()
        display_df.columns = [c.replace('_', ' ') for c in display_df.columns]
        display_df = _ensure_id(display_df)

        return render.DataGrid(
            display_df,
            width="100%",
            height="260px",
            selection_mode="rows"
        )

    return {"data": output_data, "is_custom": output_is_custom, "is_confirmed": confirmed_ready}
