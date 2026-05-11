import pandas as pd
from shiny import Inputs, Outputs, Session, module, reactive, render, ui
from XAInyPredictor.modules.data_processing import clean_data


@module.ui
def data_input_ui():

    # Helper for consistent help icons
    def help_icon(msg):
        return ui.tooltip(
            ui.span(ui.tags.i(class_="glyphicon glyphicon-question-sign"), style="cursor: pointer; color: #007bc2; margin-left: 5px;"),
            msg,
            placement="right"
        )

    return ui.layout_sidebar(
        # --- SIDE BAR ---
        ui.sidebar(
            ui.h4("Data Source"),
            ui.input_radio_buttons(
                "input_method",
                "Select Method:",
                {"form": "Manual Entry", "file": "Upload File", "example": "Example Cohort"}
            ),
            ui.panel_conditional(
                "input.input_method == 'file'",
                ui.input_file(
                    "input_dataset_file",
                    "Upload File (TSV, CSV, Excel)",
                    accept=[".tsv", ".csv", ".xlsx"],
                    multiple=False,
                ),
                ui.p("Ensure columns match the ", ui.a("standard template", href="data_template.xlsx", target="_blank"), ".", style="font-size: 0.8em; color: gray;"),
            ),
        ),

        # --- MANUAL ENTRY FORM (only when input_method = form) ---
        ui.panel_conditional(
            "input.input_method == 'form'",
            ui.card(
                ui.card_header(ui.tags.b("➕ New Patient Entry")),
                
                # We use layout_columns to create a 3-column grid
                # col_widths sets the relative width of columns (total 12)
                ui.layout_columns(
                    
                    # COLUMN 1: Demographics & Tumor Basics
                    ui.div(
                        ui.h5("1. Demographics & Tumor", style="color: #007bc2; border-bottom: 1px solid #eee; padding-bottom: 5px;"),
                        ui.input_numeric("in_age", "Age (years)", value=50, min=1, max=120, step=1),
                        ui.input_select("in_gender", "Gender", choices={'M': 'Male', 'F': 'Female'}, selected="M"),
                        ui.input_numeric("in_bmi", "BMI (kg/m²)", value=27, min=10, max=60, step=1),
                        ui.input_numeric("in_tumor_size", "Tumor Size (cm)", value=5, min=0.5, max=15, step=0.5),
                        ui.input_select("in_histology", "Histology", choices={'OTC': 'Oncocytic Thyroid Carcinoma', 'PTC': 'Papillary Thyroid Carcinoma', 'FTC': 'Follicular Thyroid Carcinoma', 'PDTC': 'Poorly Differentiated Thyroid Carcinoma'}, selected='PTC'),
                    ),

                    # COLUMN 2: Staging & Risk
                    ui.div(
                        ui.h5("2. Staging & Risk", style="color: #007bc2; border-bottom: 1px solid #eee; padding-bottom: 5px;"),                        
                        ui.div(
                            ui.input_select("in_tumor_stage", "Tumor Stage", choices=['T1a', 'T1b', 'T2', 'T3a', 'T3b', 'T4a', 'T4b'], selected='T1a'),
                            help_icon("Based on AJCC 8th Edition. T1a: ≤1cm, T1b: >1cm-2cm, T2: >2cm-4cm limited to thyroid.")
                        , style="display: flex; align-items: center;"),
                        ui.div(
                            ui.input_select("in_node_stage", "Node Stage", choices=['N0', 'N1a', 'N1b', 'Nx'], selected='N0'),
                            help_icon("N1a: Level VI/VII compartments. N1b: Lateral neck or mediastinal nodes.")
                        , style="display: flex; align-items: center;"),
                        ui.input_select("in_metastases", "Metastasis Stage", choices=['M1', 'Mx'], selected='M1'),
                        ui.div(
                            ui.input_select("in_ata_risk", "ATA Risk Level", choices=['Low', 'Intermediate', 'High'], selected='Low'),
                            help_icon("2015 ATA Guidelines. High Risk includes: Gross ETE, incomplete resection, distant mets, or large/invasive nodes.")
                        , style="display: flex; align-items: center;"),
                    ),

                    # COLUMN 3: Pathology & Treatment
                    ui.div(
                        ui.h5("3. Pathology & Treatment", style="color: #007bc2; border-bottom: 1px solid #eee; padding-bottom: 5px;"),   
                        ui.div(
                            ui.input_select("in_ete", "Extrathyroidal Ext. (ETE)", choices={'YES': 'Yes', 'NO': 'No'}, selected='NO'),
                            help_icon("Microscopic or Gross extension beyond the thyroid capsule.")
                        , style="display: flex; align-items: center;"),
                        ui.input_select("in_multifocality", "Multifocality", choices={'YES': 'Yes', 'NO': 'No'}, selected='NO'),
                        ui.input_select("in_vascular_invasion", "Vascular Invasion", choices={'YES': 'Yes', 'NO': 'No'}, selected='NO'),
                        ui.input_select("in_resection", "Resection Status", choices=['R0', 'R1', 'R2'], selected='R0'),
                        ui.input_select("in_goal_of_rai", "Goal of RAI", choices={'ABLATION': 'Ablation', 'ADIUVANT': 'Adjuvant', 'THERAPEUTIC': 'Therapeutic'}, selected='ABLATION'),
                    ),
                    col_widths=(4, 4, 4)
                ),
                
                ui.card_footer(
                    ui.input_action_button("btn_add_form", "Add Patient to Cohort", class_="btn-primary", width="100%")
                )
            ),
            ui.br()
        ),

        # --- COHORT TABLE ---
        ui.card(
            ui.card_header(ui.tags.b("Current Patient Cohort")),
            ui.output_data_frame("out_patient_table"),
            ui.panel_conditional(
                "input.input_method == 'form'",
                ui.input_action_button("btn_delete_selected", "Delete Selected", class_="btn-danger btn-sm", width="100%")
            ),
            full_screen=True
        )
    )


@module.server
def server(input: Inputs, output: Outputs, session: Session, example_raw_data):
    """
    Returns a dictionary of reactive values:
    - 'data': The processed dataframe ready for analysis
    - 'is_custom': Boolean, True if user provided their own data
    """

    # Define reactive value for the form dataframe
    feature_cols = ['Age', 'Gender', 'BMI', 'Tumor size (cm)', 'Tumor stage', 'Node stage', 'Metastases', 'ATA risk', 'Histology', 'ETE', 'Multifocality', 'Vascular invasion', 'Resection', 'Goal of RAI']
    form_df = reactive.Value(pd.DataFrame(columns=['ID'] + feature_cols))
    # Define reactive value that counts manual IDs (to prevent duplicates when
    # inserting/deleting)
    id_counter = reactive.Value(1)

    # These are the values we will expose to the main app
    output_data = reactive.Value(None)
    output_is_custom = reactive.Value(False)

    # --- HELPERS ---

    def _ensure_id(df):
        """Ensures ID column exists and is first"""
        if df is None: return None
        if 'ID' not in df.columns:
            df.insert(0, 'ID', range(1, len(df) + 1))
        else:
            cols = ['ID'] + [c for c in df.columns if c != 'ID']
            df = df[cols]
        return df

    # --- HANDLERS ---

    @reactive.Effect
    @reactive.event(input.btn_add_form)
    def _add_patient():
        current_df = form_df()
        new_id = id_counter()

        new_row = pd.DataFrame([{
            'ID': new_id,
            'Age': float(input.in_age()),
            'Gender': input.in_gender(),
            'BMI': float(input.in_bmi()),
            'Tumor size (cm)': float(input.in_tumor_size()),
            'Tumor stage': input.in_tumor_stage(),
            'Node stage': input.in_node_stage(),
            'Metastases': input.in_metastases(),
            'ATA risk': input.in_ata_risk(),
            'Histology': input.in_histology(),
            'ETE': input.in_ete(),
            'Multifocality': input.in_multifocality(),
            'Vascular invasion': input.in_vascular_invasion(),
            'Resection': input.in_resection(),
            'Goal of RAI': input.in_goal_of_rai()
        }])

        updated_df = pd.concat([current_df, new_row], ignore_index=True)
        form_df.set(updated_df)
        id_counter.set(new_id + 1)
        ui.notification_show("Patient added.", type="message")

    @reactive.Effect
    @reactive.event(input.btn_delete_selected)
    def _delete_patient():
        # Get indices of selected rows (returns a tuple of integers)
        selected_rows = input.out_patient_table_selected_rows()
        if not selected_rows:
            ui.notification_show("No rows selected.", type="warning")
            return
            
        current_df = form_df.get()        
        updated_df = current_df.drop(index=list(selected_rows)).reset_index(drop=True) # drop rows by integer index
        form_df.set(updated_df)
        ui.notification_show(f"Deleted {len(selected_rows)} patient(s).", type="message")

    # --- PIPELINE ---

    @reactive.Effect
    def _update_output_pipeline():
        """
        Determines which dataset to show
        """
        method = str(input.input_method())
        raw_df = None
        is_custom = False

        if method == "example":
            # Example data is already processed, so we just use it
            raw_df = example_raw_data.copy()
            is_custom = False
        
        elif method == "form":
            # For forms, we need to process the raw inputs into model features
            raw_df = form_df.get()
            is_custom = True

        elif method == "file":
            file_info = input.input_dataset_file()
            is_custom = True
            if not file_info:
                # If no file uploaded yet, show empty table
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
                return None

        # Final Standardization for the App
        if raw_df is not None and not raw_df.empty:
            clean_df = _ensure_id(raw_df)
            output_data.set(clean_df)
        else:
            output_data.set(None)
            
        output_is_custom.set(is_custom)
        return

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

        # Process column names so that they have spaces
        display_df.columns = [c.replace('_', ' ') for c in display_df.columns]

        # Ensure the column ID exists
        display_df = _ensure_id(display_df)

        # Round values for visualization in the UI
        for col in ["Age", "BMI", "Tumor size (cm)"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].astype(float).round(2)

        return render.DataGrid(
            display_df,
            width="100%",
            selection_mode="rows" # enables the checkbox/click selection
        )

    return {"data": output_data, "is_custom": output_is_custom}