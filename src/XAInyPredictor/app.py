"""Module providing a shiny UI."""
# Import modules
import pandas as pd
from pathlib import Path
from shiny import App, reactive, Session, ui

# Local imports
from XAInyPredictor.shinyapp import data_input, data_exploration, prediction
from XAInyPredictor.modules.data_processing import clean_data, process_raw_data, process_raw_form_data
from XAInyPredictor.modules.rai import delta_rai, read_delta_rai_formula, split_data_with_known_target, threshold_for_target_fnr


# --- Constants & Setup ---

WWW_DIR = Path(__file__).parent / "shinyapp" / "www"
EXAMPLE_RAW_FILE = Path(__file__).parent / "data" / "mock.csv"
EXAMPLE_PROC_FILE = Path(__file__).parent / "data" / "mock_processed.csv"
FORMULA_PKL_FILE = Path(__file__).parent / "data" / "formula.pkl"
BEST_FORMULA_FILE = Path(__file__).parent / "data" / "current_best_formula.txt"

# Load static model data once at startup
EXAMPLE_RAW_DF = pd.read_csv(EXAMPLE_RAW_FILE)
EXAMPLE_PROC_DF = process_raw_data(raw_df=EXAMPLE_RAW_DF, output_file=EXAMPLE_PROC_FILE, target='RAI-R')

# Ensure clean IDs for the model
EXAMPLE_RAW_DF = clean_data(EXAMPLE_RAW_DF)
EXAMPLE_RAW_DF['ID'] = [i + 1 for i in EXAMPLE_RAW_DF.index]
EXAMPLE_PROC_DF['ID'] = [i + 1 for i in EXAMPLE_PROC_DF.index]

# Pre-calculate example training components (Static)
X_TRAIN_EXAMPLE, X_TEST_EXAMPLE, Y_TRAIN_EXAMPLE, Y_TEST_EXAMPLE = split_data_with_known_target(
    EXAMPLE_PROC_DF, target='class_target', test_split=0.2
)
X_TRAIN_EXAMPLE_RAW = EXAMPLE_RAW_DF[EXAMPLE_RAW_DF['ID'].isin(X_TRAIN_EXAMPLE['ID'])] # for the visualization of data
X_TEST_EXAMPLE_RAW = EXAMPLE_RAW_DF[EXAMPLE_RAW_DF['ID'].isin(X_TEST_EXAMPLE['ID'])] # for the visualization of data
DELTA_FORMULA, _, FEATURE_ORDER, FEATS_IN_FORMULA = read_delta_rai_formula(FORMULA_PKL_FILE, BEST_FORMULA_FILE)
FEATURE_ORDER = [feat.replace(' ', '_') for feat in FEATURE_ORDER]
D_TRAIN_EXAMPLE = delta_rai(DELTA_FORMULA, X_TRAIN_EXAMPLE, FEATURE_ORDER)
D_TEST_EXAMPLE = delta_rai(DELTA_FORMULA, X_TEST_EXAMPLE, FEATURE_ORDER)
DEFAULT_PROB_THRESHOLD, _ = threshold_for_target_fnr(
    Y_TEST_EXAMPLE.to_numpy(),
    D_TEST_EXAMPLE['pred_prob'].to_numpy(),
    target_fnr=0
)


# --- UI Layout ---

page_dependencies = ui.tags.head(
    ui.tags.link(rel="stylesheet", type="text/css", href="layout.css"),
    ui.tags.link(rel="stylesheet", type="text/css", href="style.css"),
    ui.tags.script(src="index.js"),
    ui.tags.meta(name="description", content="Tool to run RAI and explore the results"),
    ui.tags.meta(name="theme-color", content="#000000"),
    ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
)

page_header = ui.tags.div(
    ui.tags.div(
        ui.tags.h3("XAInyPredictor", style="margin: 0; color: #007bff; font-weight: 800; letter-spacing: -1px;"),
        id="app-title",
        class_="navigation-title",
    ),
    ui.tags.div(
        ui.tags.div(
            ui.input_action_button(
                id="tab_data_input",
                label="1. Data Input",
                class_="navbar-button active-tab",
            ),
            id="div-navbar-map",
        ),
        ui.tags.div(
            ui.input_action_button(
                id="tab_data_exploration",
                label="2. Data Exploration",
                class_="navbar-button",
            ),
            id="div-navbar-map",
        ),
        ui.tags.div(
            ui.input_action_button(
                id="tab_prediction",
                label="3. Prediction",
                class_="navbar-button",
            ),
            id="div-navbar-plot",
        ),
        id="div-navbar-tabs",
        class_="navigation-menu",
    ),
    ui.tags.div(
        ui.tags.a(
            ui.tags.img(src="static/img/repo4eu_small_logo.png", height="40px"),
            href="https://repo4.eu/",
            target="_blank",
        ),
        id="logo-right",
        class_="navigation-logo-right",
    ),
    ui.tags.div(
        ui.input_action_button(
            id="info_icon",
            label=None,
            icon=ui.tags.i(class_="glyphicon glyphicon-info-sign"),
            class_="navbar-info",
        ),
        class_="navigation-info",
    ),
    id="div-navbar",
    class_="navbar-top page-header card-style",
)

app_ui = ui.page_fluid(
    page_dependencies,
    ui.tags.div(
        page_header,
        ui.tags.div(
            data_input.data_input_ui("data_input"),
            id="data-input-container",
            class_="page-main main-visible"
        ),
        ui.tags.div(
            data_exploration.data_exploration_ui("data_exploration"),
            id="inspect-input-container",
            class_="page-main"
        ),
        ui.tags.div(
            prediction.prediction_ui("prediction"),
            id="run-analysis-container",
            class_="page-main"
        ),
        class_="page-layout"
    ),
    title="XAInyPredictor",
)


# --- Server ---

def server(input, output, session: Session):

    # 1. Define reactive variables

    # Reactive variables related with the UI
    patient_selected = reactive.Value(None) # ID of the currently selected patient
    prob_threshold = reactive.Value(DEFAULT_PROB_THRESHOLD) # Probability threshold to classify patients
    data_available = reactive.Value(False) # Set to True when new data is available

    # Reactive variables related with the model
    delta_test_reactive = reactive.Value(D_TEST_EXAMPLE)
    x_test_reactive = reactive.Value(X_TEST_EXAMPLE)


    # 2. Load Data Module

    # We pass the static data in, and get the dynamic user data out
    input_results = data_input.server(
        "data_input", 
        X_TEST_EXAMPLE_RAW
    )
    # These are reactive.Value objects returned by the module
    analysis_data = input_results["data"] 
    is_custom_data = input_results["is_custom"]


    # 3. Update model variables

    @reactive.Effect
    def _update_analysis_context():
        current_df = analysis_data.get()
        is_custom = is_custom_data.get()

        if current_df is None:
            data_available.set(False)
            return
        data_available.set(True)

        if is_custom:
            current_proc_df = process_raw_form_data(raw_df=current_df, example_raw_df=EXAMPLE_RAW_DF)
            current_proc_df.columns = [col.replace(' ', '_') for col in current_proc_df.columns] # substitute spaces for _ before training
            d_test_curr = delta_rai(DELTA_FORMULA, current_proc_df, FEATURE_ORDER)
            delta_test_reactive.set(d_test_curr)
            x_test_reactive.set(current_proc_df)
        else:
            delta_test_reactive.set(D_TEST_EXAMPLE)
            x_test_reactive.set(X_TEST_EXAMPLE)


    # 4. Load Analysis Modules

    data_exploration.server(
        "data_exploration",
        analysis_data,
        X_TRAIN_EXAMPLE_RAW,
        patient_selected
    )
    prediction.server(
        "prediction",
        analysis_data,
        patient_selected,
        D_TRAIN_EXAMPLE,
        delta_test_reactive,
        X_TRAIN_EXAMPLE,
        Y_TRAIN_EXAMPLE,
        x_test_reactive,
        Y_TEST_EXAMPLE,
        prob_threshold,
        [feat.replace('_', ' ') for feat in FEATURE_ORDER],
        FEATS_IN_FORMULA
    )


    # 5. Define Tab Navigation Logic (JavaScript triggers)

    @reactive.Effect
    @reactive.event(input.tab_data_input)
    async def _():
        await session.send_custom_message(
            "toggleActiveTab", {"activeTab": "data_input"}
        )

    @reactive.Effect
    @reactive.event(input.tab_data_exploration)
    async def _():
        if not data_available():
            ui.notification_show("Please enter or upload data first.", type="warning")
            return
        await session.send_custom_message(
            "toggleActiveTab", {"activeTab": "data_exploration"}
        )

    @reactive.Effect
    @reactive.event(input.tab_prediction)
    async def _():
        if not data_available():
            ui.notification_show("Please enter or upload data first.", type="warning")
            return
        await session.send_custom_message(
            "toggleActiveTab", {"activeTab": "prediction"}
        )


    # 6. Help Modal Logic
    @reactive.Effect
    @reactive.event(input.info_icon)
    def _show_help_modal():
        m = ui.modal(
            ui.div(
                ui.tags.h4("Welcome to XAInyPredictor", style="color: #007bff; font-weight: bold; margin-top: 0;"),
                ui.p("This tool assists in predicting Radioiodine Refractoriness (RAI-R) in thyroid cancer patients.", style="font-style: italic; color: #666;"),
                ui.hr(),
                
                # Step 1: Input
                ui.row(
                    ui.column(2, ui.h1("1", style="background: #e9ecef; border-radius: 50%; width: 40px; height: 40px; text-align: center; line-height: 40px; font-size: 20px; color: #495057; margin: 0 auto;")),
                    ui.column(10, 
                        ui.h5("Input Patient Data", style="font-weight: bold; margin-top: 5px;"),
                        ui.p("Go to the ", ui.tags.b("Data Input"), " tab. You can upload a file (Excel/CSV) or manually enter details for a new patient (Age, TNM Staging, Histology, etc.).")
                    )
                ),
                ui.br(),

                # Step 2: Explore
                ui.row(
                    ui.column(2, ui.h1("2", style="background: #e9ecef; border-radius: 50%; width: 40px; height: 40px; text-align: center; line-height: 40px; font-size: 20px; color: #495057; margin: 0 auto;")),
                    ui.column(10, 
                        ui.h5("Explore Features", style="font-weight: bold; margin-top: 5px;"),
                        ui.p("In the ", ui.tags.b("Data Exploration"), " tab, compare your patient's specific risk factors (e.g., Tumor Size) against the reference population distribution.")
                    )
                ),
                ui.br(),

                # Step 3: Predict
                ui.row(
                    ui.column(2, ui.h1("3", style="background: #e9ecef; border-radius: 50%; width: 40px; height: 40px; text-align: center; line-height: 40px; font-size: 20px; color: #495057; margin: 0 auto;")),
                    ui.column(10, 
                        ui.h5("Run Prediction", style="font-weight: bold; margin-top: 5px;"),
                        ui.p("Click ", ui.tags.b("Prediction"), " to see the RAI-R probability. Use the Radar Chart to identify which clinical features are driving the high-risk prediction."),
                        ui.tags.div(
                            ui.tags.span("Green Row", style="background: #d1e7dd; color: #0f5132; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;"),
                            " = Low Risk | ",
                            ui.tags.span("Red Row", style="background: #f8d7da; color: #842029; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;"),
                            " = High Risk (Refractory)",
                            style="margin-top: 5px; font-size: 0.9em;"
                        )
                    )
                ),
            ),
            title="How to use this App",
            easy_close=True,
            footer=ui.modal_button("Got it!"),
            size="l" # Large modal for better readability
        )
        ui.modal_show(m)


# 7. Launch App

app = App(app_ui, server, static_assets=WWW_DIR)
