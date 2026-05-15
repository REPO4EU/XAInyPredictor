"""Module providing a shiny UI."""
import pandas as pd
from pathlib import Path
from shiny import App, reactive, Session, ui

from XAInyPredictor.shinyapp import data_input, data_exploration, prediction
from XAInyPredictor.modules.model_registry import discover_use_cases, load_use_case, UseCaseNotFoundError
from XAInyPredictor.modules.data_processing import clean_data, process_raw_data, process_raw_form_data
from XAInyPredictor.modules.xai import delta_xai, read_delta_xai_formula, split_data_with_known_target, threshold_for_target_fnr


WWW_DIR = Path(__file__).parent / "shinyapp" / "www"

USE_CASES = discover_use_cases()
_USE_CASE_CACHE = {}

def _load_model_data_cached(use_case: str):
    if use_case in _USE_CASE_CACHE:
        return _USE_CASE_CACHE[use_case]
    data = load_model_data(use_case)
    _USE_CASE_CACHE[use_case] = data
    return data


def build_startup_modal():
    use_case_choices = {name: info["display_name"] for name, info in USE_CASES.items()}
    return ui.modal(
        ui.tags.div(
            ui.tags.h3("Select Use Case", style="color: #007bff; font-weight: bold; text-align: center;"),
            ui.tags.p("Choose which prediction model to use:", style="text-align: center;"),
            ui.br(),
            ui.input_select(
                id="startup_use_case",
                label="Available Use Cases:",
                choices=use_case_choices,
                selected="rai",
            ),
        ),
        title="Welcome to XAInyPredictor",
        easy_close=False,
        footer=ui.TagList(
            ui.input_action_button("confirm_startup_use_case", "Start", class_="btn-primary", onclick="Shiny.setInputValue('startup_confirmed', 'yes');")
        )
    )


def build_page_header(config: dict, current_use_case: str):
    use_case_choices = {name: info["display_name"] for name, info in USE_CASES.items()}

    return ui.tags.div(
        ui.tags.div(
            ui.tags.h3(
                config.get("titles", {}).get("app_title", "XAInyPredictor"),
                style="margin: 0; color: #007bff; font-weight: 800; letter-spacing: -1px;"
            ),
            id="app-title",
            class_="navigation-title",
        ),
        ui.tags.div(
            ui.tags.div(
                ui.input_action_button(
                    id="tab_data_input",
                    label=config.get("titles", {}).get("tab_data_input", "1. Data Input"),
                    class_="navbar-button active-tab",
                ),
                id="div-navbar-map",
            ),
            ui.tags.div(
                ui.input_action_button(
                    id="tab_data_exploration",
                    label=config.get("titles", {}).get("tab_data_exploration", "2. Data Exploration"),
                    class_="navbar-button",
                ),
                id="div-navbar-map",
            ),
            ui.tags.div(
                ui.input_action_button(
                    id="tab_prediction",
                    label=config.get("titles", {}).get("tab_prediction", "3. Prediction"),
                    class_="navbar-button",
                ),
                id="div-navbar-plot",
            ),
            ui.tags.div(
                ui.input_select(
                    id="use_case_selector",
                    label=None,
                    choices=use_case_choices,
                    selected=current_use_case,
                ),
                style="display: flex; align-items: center; margin-left: 10px;"
            ),
            id="div-navbar-tabs",
            class_="navigation-menu",
        ),
        ui.tags.div(
            ui.tags.a(
                ui.tags.img(src="static/img/repo4eu_small_logo.png", height="40px"),
                href="https://repo4eu/",
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


def build_ui(config: dict, current_use_case: str):
    page_dependencies = ui.tags.head(
        ui.tags.link(rel="stylesheet", type="text/css", href="layout.css"),
        ui.tags.link(rel="stylesheet", type="text/css", href="style.css"),
        ui.tags.script(src="index.js"),
        ui.tags.meta(name="description", content=config.get("description", "XAI Predictor")),
        ui.tags.meta(name="theme-color", content="#000000"),
        ui.tags.meta(name="viewport", content="width=device-width, initial-scale=1"),
    )

    page_header = build_page_header(config, current_use_case)

    return ui.page_fluid(
        page_dependencies,
        ui.tags.div(
            page_header,
            ui.tags.div(
                data_input.data_input_ui("data_input", DEFAULT_CONFIG),
                id="data-input-container",
                class_="page-main main-visible"
            ),
            ui.tags.div(
                data_exploration.data_exploration_ui("data_exploration", DEFAULT_CONFIG),
                id="inspect-input-container",
                class_="page-main"
            ),
            ui.tags.div(
                prediction.prediction_ui("prediction", DEFAULT_CONFIG),
                id="run-analysis-container",
                class_="page-main"
            ),
            class_="page-layout"
        ),
        title=config.get("titles", {}).get("app_title", "XAInyPredictor"),
    )


def load_model_data(use_case_name: str):
    use_case_data = load_use_case(use_case_name)
    config = use_case_data["config"]
    encoding_dict = config.get("encoding", {})
    example_raw_df = use_case_data["example_data"].copy()
    feature_order = use_case_data["feature_order"]

    use_case_path = use_case_data["path"]
    model_file = use_case_path / "model.pkl"
    formula_file = use_case_path / "feature_order.txt"

    target_col = config.get("target_column", "target")
    allowed_columns = [f["name"] for f in config.get("features", [])]
    example_proc_df = process_raw_data(
        raw_df=example_raw_df.copy(),
        output_file=None,
        encoding_config=encoding_dict,
        target=target_col,
        allowed_columns=allowed_columns,
    )

    example_raw_df = clean_data(example_raw_df, allowed_columns)
    example_raw_df['ID'] = [i + 1 for i in example_raw_df.index]
    example_proc_df['ID'] = [i + 1 for i in example_proc_df.index]

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = split_data_with_known_target(
        example_proc_df, target='class_target', test_split=0.2
    )
    X_TRAIN_RAW = example_raw_df[example_raw_df['ID'].isin(X_TRAIN['ID'])]
    X_TEST_RAW = example_raw_df[example_raw_df['ID'].isin(X_TEST['ID'])]

    DELTA_FORMULA, _, FEATURE_ORDER, FEATS_IN_FORMULA = read_delta_xai_formula(str(model_file), str(formula_file))
    FEATURE_ORDER_CLEAN = [feat.replace(' ', '_') for feat in FEATURE_ORDER]

    D_TRAIN = delta_xai(DELTA_FORMULA, X_TRAIN, FEATURE_ORDER_CLEAN)
    D_TEST = delta_xai(DELTA_FORMULA, X_TEST, FEATURE_ORDER_CLEAN)

    DEFAULT_THRESHOLD, _ = threshold_for_target_fnr(
        Y_TEST.to_numpy(),
        D_TEST['pred_prob'].to_numpy(),
        target_fnr=0
    )

    return {
        "config": config,
        "delta_formula": DELTA_FORMULA,
        "example_raw_df": example_raw_df,
        "example_proc_df": example_proc_df,
        "X_TRAIN_RAW": X_TRAIN_RAW,
        "X_TEST_RAW": X_TEST_RAW,
        "X_TRAIN": X_TRAIN,
        "X_TEST": X_TEST,
        "Y_TRAIN": Y_TRAIN,
        "Y_TEST": Y_TEST,
        "D_TRAIN": D_TRAIN,
        "D_TEST": D_TEST,
        "FEATURE_ORDER_CLEAN": FEATURE_ORDER_CLEAN,
        "FEATURE_ORDER_DISPLAY": [feat.replace('_', ' ') for feat in FEATURE_ORDER_CLEAN],
        "FEATS_IN_FORMULA": FEATS_IN_FORMULA,
        "DEFAULT_THRESHOLD": DEFAULT_THRESHOLD,
    }


DEFAULT_USE_CASE = "rai"
MODEL_DATA = load_model_data(DEFAULT_USE_CASE)
DEFAULT_CONFIG = MODEL_DATA["config"]

app_ui = build_ui(DEFAULT_CONFIG, DEFAULT_USE_CASE)


def server(input, output, session: Session):
    current_use_case = reactive.Value(DEFAULT_USE_CASE)
    model_data = reactive.Value(MODEL_DATA)
    config = reactive.Value(DEFAULT_CONFIG)

    patient_selected = reactive.Value(None)
    prob_threshold = reactive.Value(MODEL_DATA["DEFAULT_THRESHOLD"])
    data_available = reactive.Value(False)

    delta_test_reactive = reactive.Value(MODEL_DATA["D_TEST"])
    x_test_reactive = reactive.Value(MODEL_DATA["X_TEST"])

    @reactive.Effect
    def _show_startup_modal():
        ui.modal_show(build_startup_modal())

    startup_initialized = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.confirm_startup_use_case)
    def _on_startup_confirm():
        if input.startup_confirmed() == "yes" and not startup_initialized.get():
            selected_use_case = input.startup_use_case()
            startup_initialized.set(True)

            if selected_use_case == DEFAULT_USE_CASE:
                new_model_data = MODEL_DATA
            else:
                try:
                    new_model_data = _load_model_data_cached(selected_use_case)
                except (UseCaseNotFoundError, Exception) as e:
                    ui.notification_show(f"Error loading use case: {e}", type="error")
                    startup_initialized.set(False)
                    return

            current_use_case.set(selected_use_case)
            model_data.set(new_model_data)
            config.set(new_model_data["config"])
            delta_test_reactive.set(new_model_data["D_TEST"])
            x_test_reactive.set(new_model_data["X_TEST"])
            prob_threshold.set(new_model_data["DEFAULT_THRESHOLD"])
            data_available.set(True)

            session.send_custom_message("set_input_value", {"id": "use_case_selector", "value": selected_use_case})

            ui.modal_remove()

    @reactive.Effect
    @reactive.event(input.use_case_selector)
    def _switch_use_case():
        if not startup_initialized.get():
            return
        new_use_case = input.use_case_selector()
        if new_use_case and new_use_case != current_use_case.get():
            ui.modal_show(
                ui.modal(
                    ui.tags.p("Switching use case will clear current data. Continue?"),
                    title="Confirm Use Case Change",
                    footer=ui.TagList(
                        ui.modal_button("Cancel", onclick="Shiny.setInputValue('use_case_confirm', 'cancel');"),
                        ui.input_action_button("confirm_switch", "Confirm", class_="btn-primary", onclick="Shiny.setInputValue('use_case_confirm', 'confirm');")
                    )
                )
            )

    @reactive.Effect
    @reactive.event(input.confirm_switch)
    def _confirm_switch():
        if input.use_case_confirm() == "cancel":
            return

        new_use_case = input.use_case_selector()
        try:
            new_model_data = _load_model_data_cached(new_use_case)
            current_use_case.set(new_use_case)
            model_data.set(new_model_data)
            config.set(new_model_data["config"])

            delta_test_reactive.set(new_model_data["D_TEST"])
            x_test_reactive.set(new_model_data["X_TEST"])
            prob_threshold.set(new_model_data["DEFAULT_THRESHOLD"])
            data_available.set(False)
            patient_selected.set(None)

            ui.notification_show(f"Switched to {new_model_data['config'].get('name', new_use_case)}", type="message")

            session.send_custom_message("toggleActiveTab", {"activeTab": "data_input"})

            ui.modal_remove()

        except (UseCaseNotFoundError, Exception) as e:
            ui.notification_show(f"Error loading use case: {e}", type="error")

    @reactive.Effect
    @reactive.event(input.btn_cancel_switch)
    def _cancel_switch():
        pass

    input_results = data_input.server("data_input", model_data, DEFAULT_CONFIG, config)
    analysis_data = input_results["data"]
    is_custom_data = input_results["is_custom"]

    @reactive.Effect
    def _update_analysis_context():
        current_df = analysis_data.get()
        is_custom = is_custom_data.get()
        md = model_data.get()
        cfg = config.get()

        if current_df is None:
            data_available.set(False)
            return

        data_available.set(True)

        if is_custom:
            allowed_columns = [f["name"] for f in cfg.get("features", [])]
            current_proc_df = process_raw_form_data(
                raw_df=current_df,
                example_raw_df=md["example_raw_df"],
                encoding_config=cfg.get("encoding", {}),
                allowed_columns=allowed_columns
            )
            current_proc_df.columns = [col.replace(' ', '_') for col in current_proc_df.columns]
            d_test_curr = delta_xai(md.get("delta_formula"), current_proc_df, md["FEATURE_ORDER_CLEAN"])
            delta_test_reactive.set(d_test_curr)
            x_test_reactive.set(current_proc_df)
        else:
            delta_test_reactive.set(md["D_TEST"])
            x_test_reactive.set(md["X_TEST"])

    data_exploration.server(
        "data_exploration",
        analysis_data,
        model_data,
        patient_selected,
        DEFAULT_CONFIG,
        config
    )

    prediction.server(
        "prediction",
        analysis_data,
        patient_selected,
        model_data,
        delta_test_reactive,
        x_test_reactive,
        prob_threshold,
        DEFAULT_CONFIG,
        config
    )

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

    @reactive.Effect
    @reactive.event(input.info_icon)
    def _show_help_modal():
        cfg = config.get()
        labels = cfg.get("labels", {})
        titles = cfg.get("titles", {})

        m = ui.modal(
            ui.div(
                ui.tags.h4(f"Welcome to {titles.get('app_title', 'XAInyPredictor')}", style="color: #007bff; font-weight: bold; margin-top: 0;"),
                ui.p(cfg.get("description", ""), style="font-style: italic; color: #666;"),
                ui.hr(),

                ui.row(
                    ui.column(2, ui.h1("1", style="background: #e9ecef; border-radius: 50%; width: 40px; height: 40px; text-align: center; line-height: 40px; font-size: 20px; color: #495057; margin: 0 auto;")),
                    ui.column(10,
                        ui.h5("Input Patient Data", style="font-weight: bold; margin-top: 5px;"),
                        ui.p(f"Go to the ", ui.tags.b(titles.get('tab_data_input', 'Data Input')), " tab. You can upload a file or manually enter patient details.")
                    )
                ),
                ui.br(),

                ui.row(
                    ui.column(2, ui.h1("2", style="background: #e9ecef; border-radius: 50%; width: 40px; height: 40px; text-align: center; line-height: 40px; font-size: 20px; color: #495057; margin: 0 auto;")),
                    ui.column(10,
                        ui.h5("Explore Features", style="font-weight: bold; margin-top: 5px;"),
                        ui.p(f"In the ", ui.tags.b(titles.get('tab_data_exploration', 'Data Exploration')), " tab, compare patient features against the reference population.")
                    )
                ),
                ui.br(),

                ui.row(
                    ui.column(2, ui.h1("3", style="background: #e9ecef; border-radius: 50%; width: 40px; height: 40px; text-align: center; line-height: 40px; font-size: 20px; color: #495057; margin: 0 auto;")),
                    ui.column(10,
                        ui.h5("Run Prediction", style="font-weight: bold; margin-top: 5px;"),
                        ui.p(f"Click ", ui.tags.b(titles.get('tab_prediction', 'Prediction')), f" to see the {labels.get('probability_column', 'Probability')}. Use the charts to identify which features are driving the prediction."),
                        ui.tags.div(
                            ui.tags.span(f"{labels.get('negative_class', 'Negative')}", style="background: #d1e7dd; color: #0f5132; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;"),
                            " = Low Risk | ",
                            ui.tags.span(f"{labels.get('positive_class', 'Positive')}", style="background: #f8d7da; color: #842029; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;"),
                            " = High Risk",
                            style="margin-top: 5px; font-size: 0.9em;"
                        )
                    )
                ),
            ),
            title="How to use this App",
            easy_close=True,
            footer=ui.modal_button("Got it!"),
            size="l"
        )
        ui.modal_show(m)


app = App(app_ui, server, static_assets=WWW_DIR)