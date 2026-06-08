"""Module providing a shiny UI."""
import os
from pathlib import Path
import sys

import time
import socket
import matplotlib
import threading
import webbrowser
import pandas as pd

from sklearn.preprocessing import StandardScaler
from shiny import App, reactive, Session, ui

matplotlib.use("Agg")

from XAInyPredictor.shinyapp import data_input, data_exploration, prediction
from XAInyPredictor.modules.model_registry import discover_use_cases, load_use_case, UseCaseNotFoundError
from XAInyPredictor.modules.data_processing import clean_data, process_raw_data, process_raw_form_data
from XAInyPredictor.modules.neoag_processing import prepare_neoantigen_features
from XAInyPredictor.modules.xai import delta_xai, read_delta_xai_formula, split_data_with_known_target, threshold_for_target_fnr


def _package_resource_dir(*parts: str) -> Path:
    candidates = []

    if getattr(sys, "frozen", False):
        bundle_root = Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
        candidates.extend(
            [
                bundle_root / "XAInyPredictor" / Path(*parts),
                Path(sys.executable).parent / "_internal" / "XAInyPredictor" / Path(*parts),
            ]
        )

    candidates.append(Path(__file__).resolve().parent / Path(*parts))

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


WWW_DIR = _package_resource_dir("shinyapp", "www")

_USE_CASE_CACHE = {}

def _get_use_cases() -> dict:
    return discover_use_cases()


def _use_case_choices() -> dict:
    return {name: info["display_name"] for name, info in _get_use_cases().items()}


def _load_model_data_cached(use_case: str):
    if use_case in _USE_CASE_CACHE:
        return _USE_CASE_CACHE[use_case]
    data = load_model_data(use_case)
    _USE_CASE_CACHE[use_case] = data
    return data


def build_startup_modal():
    use_case_choices = _use_case_choices()
    return ui.modal(
        ui.tags.div(
            ui.tags.h3("Select Use Case", style="color: #007bff; font-weight: bold; text-align: center;"),
            ui.tags.p("Interpretable stratification support for clinical research workflows.", style="text-align: center; color: #5f6f7f;"),
            ui.tags.p("Choose which stratification support model to use:", style="text-align: center;"),
            ui.br(),
            ui.input_select(
                id="startup_use_case",
                label="Available Use Cases:",
                choices=use_case_choices,
                selected=DEFAULT_USE_CASE if DEFAULT_USE_CASE in use_case_choices else None,
            ),
        ),
        title="Welcome to XAInyPredictor",
        easy_close=False,
        footer=ui.TagList(
            ui.input_action_button(
                "confirm_startup_use_case",
                "Start",
                class_="btn-primary",
            )
        )
    )


def build_page_header(config: dict, current_use_case: str):
    use_case_choices = _use_case_choices()

    return ui.tags.div(
        ui.tags.div(
            ui.tags.div(
                ui.tags.h3(
                    config.get("titles", {}).get("app_title", "XAInyPredictor"),
                ),
                ui.tags.span(
                    config.get("titles", {}).get("app_subtitle", "Interpretable Stratification Support"),
                    class_="navigation-subtitle",
                ),
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
                    label=config.get("titles", {}).get("tab_data_exploration", "2. Cohort Context"),
                    class_="navbar-button",
                ),
                id="div-navbar-map",
            ),
            ui.tags.div(
                ui.input_action_button(
                    id="tab_prediction",
                    label=config.get("titles", {}).get("tab_prediction", "3. Patient Stratification"),
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
        ui.tags.link(rel="stylesheet", type="text/css", href="layout.css?v=20260608a"),
        ui.tags.link(rel="stylesheet", type="text/css", href="style.css?v=20260608a"),
        ui.tags.script(src="index.js?v=20260608a"),
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
            ui.tags.div(
                ui.tags.div(
                    ui.tags.div(class_="loading-spinner"),
                    ui.tags.div("Loading use case...", id="loading-overlay-title", class_="loading-title"),
                    ui.tags.div("Preparing model outputs and reference context.", id="loading-overlay-subtitle", class_="loading-subtitle"),
                    class_="loading-panel",
                ),
                id="use-case-loading-overlay",
                class_="loading-overlay",
                **{"aria-live": "polite"},
            ),
            class_="page-layout"
        ),
        ui.tags.div(
            config.get("titles", {}).get(
                "footer_text",
                "Research prototype: clinical workflow utility and interpretation should be validated with clinical collaborators.",
            ),
            class_="app-validation-footer",
        ),
        title=config.get("titles", {}).get("app_title", "XAInyPredictor"),
    )


async def update_navigation_labels(session: Session, config: dict):
    titles = config.get("titles", {})
    labels = config.get("labels", {})
    entity = labels.get("entity", {})
    ui.update_action_button("tab_data_input", label=titles.get("tab_data_input", "1. Data Input"))
    ui.update_action_button("tab_data_exploration", label=titles.get("tab_data_exploration", "2. Cohort Context"))
    ui.update_action_button("tab_prediction", label=titles.get("tab_prediction", "3. Patient Stratification"))
    await session.send_custom_message(
        "setStratificationTabLabels",
        {
            "patient": entity.get("singular_title", "Patient"),
            "cohort": entity.get("set_title", "Cohort"),
            "reference": entity.get("reference_title", "Reference Patients"),
        },
    )
    await session.send_custom_message(
        "setDataInputMethodLabels",
        {
            "form": labels.get("manual_entry", "Manual Entry"),
            "file": labels.get("upload_file", "Upload File"),
            "example": labels.get("example_cohort", "Example Cohort"),
        },
    )


def load_model_data(use_case_name: str):
    use_case_data = load_use_case(use_case_name)
    config = use_case_data["config"]
    if config.get("use_case_type") == "neoantigen":
        return load_neoag_model_data(use_case_data)

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


def _with_id(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    if "ID" not in df.columns:
        df.insert(0, "ID", range(1, len(df) + 1))
    return df


def _scale_features(features: pd.DataFrame, scaler: StandardScaler, feature_order: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        scaler.transform(features[feature_order]),
        columns=feature_order,
        index=features.index,
    )


def _limit_neoag_reference_df(reference_df: pd.DataFrame, target_col: str, max_candidates: int | None) -> pd.DataFrame:
    if not max_candidates or len(reference_df) <= max_candidates:
        return reference_df

    if target_col not in reference_df.columns:
        return reference_df.sample(n=max_candidates, random_state=42).reset_index(drop=True)

    sampled_parts = []
    class_counts = reference_df[target_col].value_counts()
    for class_value, class_count in class_counts.items():
        class_fraction = class_count / len(reference_df)
        class_sample_size = max(1, round(max_candidates * class_fraction))
        class_df = reference_df[reference_df[target_col] == class_value]
        sampled_parts.append(
            class_df.sample(n=min(class_sample_size, len(class_df)), random_state=42)
        )

    sampled_df = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled_df) > max_candidates:
        sampled_df = sampled_df.sample(n=max_candidates, random_state=42)

    return sampled_df.sample(frac=1, random_state=42).reset_index(drop=True)


def load_neoag_model_data(use_case_data: dict):
    config = use_case_data["config"]
    use_case_path = use_case_data["path"]
    feature_order = use_case_data["feature_order"]
    model_file = use_case_path / "model.pkl"
    formula_file = use_case_path / "feature_order.txt"
    target_col = config.get("target_column", "Qualitative_Measure")

    reference_raw_df = pd.read_csv(use_case_path / "reference_data.csv")
    reference_raw_df = _limit_neoag_reference_df(
        reference_raw_df,
        target_col=target_col,
        max_candidates=config.get("max_reference_candidates"),
    )
    example_raw_df = use_case_data["example_data"].copy()

    reference_prepared = prepare_neoantigen_features(reference_raw_df, feature_order, target=target_col)
    example_prepared = prepare_neoantigen_features(example_raw_df, feature_order, target=target_col)

    scaler = StandardScaler()
    train_scaled = pd.DataFrame(
        scaler.fit_transform(reference_prepared.features[feature_order]),
        columns=feature_order,
        index=reference_prepared.features.index,
    )
    test_scaled = _scale_features(example_prepared.features, scaler, feature_order)

    delta_formula, _, _, feats_in_formula = read_delta_xai_formula(
        str(model_file),
        str(formula_file),
        simplify_formula=False,
    )
    combined_scaled = pd.concat([train_scaled, test_scaled], ignore_index=True)
    d_combined = delta_xai(delta_formula, combined_scaled, feature_order)
    d_train = d_combined.iloc[:len(train_scaled)].reset_index(drop=True)
    d_test = d_combined.iloc[len(train_scaled):].reset_index(drop=True)

    y_train = reference_prepared.target if reference_prepared.target is not None else pd.Series([0] * len(reference_raw_df))
    y_test = example_prepared.target if example_prepared.target is not None else pd.Series([0] * len(example_raw_df))
    default_threshold, _ = threshold_for_target_fnr(
        y_train.to_numpy(),
        d_train["pred_prob"].to_numpy(),
        target_fnr=float(config.get("false_negative_rate", 0.10)),
    )

    x_train = _with_id(reference_prepared.features)
    x_test = _with_id(example_prepared.features)
    x_train_raw = _with_id(reference_raw_df.drop(columns=[target_col], errors="ignore"))
    x_test_raw = _with_id(example_raw_df.drop(columns=[target_col], errors="ignore"))

    return {
        "config": config,
        "delta_formula": delta_formula,
        "scaler": scaler,
        "example_raw_df": _with_id(example_raw_df.drop(columns=[target_col], errors="ignore")),
        "example_proc_df": _with_id(example_prepared.features),
        "X_TRAIN_RAW": x_train_raw,
        "X_TEST_RAW": x_test_raw,
        "X_TRAIN": x_train,
        "X_TEST": x_test,
        "Y_TRAIN": y_train,
        "Y_TEST": y_test,
        "D_TRAIN": d_train,
        "D_TEST": d_test,
        "FEATURE_ORDER_CLEAN": feature_order,
        "FEATURE_ORDER_DISPLAY": [feat.replace("_", " ") for feat in feature_order],
        "FEATS_IN_FORMULA": [feat.replace("_", " ") for feat in feats_in_formula],
        "DEFAULT_THRESHOLD": default_threshold,
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
    pending_use_case = reactive.Value(None)

    delta_test_reactive = reactive.Value(MODEL_DATA["D_TEST"])
    x_test_reactive = reactive.Value(MODEL_DATA["X_TEST"])

    @reactive.Effect
    def _show_startup_modal():
        ui.modal_show(build_startup_modal())

    @reactive.Effect
    def _refresh_use_case_selector():
        choices = _use_case_choices()
        ui.update_select(
            "use_case_selector",
            choices=choices,
            selected=current_use_case.get() if current_use_case.get() in choices else None,
        )

    startup_initialized = reactive.Value(False)

    @reactive.Effect
    @reactive.event(input.confirm_startup_use_case)
    async def _on_startup_confirm():
        if not startup_initialized.get():
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
                    await session.send_custom_message("setUseCaseLoading", {"visible": False})
                    return

            current_use_case.set(selected_use_case)
            model_data.set(new_model_data)
            config.set(new_model_data["config"])
            await update_navigation_labels(session, new_model_data["config"])
            delta_test_reactive.set(new_model_data["D_TEST"])
            x_test_reactive.set(new_model_data["X_TEST"])
            prob_threshold.set(new_model_data["DEFAULT_THRESHOLD"])

            ui.update_select("use_case_selector", choices=_use_case_choices(), selected=selected_use_case)

            ui.modal_remove()
            await session.send_custom_message("setUseCaseLoading", {"visible": False})

    @reactive.Effect
    @reactive.event(input.use_case_selector)
    def _switch_use_case():
        if not startup_initialized.get():
            return
        new_use_case = input.use_case_selector()
        if new_use_case and new_use_case != current_use_case.get():
            choices = _use_case_choices()
            if new_use_case not in choices:
                ui.notification_show(f"Use case '{new_use_case}' is no longer available.", type="error")
                ui.update_select("use_case_selector", choices=choices, selected=current_use_case.get())
                return
            pending_use_case.set(new_use_case)
            ui.update_select("use_case_selector", selected=current_use_case.get())
            ui.modal_show(
                ui.modal(
                    ui.tags.p("Switching use case will clear current data. Continue?"),
                    title="Confirm Use Case Change",
                    footer=ui.TagList(
                        ui.input_action_button("cancel_switch", "Cancel", class_="btn-default"),
                        ui.input_action_button("confirm_switch", "Confirm", class_="btn-primary")
                    )
                )
            )

    @reactive.Effect
    @reactive.event(input.cancel_switch)
    def _cancel_switch():
        pending_use_case.set(None)
        ui.update_select("use_case_selector", selected=current_use_case.get())
        ui.modal_remove()

    @reactive.Effect
    @reactive.event(input.confirm_switch)
    async def _confirm_switch():
        new_use_case = pending_use_case.get()
        if not new_use_case:
            ui.update_select("use_case_selector", selected=current_use_case.get())
            ui.modal_remove()
            await session.send_custom_message("setUseCaseLoading", {"visible": False})
            return

        try:
            new_model_data = _load_model_data_cached(new_use_case)
            current_use_case.set(new_use_case)
            model_data.set(new_model_data)
            config.set(new_model_data["config"])
            await update_navigation_labels(session, new_model_data["config"])

            delta_test_reactive.set(new_model_data["D_TEST"])
            x_test_reactive.set(new_model_data["X_TEST"])
            prob_threshold.set(new_model_data["DEFAULT_THRESHOLD"])
            data_available.set(False)
            patient_selected.set(None)
            pending_use_case.set(None)
            ui.update_select("use_case_selector", selected=new_use_case)

            ui.notification_show(f"Switched to {new_model_data['config'].get('name', new_use_case)}", type="message")

            await session.send_custom_message("toggleActiveTab", {"activeTab": "data_input"})

            ui.modal_remove()
            await session.send_custom_message("setUseCaseLoading", {"visible": False})

        except (UseCaseNotFoundError, Exception) as e:
            ui.notification_show(f"Error loading use case: {e}", type="error")
            pending_use_case.set(None)
            ui.update_select("use_case_selector", selected=current_use_case.get())
            await session.send_custom_message("setUseCaseLoading", {"visible": False})

    input_results = data_input.server("data_input", model_data, DEFAULT_CONFIG, config)
    analysis_data = input_results["data"]
    is_custom_data = input_results["is_custom"]
    is_input_confirmed = input_results["is_confirmed"]

    def _confirmation_required_message():
        labels = config.get().get("labels", {})
        entity = labels.get("entity", {})
        set_lower = entity.get("set_lower", "cohort")
        return f"Confirm the {set_lower} in Data Input first."

    @reactive.Effect
    async def _update_analysis_context():
        current_df = analysis_data.get()
        is_custom = is_custom_data.get()
        input_confirmed = is_input_confirmed.get()
        md = model_data.get()
        cfg = config.get()

        if not input_confirmed or current_df is None or current_df.empty:
            data_available.set(False)
            delta_test_reactive.set(pd.DataFrame())
            x_test_reactive.set(pd.DataFrame())
            patient_selected.set(None)
            await session.send_custom_message("setUseCaseLoading", {"visible": False})
            return

        try:
            if is_custom:
                if cfg.get("use_case_type") == "neoantigen":
                    raw_for_model = current_df.drop(columns=["ID"], errors="ignore")
                    prepared = prepare_neoantigen_features(
                        raw_for_model,
                        md["FEATURE_ORDER_CLEAN"],
                        target=cfg.get("target_column", "Qualitative_Measure"),
                    )
                    current_proc_df = _with_id(prepared.features)
                    scaled_features = _scale_features(
                        prepared.features,
                        md["scaler"],
                        md["FEATURE_ORDER_CLEAN"],
                    )
                    d_test_curr = delta_xai(md.get("delta_formula"), scaled_features, md["FEATURE_ORDER_CLEAN"])
                else:
                    allowed_columns = [f["name"] for f in cfg.get("features", [])]
                    current_proc_df = process_raw_form_data(
                        raw_df=current_df,
                        example_raw_df=md["example_raw_df"],
                        encoding_config=cfg.get("encoding", {}),
                        allowed_columns=allowed_columns
                    )
                    current_proc_df.columns = [col.replace(' ', '_') for col in current_proc_df.columns]
                    d_test_curr = delta_xai(md.get("delta_formula"), current_proc_df, md["FEATURE_ORDER_CLEAN"])
                delta_test_reactive.set(d_test_curr.reset_index(drop=True))
                x_test_reactive.set(current_proc_df.reset_index(drop=True))
            else:
                delta_test_reactive.set(md["D_TEST"].reset_index(drop=True))
                x_test_reactive.set(md["X_TEST"].reset_index(drop=True))

            first_id = int(current_df["ID"].iloc[0]) if "ID" in current_df.columns and not current_df.empty else None
            patient_selected.set(first_id)
            data_available.set(True)
            await session.send_custom_message("setUseCaseLoading", {"visible": False})
        except Exception as e:
            data_available.set(False)
            delta_test_reactive.set(pd.DataFrame())
            x_test_reactive.set(pd.DataFrame())
            patient_selected.set(None)
            ui.notification_show(f"Error preparing analysis context: {e}", type="error")
            await session.send_custom_message("setUseCaseLoading", {"visible": False})

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
    async def _sync_step_lock_state():
        await session.send_custom_message(
            "setAnalysisStepsLocked",
            {"locked": not bool(is_input_confirmed.get())},
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
        if not is_input_confirmed.get() or not data_available():
            ui.notification_show(_confirmation_required_message(), type="warning")
            return
        await session.send_custom_message(
            "toggleActiveTab", {"activeTab": "data_exploration"}
        )

    @reactive.Effect
    @reactive.event(input.tab_prediction)
    async def _():
        if not is_input_confirmed.get() or not data_available():
            ui.notification_show(_confirmation_required_message(), type="warning")
            await session.send_custom_message("setUseCaseLoading", {"visible": False})
            return
        await session.send_custom_message(
            "toggleActiveTab", {"activeTab": "prediction"}
        )
        await session.send_custom_message("setUseCaseLoading", {"visible": False})

    @reactive.Effect
    @reactive.event(input.info_icon)
    def _show_help_modal():
        cfg = config.get()
        labels = cfg.get("labels", {})
        text = labels.get("text", {})
        titles = cfg.get("titles", {})
        features = cfg.get("features", [])
        threshold = float(prob_threshold.get()) if prob_threshold.get() is not None else 0
        positive_label = labels.get("positive_class_label", cfg.get("positive_class", "Positive"))
        negative_label = labels.get("negative_class_label", cfg.get("negative_class", "Negative"))

        m = ui.modal(
            ui.div(
                ui.tags.h4(f"Welcome to {titles.get('app_title', 'XAInyPredictor')}", style="color: #007bff; font-weight: bold; margin-top: 0;"),
                ui.p(cfg.get("description", ""), style="font-style: italic; color: #666;"),
                ui.tags.div(
                    ui.tags.b("Research prototype context"),
                    ui.p(
                        text.get(
                            "info_context",
                            "This app supports patient stratification research by comparing patient-level inputs with a reference cohort and assigning a model-based patient group. Its workflow utility and interpretation should be validated with clinical collaborators.",
                        ),
                        style="margin: 6px 0 0;",
                    ),
                    style="background: #f4f9ff; border-left: 4px solid #007bff; padding: 10px 12px; margin: 10px 0;"
                ),
                ui.hr(),

                ui.row(
                    ui.column(2, ui.h1("1", style="background: #e9ecef; border-radius: 50%; width: 40px; height: 40px; text-align: center; line-height: 40px; font-size: 20px; color: #495057; margin: 0 auto;")),
                    ui.column(10,
                        ui.h5(text.get("info_step_input_title", "Input Patient Data"), style="font-weight: bold; margin-top: 5px;"),
                        ui.p(text.get("info_step_input_prefix", "Go to the "), ui.tags.b(titles.get('tab_data_input', 'Data Input')), text.get("info_step_input_suffix", " tab. You can upload a file or manually enter patient details."))
                    )
                ),
                ui.br(),

                ui.row(
                    ui.column(2, ui.h1("2", style="background: #e9ecef; border-radius: 50%; width: 40px; height: 40px; text-align: center; line-height: 40px; font-size: 20px; color: #495057; margin: 0 auto;")),
                    ui.column(10,
                        ui.h5(text.get("info_step_context_title", "Explore Features"), style="font-weight: bold; margin-top: 5px;"),
                        ui.p(text.get("info_step_context_prefix", "In the "), ui.tags.b(titles.get('tab_data_exploration', 'Cohort Context')), text.get("info_step_context_suffix", " tab, compare patient features against the reference population."))
                    )
                ),
                ui.br(),

                ui.row(
                    ui.column(2, ui.h1("3", style="background: #e9ecef; border-radius: 50%; width: 40px; height: 40px; text-align: center; line-height: 40px; font-size: 20px; color: #495057; margin: 0 auto;")),
                    ui.column(10,
                        ui.h5(text.get("info_step_output_title", "Review Stratification"), style="font-weight: bold; margin-top: 5px;"),
                        ui.p(
                            text.get("info_step_output_prefix", "Click "),
                            ui.tags.b(titles.get('tab_prediction', 'Patient Stratification')),
                            text.get("info_step_output_middle", f" to see the {labels.get('probability_column', 'Stratification Score')} and assigned patient group. "),
                            f"The current decision threshold is {threshold:.3f}. ",
                            text.get("info_step_output_suffix", "Use the charts to understand the patient's profile against the reference cohort."),
                        ),
                        ui.tags.div(
                            ui.tags.span(f"{negative_label}", style="background: #d1e7dd; color: #0f5132; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;"),
                            " = below threshold | ",
                            ui.tags.span(f"{positive_label}", style="background: #f8d7da; color: #842029; padding: 2px 6px; border-radius: 4px; font-size: 0.8em;"),
                            " = at or above threshold",
                            style="margin-top: 5px; font-size: 0.9em;"
                        )
                    )
                ),
                ui.br(),
                ui.tags.div(
                    ui.tags.b("Model summary"),
                    ui.tags.ul(
                        ui.tags.li(f"Use case: {cfg.get('name', 'Patient stratification model')}"),
                        ui.tags.li(f"Target: {cfg.get('target_column', 'target')}"),
                        ui.tags.li(f"Input variables: {len(features)}"),
                        ui.tags.li(f"{text.get('groups_label', 'Patient groups')}: {negative_label} / {positive_label}"),
                    ),
                    style="background: #fff8e6; border: 1px solid #f1d28c; border-radius: 6px; padding: 10px 12px;"
                ),
            ),
            title="Research Prototype Info",
            easy_close=True,
            footer=ui.modal_button("Close"),
            size="l"
        )
        ui.modal_show(m)


# 7. Launch App
# Executable launch commands
app = App(app_ui, server, static_assets=WWW_DIR)
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]
PORT = find_free_port()
def open_browser():
    time.sleep(2)
    webbrowser.open(f"http://127.0.0.1:{PORT}", new=1)
def ensure_stdio_streams():
    if sys.stdin is None:
        sys.stdin = open(os.devnull, "r")
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")


if __name__ == "__main__":
    ensure_stdio_streams()
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=PORT, log_level="critical")
