from shiny import run_app
from XAInyPredictor.app import app

def main():
    run_app(
        app,
        host="127.0.0.1",
        port=8001,
        launch_browser=True,
    )
