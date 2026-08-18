import importlib.util
import os
import shutil
import traceback

from config import INPUT_DIRECTORY, OUTPUT_DIRECTORY, INTERMEDIATE_DIRECTORY


def _load_stage(filename: str):
    """Load a numbered pipeline stage module from src/ by file path."""
    path = os.path.join(os.path.dirname(__file__), "src", filename)
    spec = importlib.util.spec_from_file_location(filename[:-3], path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


feature_engineering     = _load_stage("01_feature_engineering.py")
fuel_prediction         = _load_stage("02_fuel_prediction.py")
lof_enrichment          = _load_stage("03_lof_enrichment.py")
eligibility_filter      = _load_stage("04_eligibility_filter.py")
assignment_optimisation = _load_stage("05_assignment_optimisation.py")
post_processing         = _load_stage("06_post_processing.py")

pipeline = [
    ("Feature Engineering",      feature_engineering),
    ("Fuel Prediction",          fuel_prediction),
    ("LoF Enrichment",           lof_enrichment),
    ("Eligibility Filter",       eligibility_filter),
    ("Assignment Optimisation",  assignment_optimisation),
    ("Post-Processing",          post_processing),
]


def save_config() -> None:
    shutil.copy('config.py', f'{OUTPUT_DIRECTORY}/config.py')


if __name__ == '__main__':
    os.makedirs(INPUT_DIRECTORY, exist_ok=True)
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(INTERMEDIATE_DIRECTORY, exist_ok=True)
    save_config()

    for step_name, module in pipeline:
        print(f"\nRunning: {step_name}...")
        try:
            module.main()
            print(f"{step_name} completed")
        except Exception as e:
            print(f"ERROR in {step_name}: {e}")
            print("\nFull traceback:")
            traceback.print_exc()
            break
