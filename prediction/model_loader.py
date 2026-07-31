from pathlib import Path

from catboost import CatBoostRegressor


class ModelLoader:
    """
    Loads all trained CatBoost models from the models directory.

    Each model is stored using its filename (without .cbm)
    as the dictionary key.

    Example:

    target_fuel_level_l.cbm
            ↓
    models["target_fuel_level_l"]
    """

    def __init__(self, models_directory: Path):

        self.models_directory = Path(models_directory)

        self.models = {}

    def load_models(self):

        print("=" * 60)
        print("Loading Trained Models")
        print("=" * 60)

        if not self.models_directory.exists():

            raise FileNotFoundError(
                f"Models directory not found:\n{self.models_directory}"
            )

        model_files = sorted(
            self.models_directory.glob("*.cbm")
        )

        if len(model_files) == 0:

            raise FileNotFoundError(
                f"No CatBoost models found in:\n{self.models_directory}"
            )

        for model_path in model_files:

            model_name = model_path.stem

            try:

                model = CatBoostRegressor()

                model.load_model(model_path)

                self.models[model_name] = model

                print(f"✓ Loaded {model_name}")

            except Exception as error:

                print(f"✗ Failed {model_name}")

                print(error)

        print("-" * 60)

        print(f"Total Models Loaded : {len(self.models)}")

        print("=" * 60)

        return self.models

    def get_model(self, target_name: str):

        return self.models.get(target_name)

    def get_all_models(self):

        return self.models