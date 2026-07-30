from pathlib import Path

# Root folder of the project
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Data folder
DATA_DIR = PROJECT_ROOT / "data"

# Models folder
MODELS_DIR = PROJECT_ROOT / "models"

# Logs folder
LOGS_DIR = PROJECT_ROOT / "logs"

# Evaluation folder
EVALUATION_DIR = PROJECT_ROOT / "evaluation"

# Create folders automatically
MODELS_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
EVALUATION_DIR.mkdir(exist_ok=True)