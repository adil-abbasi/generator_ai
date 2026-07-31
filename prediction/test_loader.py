from config import MODEL_DIR

from prediction.model_loader import ModelLoader

loader = ModelLoader(MODEL_DIR)

models = loader.load_models()

print(models.keys())


