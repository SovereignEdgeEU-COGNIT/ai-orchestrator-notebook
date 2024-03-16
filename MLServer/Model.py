from MLModelInterface import MLModelInterface


class SchedulerModel(MLModelInterface):
    def __init__(self):
        self.models = {}  # A dictionary to store models
        self.current_model = None

    def initialize(self, models_info):
        # Load models into memory based on provided models_info
        pass

    def set_model(self, model_name):
        # Set the current model based on model_name
        self.current_model = self.models.get(model_name)

    def predict(self, data):
        # Make a prediction using the current model
        if self.current_model is not None:
            return "Scheduler prediction"
        else:
            return "No model set"

class ClassifierModel(MLModelInterface):
    def __init__(self):
        self.models = {}
        self.current_model = None

    def initialize(self, models_info):
        # Similar to SchedulerModel, load classifier models
        pass

    def set_model(self, model_name):
        self.current_model = self.models.get(model_name)

    def predict(self, data):
        if self.current_model is not None:
            return "Classifier prediction"
        else:
            return "No model set"
