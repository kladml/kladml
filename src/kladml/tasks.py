from enum import Enum

class TaskType(str, Enum):
    SUPERVISED = "supervised"
    UNSUPERVISED = "unsupervised"
    SELF_SUPERVISED = "self_supervised"
    GENERATIVE = "generative"
    REINFORCEMENT = "reinforcement"

class MLTask(str, Enum):
    """
    Enumeration of supported Machine Learning Tasks.
    
    This must stay in sync with the platform taxonomy seeded in the backend.
    """
    # Supervised - Classification
    IMAGE_CLASSIFICATION = "image_classification"
    TEXT_CLASSIFICATION = "text_classification"
    AUDIO_CLASSIFICATION = "audio_classification"
    TABULAR_CLASSIFICATION = "tabular_classification"
    
    # Supervised - Regression
    REGRESSION = "regression"
    TIMESERIES_FORECASTING = "timeseries_forecasting"
    
    # Unsupervised
    ANOMALY_DETECTION = "anomaly_detection"
    CLUSTERING = "clustering"
    
    # Generative
    TEXT_GENERATION = "text_generation"
    IMAGE_GENERATION = "image_generation"
    
    # Other
    OTHER = "other"
    
    @classmethod
    def has_value(cls, value):
        return value in cls._value2member_map_
