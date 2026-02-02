DEFAULT_MODEL_HYPERPARAMETERS = {
            "n_estimators":100, # number of trees in the forest
            "max_depth":10, # maximum depth in each tree
            "min_samples_split": 5, # minimum samples to split a node
            "max_features": 0.7, # features considered at each split. 70% features considered
            "random_state":42, 
            "n_jobs": -1, # parallel processin. use all cpus avilable
        }

NUMERIC_FEATURES = [
        'age',              # Student age in years (17-24)
        'study_hours',      # Daily study hours (0-8)
        'class_attendance', # Attendance percentage (40-100)
        'sleep_hours',      # Hours of sleep per night (4-10)
    ]

CATEGORICAL_FEATURES = [
        'gender',           # male, female, other
        'course',           # diploma, bca, b.sc, b.tech, bba, ba, b.com
        'internet_access',  # yes, no
        'sleep_quality',    # poor, average, good
        'study_method',     # coaching, online videos, self-study, group study, mixed
        'facility_rating',  # low, medium, high
        'exam_difficulty',  # easy, moderate, hard
    ]

DATA_PATH = r"C:\Users\jhoni\Documents\LooperAI\repositorios\ai-ml-mlops-katas\data\raw\Exam_Score_Prediction.csv"

ID_PATTERNS = ['_id', 'id_', 'student_id', 'index', 'key']
