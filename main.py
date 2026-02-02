import pandas as pd
import numpy as np
import pandera as pa
from pandera.errors import SchemaError

from src.pipelines.preprocessing import MLDataPreprocessor
from src.monitoring.data_validation import validate_exam_data
from src.monitoring.drift_detection import check_categorical_drif, check_numerical_drift
from src.pipelines.pipeline import ExamScorePipeline

from config.config import DATA_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES

def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)
    
def main():    
    # Initialize loader
    print("EXAM SCORE PREDICTION - START EXECUTION")
    
    # =========================================================================
    # STEP 1: LOAD DATA
    # =========================================================================
    print_section("LOAD DATA")
    data_processor = MLDataPreprocessor(DATA_PATH)
    df_raw = data_processor.load_data()

    # Validate raw data
    try:
        validated = validate_exam_data(df_raw)
        print("Validation succeed!!")
    except SchemaError as e:
        print(f"Validation failed: {e}")
    
    # =========================================================================
    # STEP 2: EXPLORE DATA
    # =========================================================================
    print_section("EXPLORE DATA")
    print("\nFirst 5 rows:")
    print(df_raw.head().to_string())
    
    print(f"\nTarget variable (exam_score):")
    print(f"Range:  {df_raw['exam_score'].min():.1f} - {df_raw['exam_score'].max():.1f}")
    print(f"Mean:   {df_raw['exam_score'].mean():.1f}")
    print(f"Std:    {df_raw['exam_score'].std():.1f}")
    
    # =========================================================================
    # STEP 3: CHECK MISSING VALUES
    # =========================================================================
    print_section("CHECK MISSING VALUES")
    
    missing = data_processor.check_missing_values(df_raw)
    if len(missing) == 0:
        print("No missing values found!")
    else:
        print("Missing values detected:")
        print(missing)
    
    # =========================================================================
    # STEP 4: PREPROCESS
    # =========================================================================
    print_section("PREPROCESS DATA")
    
    # Remove ID columns
    df_clean = data_processor.remove_id_columns(df_raw)
    removed = set(df_clean.columns) - set(df_clean.columns)
    print(f"Removed ID columns: {removed}")
    
    # Remove high cardinality
    df_clean = data_processor.remove_high_cardinality(df_clean)
    
    # Split features and target
    X, y = data_processor.split_features_and_target(df_clean, 'exam_score')
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    
    # =========================================================================
    # STEP 5: TRAIN/TEST SPLIT
    # =========================================================================
    print_section("TRAIN/TEST SPLIT")
    
    # Split train, val, test datasets
    X_train, y_train, X_val, y_val, X_test, y_test = data_processor.train_test_split(
        X,y,test_size=0.2, 
        val_size=0.2
    )
    
    print(f"Training set: {len(X_train):,} samples (80%)")
    print(f"Test set:     {len(X_test):,} samples (20%)")
    
    # =========================================================================
    # STEP 6: CHECK DATA DRIFT
    # =========================================================================
    print_section("CHECK DATA DRIFT")
    drift_report = check_numerical_drift(X_train, X_test, numeric_columns=NUMERIC_FEATURES)
    
    print(f"\nNumeric feature drift (PSI):")
    for col, results in drift_report.items():
        status = "DRIFT" if results['drift_detected_psi'] else "OK"
        print(f"{col}: PSI={results['psi']:.4f} [{status}]")

    print("Preprocessing complete!!\n")
    print(f"Found {X_train.shape[0]} registers in train dataset")
    print(f"Found {X_val.shape[0]} registers in train dataset")
    print(f"Found {X_test.shape[0]} registers in train dataset")
    
    # =========================================================================
    # STEP 7: TRAIN MODEL WITH MLFLOW
    # =========================================================================
    print_section("TRAIN MODEL WITH MLFLOW TRACKING")

    # Initialize pipeline with MLflow
    pipeline = ExamScorePipeline(
        model_params={
            'n_estimators': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'random_state': 42,
            'n_jobs': -1
        },
        experiment_name="exam_score_prediction",
        tracking_uri="mlruns",  # Local folder
        enable_mlflow=True
    )
    
    print("\nTraining model...")
    artifact = pipeline.train(
        X_train, y_train,
        run_name="baseline_model",
        tags={"version": "1.0", "author": "mrmorgan.ai"}
    )
    
    print(f"\nTraining complete!")
    print(f"MLflow Run ID: {artifact.mlflow_run_id}")
    print(f"\nTraining Metrics:")
    print(f"RMSE: {artifact.training_metrics['train_rmse']:.2f} points")
    print(f"MAE:  {artifact.training_metrics['train_mae']:.2f} points")
    print(f"R²:   {artifact.training_metrics['train_r2']:.4f} ({artifact.training_metrics['train_r2']*100:.1f}% variance explained)")
    
    # =========================================================================
    # STEP 8: EVALUATE ON TEST SET
    # =========================================================================
    print_section("EVALUATE ON TEST SET")
    
    test_metrics = pipeline.evaluate(X_test, y_test, dataset_name="test")
    
    print(f"\nTest Metrics:")
    print(f"RMSE: {test_metrics['rmse']:.2f} points")
    print(f"MAE:  {test_metrics['mae']:.2f} points")
    print(f"R²:   {test_metrics['r2']:.4f} ({test_metrics['r2']*100:.1f}% variance explained)")
    
    # Compare train vs test
    train_rmse = artifact.training_metrics['train_rmse']
    test_rmse = test_metrics['rmse']
    gap_pct = (test_rmse - train_rmse) / train_rmse * 100
    
    print(f"\nTrain vs Test Comparison:")
    print(f"Train RMSE: {train_rmse:.2f} --- Test RMSE: {test_rmse:.2f} ({gap_pct:+.1f}%)")
    
    if gap_pct > 30:
        print("Warning: Large gap suggests overfitting")
    elif gap_pct > 15:
        print("Moderate gap - acceptable generalization")
    else:
        print("Good generalization!")
    
    # =========================================================================
    # STEP 9: FEATURE IMPORTANCE
    # =========================================================================
    print_section("FEATURE IMPORTANCE")
    
    importance = pipeline.get_feature_importance()
    
    print("\nTop 10 most important features:")
    for i, row in importance.head(10).iterrows():
        bar = "*" * int(row['importance'] * 40) #"█"
        print(f"{row['feature']:<25} {row['importance']:.3f} {bar}")
    
    print(f"\nKey insight: '{importance.iloc[0]['feature']}' is the strongest predictor")
    print(f"Top 3 features explain {importance.head(3)['importance'].sum()*100:.1f}% of model decisions")
    
    # =========================================================================
    # STEP 10: PREDICT NEW STUDENTS
    # =========================================================================
    print_section("PREDICT NEW STUDENTS")
    
    new_students = pd.DataFrame({
        'age': [19, 22, 18, 21],
        'study_hours': [2.0, 6.5, 1.0, 8.0],
        'class_attendance': [60.0, 95.0, 50.0, 98.0],
        'sleep_hours': [8.0, 6.5, 9.0, 5.0],
        'gender': ['female', 'male', 'other', 'female'],
        'course': ['bca', 'b.tech', 'diploma', 'b.sc'],
        'internet_access': ['yes', 'yes', 'no', 'yes'],
        'sleep_quality': ['good', 'average', 'poor', 'average'],
        'study_method': ['online videos', 'mixed', 'self-study', 'coaching'],
        'facility_rating': ['medium', 'high', 'low', 'high'],
        'exam_difficulty': ['moderate', 'hard', 'easy', 'hard'],
    })
    
    print("\nNew students to predict:")
    print("Student 1: Studies 2h/day, 60% attendance")
    print("Student 2: Studies 6.5h/day, 95% attendance")
    print("Student 3: Studies 1h/day, 50% attendance")
    print("Student 4: Studies 8h/day, 98% attendance")
    
    predictions = pipeline.predict(new_students)
    
    print("\nPredicted exam scores:")
    for i, pred in enumerate(predictions, 1):
        print(f"  Student {i}: {pred:.1f} points")
    
    # =========================================================================
    # STEP 11: SAVE AND LOAD MODEL
    # =========================================================================
    print_section("SAVE AND LOAD MODEL")
    
    model_path = "exam_score_model.joblib"
    
    pipeline.save(model_path)
    print(f"Model saved to '{model_path}'")
    
    loaded = ExamScorePipeline.load(model_path)
    loaded_preds = loaded.predict(new_students)
    
    if np.allclose(predictions, loaded_preds):
        print("Loaded model produces identical predictions!")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print_section("SUMMARY")
    
    print(f"""
    Dataset: {len(df_clean):,} students
    Features: {len(X.columns)} ({len(pipeline.numeric_features)} numeric, {len(pipeline.categorical_features)} categorical)
    
    Model Performance:
    ├── Training RMSE: {train_rmse:.2f} points
    ├── Test RMSE:     {test_rmse:.2f} points
    └── Test R²:       {test_metrics['r2']:.4f}
    
    Top 3 Predictive Features:
    ├── {importance.iloc[0]['feature']}: {importance.iloc[0]['importance']*100:.1f}%
    ├── {importance.iloc[1]['feature']}: {importance.iloc[1]['importance']*100:.1f}%
    └── {importance.iloc[2]['feature']}: {importance.iloc[2]['importance']*100:.1f}%
    
    MLflow:
    ├── Experiment: exam_score_prediction
    ├── Run ID: {artifact.mlflow_run_id}
    └── View UI: mlflow ui --port 5000
    
    Model saved to: {model_path}
    """)
    
    print("=" * 70)
    print(" Demo Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
