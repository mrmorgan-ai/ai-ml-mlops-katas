import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path

from src.pipeline import ExamScorePipeline, ModelArtifact
from config.config import DEFAULT_MODEL_HYPERPARAMETERS, NUMERIC_FEATURES, CATEGORICAL_FEATURES

@pytest.fixture
def sample_exam_data():
    """
    Generate reproducible sample exam data for testing.
        
    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        X: Features DataFrame (500 rows, 11 columns)
        y: Target Series (exam scores)
    """
    # Set seed for reproducibility
    np.random.seed(42)
    n_samples = 500
    
    # Generate numeric features
    study_hours = np.random.uniform(0, 8, n_samples)
    class_attendance = np.random.uniform(40, 100, n_samples)
    sleep_hours = np.random.uniform(4, 10, n_samples)
    age = np.random.randint(17, 25, n_samples)
    
    # Generate categorical features
    genders = np.random.choice(['male', 'female', 'other'], n_samples)
    courses = np.random.choice(
        ['diploma', 'bca', 'b.sc', 'b.tech', 'bba', 'ba', 'b.com'],
        n_samples
    )
    internet_access = np.random.choice(['yes', 'no'], n_samples, p=[0.85, 0.15])
    sleep_quality = np.random.choice(['poor', 'average', 'good'], n_samples)
    study_method = np.random.choice(
        ['coaching', 'online videos', 'self-study', 'group study', 'mixed'],
        n_samples
    )
    facility_rating = np.random.choice(['low', 'medium', 'high'], n_samples)
    exam_difficulty = np.random.choice(['easy', 'moderate', 'hard'], n_samples)
    
    # Create target with LEARNABLE pattern
    # Formula: base + study_effect + attendance_effect + noise
    base_score = 30
    study_effect = study_hours * 5  # Each study hour adds ~5 points
    attendance_effect = (class_attendance - 40) * 0.3  # Attendance bonus
    noise = np.random.normal(0, 8, n_samples)  # Random variation
    
    exam_score = base_score + study_effect + attendance_effect + noise
    exam_score = np.clip(exam_score, 0, 100)  # Keep in valid range
    
    # Build features DataFrame
    X = pd.DataFrame({
        'age': age,
        'study_hours': study_hours,
        'class_attendance': class_attendance,
        'sleep_hours': sleep_hours,
        'gender': genders,
        'course': courses,
        'internet_access': internet_access,
        'sleep_quality': sleep_quality,
        'study_method': study_method,
        'facility_rating': facility_rating,
        'exam_difficulty': exam_difficulty,
    })
    
    # Build target Series
    y = pd.Series(exam_score, name='exam_score')
    
    return X, y


@pytest.fixture
def small_exam_data():
    """
    Create minimal valid data for quick tests.

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series]
        Minimal valid data (3 rows).
    """
    X = pd.DataFrame({
        'age': [20, 21, 22],
        'study_hours': [4.0, 5.0, 6.0],
        'class_attendance': [80.0, 85.0, 90.0],
        'sleep_hours': [7.0, 6.5, 7.5],
        'gender': ['male', 'female', 'other'],
        'course': ['bca', 'b.tech', 'b.sc'],
        'internet_access': ['yes', 'yes', 'no'],
        'sleep_quality': ['good', 'average', 'poor'],
        'study_method': ['self-study', 'coaching', 'mixed'],
        'facility_rating': ['high', 'medium', 'low'],
        'exam_difficulty': ['moderate', 'hard', 'easy'],
    })
    
    y = pd.Series([70.0, 75.0, 80.0], name='exam_score')
    
    return X, y


@pytest.fixture
def trained_pipeline(sample_exam_data):
    """
    Provide a pre-trained pipeline for tests that need it.
    
    Returns
    -------
    ExamScorePipeline
        Trained pipeline ready for predictions.
    """
    X, y = sample_exam_data
    
    # Create pipeline with MLflow DISABLED for faster tests
    pipeline = ExamScorePipeline(enable_mlflow=False)
    pipeline.train(X, y)
    
    return pipeline

class TestPipelineInitialization:
    """
    Tests for pipeline initialization and configuration.
    
    WHAT WE'RE TESTING:
    - Default hyperparameters are used when none provided
    - Custom hyperparameters override defaults
    - Pipeline starts in untrained state
    - Feature lists are properly defined
    - MLflow is configured correctly
    """
    
    def test_default_parameters_used_when_none_provided(self):
        """
        Test that default hyperparameters are applied.
        
        DETERMINISTIC TEST:
        Same input (no params) should always give same output (defaults).
        """
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        # Check all defaults are applied
        assert pipeline.model_params['n_estimators'] == DEFAULT_MODEL_HYPERPARAMETERS['n_estimators']
        assert pipeline.model_params['max_depth'] == DEFAULT_MODEL_HYPERPARAMETERS['max_depth']
        assert pipeline.model_params['random_state'] == DEFAULT_MODEL_HYPERPARAMETERS['random_state']
        assert pipeline.model_params['min_samples_split'] == DEFAULT_MODEL_HYPERPARAMETERS['min_samples_split']
        assert pipeline.model_params['min_samples_leaf'] == DEFAULT_MODEL_HYPERPARAMETERS['min_samples_leaf']
        assert pipeline.model_params['max_features'] == DEFAULT_MODEL_HYPERPARAMETERS['max_features']
        assert pipeline.model_params['n_jobs'] == DEFAULT_MODEL_HYPERPARAMETERS['n_jobs']
    
    def test_custom_parameters_override_defaults(self):
        """
        Test that custom parameters override defaults.
        
        DETERMINISTIC TEST:
        Custom params should appear in model_params.
        Unspecified params should still have defaults.
        """
        custom_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 123
        }
        
        pipeline = ExamScorePipeline(
            model_params=custom_params,
            enable_mlflow=False
        )
        
        # Custom values should be used
        assert pipeline.model_params['n_estimators'] == 50
        assert pipeline.model_params['max_depth'] == 5
        assert pipeline.model_params['random_state'] == 123
        
        # Other defaults should still be present
        assert pipeline.model_params['min_samples_split'] == DEFAULT_MODEL_HYPERPARAMETERS['min_samples_split']
        assert pipeline.model_params['n_jobs'] == DEFAULT_MODEL_HYPERPARAMETERS['n_jobs']
    
    def test_pipeline_not_trained_initially(self):
        """
        Test that pipeline starts in untrained state.
        
        DETERMINISTIC TEST:
        New pipeline should have None for trained components.
        """
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        assert pipeline.pipeline is None, "pipeline should be None before training"
        assert pipeline.feature_names is None, "feature_names should be None before training"
    
    def test_numeric_and_categorical_features_defined(self):
        """
        Test that feature lists are properly defined.
        
        DETERMINISTIC TEST:
        Feature lists should contain expected column names.
        """
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        # Check numeric features
        expected_numeric = NUMERIC_FEATURES
        for feature in expected_numeric:
            assert feature in pipeline.numeric_features, f"Missing numeric feature: {feature}"
        
        # Check categorical features
        expected_categorical = CATEGORICAL_FEATURES
        for feature in expected_categorical:
            assert feature in pipeline.categorical_features, f"Missing categorical feature: {feature}"


class TestPipelineTraining:
    """
    Tests for the training process.
    
    WHAT WE'RE TESTING:
    - Training returns ModelArtifact
    - Artifact contains required fields
    - Training fails appropriately on bad input
    - Feature names are stored correctly
    """
    
    def test_train_returns_model_artifact(self, sample_exam_data):
        """
        Test that training returns a properly structured ModelArtifact.
        
        DETERMINISTIC TEST:
        train() should always return ModelArtifact with correct type.
        """
        X, y = sample_exam_data
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        artifact = pipeline.train(X, y)
        
        # Check return type
        assert isinstance(artifact, ModelArtifact), "Should return ModelArtifact"
        
        # Check required fields are not None
        assert artifact.model is not None, "model should not be None"
        assert artifact.feature_names is not None, "feature_names should not be None"
        assert artifact.version is not None, "version should not be None"
        assert artifact.training_metrics is not None, "training_metrics should not be None"
    
    def test_artifact_contains_required_metrics(self, sample_exam_data):
        """
        Test that artifact contains all required metrics.
        
        DETERMINISTIC TEST:
        Specific metric keys should always be present.
        """
        X, y = sample_exam_data
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        artifact = pipeline.train(X, y)
        
        # Check required metrics exist
        required_metrics = ['train_rmse', 'train_mae', 'train_r2']
        for metric in required_metrics:
            assert metric in artifact.training_metrics, f"Missing metric: {metric}"
            assert isinstance(artifact.training_metrics[metric], float), f"{metric} should be float"
    
    def test_feature_names_stored_correctly(self, sample_exam_data):
        """
        Test that feature names are captured during training.
        
        DETERMINISTIC TEST:
        Stored feature names should match input columns.
        """
        X, y = sample_exam_data
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        artifact = pipeline.train(X, y)
        
        # Feature names should match input columns
        assert set(artifact.feature_names) == set(X.columns), "Feature names should match input"
    
    def test_train_fails_on_empty_data(self):
        """
        Test that training on empty data raises ValueError.
        
        ERROR HANDLING TEST:
        Empty input should raise clear error, not crash silently.
        """
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        with pytest.raises(ValueError, match="empty"):
            pipeline.train(pd.DataFrame(), pd.Series(dtype=float))
    
    def test_train_fails_on_missing_numeric_features(self, small_exam_data):
        """
        Test that training fails if numeric features are missing.
        
        ERROR HANDLING TEST:
        Missing required columns should raise clear error.
        """
        X, y = small_exam_data
        X_missing = X.drop(columns=['study_hours'])  # Remove required column
        
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        with pytest.raises(ValueError, match="Missing numeric features"):
            pipeline.train(X_missing, y)
    
    def test_train_fails_on_missing_categorical_features(self, small_exam_data):
        """
        Test that training fails if categorical features are missing.
        
        ERROR HANDLING TEST:
        Missing required columns should raise clear error.
        """
        X, y = small_exam_data
        X_missing = X.drop(columns=['gender'])  # Remove required column
        
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        with pytest.raises(ValueError, match="Missing categorical features"):
            pipeline.train(X_missing, y)
    
    def test_pipeline_is_fitted_after_training(self, sample_exam_data):
        """
        Test that pipeline object is fitted after training.
        
        DETERMINISTIC TEST:
        After train(), pipeline.pipeline should not be None.
        """
        X, y = sample_exam_data
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        assert pipeline.pipeline is None, "Should be None before training"
        
        pipeline.train(X, y)
        
        assert pipeline.pipeline is not None, "Should not be None after training"


# =============================================================================
# TEST CLASS: Pipeline Prediction
# =============================================================================

class TestPipelinePrediction:
    """
    Tests for the prediction process.
    
    WHAT WE'RE TESTING:
    - Prediction fails before training
    - Prediction fails on missing features
    - Predictions have correct shape
    - Predictions are numpy arrays
    """
    
    def test_predict_fails_before_training(self, small_exam_data):
        """
        Test that predict raises error if called before training.
        
        ERROR HANDLING TEST:
        Calling predict() on untrained pipeline should fail clearly.
        """
        X, _ = small_exam_data
        pipeline = ExamScorePipeline(enable_mlflow=False)  # Not trained!
        
        with pytest.raises(RuntimeError, match="not trained"):
            pipeline.predict(X)
    
    def test_predict_fails_on_missing_numeric_features(self, trained_pipeline, small_exam_data):
        """
        Test that predict fails if required features are missing.
        
        ERROR HANDLING TEST:
        Production data missing columns should raise clear error.
        """
        X, _ = small_exam_data
        X_missing = X.drop(columns=['study_hours'])
        
        with pytest.raises(ValueError, match="Missing numeric features"):
            trained_pipeline.predict(X_missing)
    
    def test_predict_fails_on_missing_categorical_features(self, trained_pipeline, small_exam_data):
        """
        Test that predict fails if categorical features are missing.
        
        ERROR HANDLING TEST:
        Production data missing columns should raise clear error.
        """
        X, _ = small_exam_data
        X_missing = X.drop(columns=['gender'])
        
        with pytest.raises(ValueError, match="Missing categorical features"):
            trained_pipeline.predict(X_missing)
    
    def test_predict_returns_correct_shape(self, trained_pipeline, small_exam_data):
        """
        Test that predictions have correct shape.
        
        DETERMINISTIC TEST:
        Number of predictions should match number of input rows.
        """
        X, _ = small_exam_data
        
        predictions = trained_pipeline.predict(X)
        
        assert len(predictions) == len(X), "Should have one prediction per row"
        assert isinstance(predictions, np.ndarray), "Should return numpy array"
    
    def test_predict_handles_single_row(self, trained_pipeline):
        """
        Test that prediction works for a single row.
        
        EDGE CASE TEST:
        Single-row prediction should work without error.
        """
        single_student = pd.DataFrame({
            'age': [20],
            'study_hours': [5.0],
            'class_attendance': [85.0],
            'sleep_hours': [7.0],
            'gender': ['female'],
            'course': ['b.tech'],
            'internet_access': ['yes'],
            'sleep_quality': ['good'],
            'study_method': ['mixed'],
            'facility_rating': ['high'],
            'exam_difficulty': ['moderate'],
        })
        
        predictions = trained_pipeline.predict(single_student)
        
        assert len(predictions) == 1, "Should return single prediction"


# =============================================================================
# TEST CLASS: Pipeline Evaluation
# =============================================================================

class TestPipelineEvaluation:
    """
    Tests for model evaluation.
    
    WHAT WE'RE TESTING:
    - Evaluate returns required metrics
    - Evaluate fails before training
    - Metric values are valid
    """
    
    def test_evaluate_returns_required_metrics(self, trained_pipeline, sample_exam_data):
        """
        Test that evaluate returns all required metrics.
        
        DETERMINISTIC TEST:
        Specific metric keys should always be present.
        """
        X, y = sample_exam_data
        
        metrics = trained_pipeline.evaluate(X, y)
        
        assert 'rmse' in metrics, "Should include RMSE"
        assert 'mae' in metrics, "Should include MAE"
        assert 'r2' in metrics, "Should include R²"
    
    def test_evaluate_fails_before_training(self, sample_exam_data):
        """
        Test that evaluate raises error if called before training.
        
        ERROR HANDLING TEST:
        Calling evaluate() on untrained pipeline should fail clearly.
        """
        X, y = sample_exam_data
        pipeline = ExamScorePipeline(enable_mlflow=False)  # Not trained!
        
        with pytest.raises(RuntimeError, match="not trained"):
            pipeline.evaluate(X, y)
    
    def test_metrics_are_valid_numbers(self, trained_pipeline, sample_exam_data):
        """
        Test that metric values are valid numbers.
        
        PROPERTY TEST:
        Metrics should be finite numbers, not NaN or Inf.
        """
        X, y = sample_exam_data
        
        metrics = trained_pipeline.evaluate(X, y)
        
        for name, value in metrics.items():
            assert isinstance(value, float), f"{name} should be float"
            assert not np.isnan(value), f"{name} should not be NaN"
            assert not np.isinf(value), f"{name} should not be Inf"
    
    def test_rmse_is_non_negative(self, trained_pipeline, sample_exam_data):
        """
        Test that RMSE is non-negative.
        
        PROPERTY TEST:
        RMSE is a distance metric, must be >= 0.
        """
        X, y = sample_exam_data
        
        metrics = trained_pipeline.evaluate(X, y)
        
        assert metrics['rmse'] >= 0, "RMSE must be non-negative"
        assert metrics['mae'] >= 0, "MAE must be non-negative"


# =============================================================================
# TEST CLASS: Model Behavior
# =============================================================================

class TestModelBehavior:
    """
    Tests for model learning and prediction behavior.
    
    WHAT WE'RE TESTING:
    - Model learns expected patterns (study hours → scores)
    - Predictions are in valid range
    - Model beats baseline (mean predictor)
    - Model is reproducible with same seed
    
    These are STOCHASTIC tests - they test behavior, not exact values.
    """
    
    def test_predictions_are_in_valid_range(self, trained_pipeline, sample_exam_data):
        """
        Test that predictions are in a reasonable range.
        
        PROPERTY TEST:
        Exam scores should be roughly 0-100.
        Allow some slack since RF can extrapolate slightly.
        """
        X, _ = sample_exam_data
        
        predictions = trained_pipeline.predict(X)
        
        # Allow some slack for extrapolation
        assert predictions.min() >= -10, f"Predictions too low: {predictions.min()}"
        assert predictions.max() <= 110, f"Predictions too high: {predictions.max()}"
    
    def test_model_learns_study_hours_pattern(self, sample_exam_data):
        """
        Test that model learns: more study hours → higher scores.
        
        BEHAVIORAL TEST:
        We created data with this pattern, model should learn it.
        """
        X, y = sample_exam_data
        
        pipeline = ExamScorePipeline(enable_mlflow=False)
        pipeline.train(X, y)
        
        # Create two test cases: low vs high study hours
        # Keep everything else EXACTLY the same
        test_data = pd.DataFrame({
            'age': [20, 20],
            'study_hours': [1.0, 7.0],  # Low vs High (only difference!)
            'class_attendance': [80.0, 80.0],
            'sleep_hours': [7.0, 7.0],
            'gender': ['male', 'male'],
            'course': ['bca', 'bca'],
            'internet_access': ['yes', 'yes'],
            'sleep_quality': ['average', 'average'],
            'study_method': ['self-study', 'self-study'],
            'facility_rating': ['medium', 'medium'],
            'exam_difficulty': ['moderate', 'moderate'],
        })
        
        predictions = pipeline.predict(test_data)
        
        low_study_pred = predictions[0]
        high_study_pred = predictions[1]
        
        assert high_study_pred > low_study_pred, (
            f"More study hours should predict higher score. "
            f"Low study (1h): {low_study_pred:.1f}, High study (7h): {high_study_pred:.1f}"
        )
    
    def test_model_learns_attendance_pattern(self, sample_exam_data):
        """
        Test that model learns: higher attendance → higher scores.
        
        BEHAVIORAL TEST:
        We created data with this pattern, model should learn it.
        """
        X, y = sample_exam_data
        
        pipeline = ExamScorePipeline(enable_mlflow=False)
        pipeline.train(X, y)
        
        # Create two test cases: low vs high attendance
        test_data = pd.DataFrame({
            'age': [20, 20],
            'study_hours': [4.0, 4.0],
            'class_attendance': [50.0, 95.0],  # Low vs High (only difference!)
            'sleep_hours': [7.0, 7.0],
            'gender': ['male', 'male'],
            'course': ['bca', 'bca'],
            'internet_access': ['yes', 'yes'],
            'sleep_quality': ['average', 'average'],
            'study_method': ['self-study', 'self-study'],
            'facility_rating': ['medium', 'medium'],
            'exam_difficulty': ['moderate', 'moderate'],
        })
        
        predictions = pipeline.predict(test_data)
        
        assert predictions[1] > predictions[0], (
            f"Higher attendance should predict higher score. "
            f"Low att (50%): {predictions[0]:.1f}, High att (95%): {predictions[1]:.1f}"
        )
    
    def test_model_performs_better_than_baseline(self, sample_exam_data):
        """
        Test that model performs better than predicting the mean.
        
        BEHAVIORAL TEST:
        The simplest baseline is predicting average score for everyone.
        Our model should do better than this naive approach.
        """
        X, y = sample_exam_data
        
        pipeline = ExamScorePipeline(enable_mlflow=False)
        artifact = pipeline.train(X, y)
        
        # Calculate baseline (mean predictor) RMSE
        mean_prediction = y.mean()
        baseline_errors = y - mean_prediction
        baseline_rmse = np.sqrt((baseline_errors ** 2).mean())
        
        # Model should have lower RMSE than baseline
        model_rmse = artifact.training_metrics['train_rmse']
        
        assert model_rmse < baseline_rmse, (
            f"Model RMSE ({model_rmse:.2f}) should be lower than "
            f"baseline RMSE ({baseline_rmse:.2f})"
        )
    
    def test_r2_score_is_positive(self, sample_exam_data):
        """
        Test that R² score is positive (better than mean predictor).
        
        PROPERTY TEST:
        R² interpretation:
        - R² = 1: Perfect predictions
        - R² = 0: As good as predicting mean
        - R² < 0: Worse than predicting mean (bad!)
        """
        X, y = sample_exam_data
        
        pipeline = ExamScorePipeline(enable_mlflow=False)
        artifact = pipeline.train(X, y)
        
        r2 = artifact.training_metrics['train_r2']
        
        assert r2 > 0, f"R² ({r2:.3f}) should be positive"
    
    def test_predictions_reproducible_with_same_seed(self, sample_exam_data):
        """
        Test that same random_state produces identical results.
        
        REPRODUCIBILITY TEST:
        Same data + same seed = same predictions.
        Critical for debugging and comparing experiments.
        """
        X, y = sample_exam_data
        
        # Train two pipelines with same seed
        p1 = ExamScorePipeline(
            model_params={'random_state': 42, 'n_estimators': 10},
            enable_mlflow=False
        )
        p1.train(X, y)
        pred1 = p1.predict(X)
        
        p2 = ExamScorePipeline(
            model_params={'random_state': 42, 'n_estimators': 10},
            enable_mlflow=False
        )
        p2.train(X, y)
        pred2 = p2.predict(X)
        
        # Should be identical
        np.testing.assert_array_almost_equal(
            pred1, pred2,
            err_msg="Same seed should produce identical predictions"
        )
    
    def test_different_seeds_produce_different_results(self, sample_exam_data):
        """
        Test that different seeds produce different models.
        
        SANITY CHECK:
        If different seeds gave same results, randomness isn't working.
        """
        X, y = sample_exam_data
        
        p1 = ExamScorePipeline(
            model_params={'random_state': 42, 'n_estimators': 10},
            enable_mlflow=False
        )
        p1.train(X, y)
        pred1 = p1.predict(X)
        
        p2 = ExamScorePipeline(
            model_params={'random_state': 123, 'n_estimators': 10},
            enable_mlflow=False
        )
        p2.train(X, y)
        pred2 = p2.predict(X)
        
        # Should be different
        assert not np.allclose(pred1, pred2), (
            "Different seeds should produce different results"
        )


# =============================================================================
# TEST CLASS: Feature Importance
# =============================================================================

class TestFeatureImportance:
    """
    Tests for feature importance extraction.
    
    WHAT WE'RE TESTING:
    - Returns DataFrame with correct structure
    - Importances sum to ~1.0
    - All importances are non-negative
    - Fails before training
    """
    
    def test_get_feature_importance_returns_dataframe(self, trained_pipeline):
        """
        Test that feature importance returns a DataFrame.
        
        DETERMINISTIC TEST:
        Return type and structure should be consistent.
        """
        importance = trained_pipeline.get_feature_importance()
        
        assert isinstance(importance, pd.DataFrame), "Should return DataFrame"
        assert 'feature' in importance.columns, "Should have 'feature' column"
        assert 'importance' in importance.columns, "Should have 'importance' column"
    
    def test_importances_sum_to_one(self, trained_pipeline):
        """
        Test that feature importances sum to approximately 1.
        
        PROPERTY TEST:
        RF importances are normalized to sum to 1.0.
        """
        importance = trained_pipeline.get_feature_importance()
        
        total = importance['importance'].sum()
        assert abs(total - 1.0) < 0.01, f"Importances sum to {total}, expected ~1.0"
    
    def test_all_importances_non_negative(self, trained_pipeline):
        """
        Test that all feature importances are non-negative.
        
        PROPERTY TEST:
        Importance is based on error reduction, can't be negative.
        """
        importance = trained_pipeline.get_feature_importance()
        
        assert (importance['importance'] >= 0).all(), (
            "All importances should be >= 0"
        )
    
    def test_feature_importance_fails_before_training(self):
        """
        Test that feature importance raises error if not trained.
        
        ERROR HANDLING TEST:
        Can't get importance from untrained model.
        """
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        with pytest.raises(RuntimeError, match="not trained"):
            pipeline.get_feature_importance()
    
    def test_importance_sorted_descending(self, trained_pipeline):
        """
        Test that importance is sorted highest first.
        
        PROPERTY TEST:
        Makes it easy to see most important features.
        """
        importance = trained_pipeline.get_feature_importance()
        
        # Check if sorted descending
        values = importance['importance'].values
        assert all(values[i] >= values[i+1] for i in range(len(values)-1)), (
            "Importances should be sorted descending"
        )


# =============================================================================
# TEST CLASS: Model Persistence
# =============================================================================

class TestModelPersistence:
    """
    Tests for model saving and loading.
    
    WHAT WE'RE TESTING:
    - Save fails before training
    - Save creates file
    - Load restores working pipeline
    - Loaded model produces identical predictions
    """
    
    def test_save_fails_before_training(self, tmp_path):
        """
        Test that save raises error if pipeline not trained.
        
        ERROR HANDLING TEST:
        Can't save what doesn't exist.
        """
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        with pytest.raises(RuntimeError, match="untrained"):
            pipeline.save(str(tmp_path / "model.joblib"))
    
    def test_save_creates_file(self, trained_pipeline, tmp_path):
        """
        Test that save creates a file on disk.
        
        DETERMINISTIC TEST:
        File should exist after save.
        """
        model_path = tmp_path / "model.joblib"
        
        trained_pipeline.save(str(model_path))
        
        assert model_path.exists(), "Model file should be created"
    
    def test_load_restores_pipeline(self, trained_pipeline, tmp_path, sample_exam_data):
        """
        Test that load restores a working pipeline.
        
        INTEGRATION TEST:
        Loaded pipeline should be able to predict.
        """
        X, _ = sample_exam_data
        model_path = tmp_path / "model.joblib"
        
        # Save
        trained_pipeline.save(str(model_path))
        
        # Load
        loaded = ExamScorePipeline.load(str(model_path))
        
        # Should be able to predict
        predictions = loaded.predict(X)
        assert len(predictions) == len(X), "Should make predictions"
    
    def test_loaded_model_produces_identical_predictions(
        self, trained_pipeline, tmp_path, sample_exam_data
    ):
        """
        Test that loaded model produces identical predictions.
        
        CRITICAL TEST:
        If predictions change after save/load, you can't trust deployment!
        """
        X, _ = sample_exam_data
        model_path = tmp_path / "model.joblib"
        
        # Get predictions before save
        original_predictions = trained_pipeline.predict(X)
        
        # Save and load
        trained_pipeline.save(str(model_path))
        loaded = ExamScorePipeline.load(str(model_path))
        
        # Get predictions after load
        loaded_predictions = loaded.predict(X)
        
        # Must be identical
        np.testing.assert_array_almost_equal(
            original_predictions,
            loaded_predictions,
            decimal=10,
            err_msg="Loaded model predictions must match original"
        )
    
    def test_loaded_model_has_correct_configuration(self, trained_pipeline, tmp_path):
        """
        Test that loaded model has correct feature configuration.
        
        DETERMINISTIC TEST:
        Configuration should be preserved through save/load.
        """
        model_path = tmp_path / "model.joblib"
        
        trained_pipeline.save(str(model_path))
        loaded = ExamScorePipeline.load(str(model_path))
        
        # Check configuration preserved
        assert loaded.numeric_features == trained_pipeline.numeric_features
        assert loaded.categorical_features == trained_pipeline.categorical_features
        assert loaded.feature_names == trained_pipeline.feature_names


# =============================================================================
# TEST CLASS: MLflow Integration
# =============================================================================

class TestMLflowIntegration:
    """
    Tests for MLflow experiment tracking.
    
    WHAT WE'RE TESTING:
    - MLflow run is created
    - Parameters are logged
    - Metrics are logged
    - Artifacts are logged
    """
    
    @pytest.mark.slow
    def test_training_creates_mlflow_run(self, sample_exam_data, tmp_path):
        """
        Test that training creates an MLflow run.
        
        INTEGRATION TEST:
        MLflow run should be created and ID stored.
        """
        X, y = sample_exam_data
        
        # Use temp directory for MLflow
        mlflow_dir = str(tmp_path / "mlruns")
        
        pipeline = ExamScorePipeline(
            experiment_name="test_experiment",
            tracking_uri=mlflow_dir,
            enable_mlflow=True
        )
        
        artifact = pipeline.train(X, y)
        
        # Should have run ID
        assert artifact.mlflow_run_id is not None, "Should have MLflow run ID"
    
    @pytest.mark.slow
    def test_training_logs_parameters(self, sample_exam_data, tmp_path):
        """
        Test that hyperparameters are logged to MLflow.
        
        INTEGRATION TEST:
        Logged params should match what we set.
        """
        import mlflow
        
        X, y = sample_exam_data
        mlflow_dir = str(tmp_path / "mlruns")
        
        custom_params = {'n_estimators': 50, 'max_depth': 5}
        
        pipeline = ExamScorePipeline(
            model_params=custom_params,
            experiment_name="test_params",
            tracking_uri=mlflow_dir,
            enable_mlflow=True
        )
        
        artifact = pipeline.train(X, y)
        
        # Verify by querying MLflow
        mlflow.set_tracking_uri(mlflow_dir)
        run = mlflow.get_run(artifact.mlflow_run_id)
        
        assert run.data.params['n_estimators'] == '50', "n_estimators should be logged"
        assert run.data.params['max_depth'] == '5', "max_depth should be logged"
    
    @pytest.mark.slow
    def test_training_logs_metrics(self, sample_exam_data, tmp_path):
        """
        Test that training metrics are logged to MLflow.
        
        INTEGRATION TEST:
        RMSE, MAE, R² should be logged.
        """
        import mlflow
        
        X, y = sample_exam_data
        mlflow_dir = str(tmp_path / "mlruns")
        
        pipeline = ExamScorePipeline(
            experiment_name="test_metrics",
            tracking_uri=mlflow_dir,
            enable_mlflow=True
        )
        
        artifact = pipeline.train(X, y)
        
        # Verify by querying MLflow
        mlflow.set_tracking_uri(mlflow_dir)
        run = mlflow.get_run(artifact.mlflow_run_id)
        
        assert 'train_rmse' in run.data.metrics, "train_rmse should be logged"
        assert 'train_mae' in run.data.metrics, "train_mae should be logged"
        assert 'train_r2' in run.data.metrics, "train_r2 should be logged"


# =============================================================================
# TEST CLASS: End-to-End Workflow
# =============================================================================

class TestEndToEndWorkflow:
    """
    Integration tests for complete ML workflow.
    
    WHAT WE'RE TESTING:
    - Complete train → predict → evaluate workflow
    - Workflow with custom hyperparameters
    """
    
    def test_complete_train_predict_evaluate_workflow(self, sample_exam_data):
        """
        Test the complete ML workflow from start to finish.
        
        INTEGRATION TEST:
        All components should work together.
        """
        X, y = sample_exam_data
        
        # Split data
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Initialize pipeline
        pipeline = ExamScorePipeline(enable_mlflow=False)
        
        # Train
        artifact = pipeline.train(X_train, y_train)
        assert artifact is not None, "Training should return artifact"
        
        # Predict
        predictions = pipeline.predict(X_test)
        assert len(predictions) == len(X_test), "Should predict for all test samples"
        
        # Evaluate
        metrics = pipeline.evaluate(X_test, y_test)
        assert metrics['rmse'] > 0, "RMSE should be positive"
        assert -1 <= metrics['r2'] <= 1, "R² should be in [-1, 1]"
    
    def test_workflow_with_custom_hyperparameters(self, sample_exam_data):
        """
        Test workflow with custom hyperparameters.
        
        INTEGRATION TEST:
        Custom params should affect training.
        """
        X, y = sample_exam_data
        
        # Use custom parameters
        custom_params = {
            'n_estimators': 50,
            'max_depth': 5,
            'random_state': 123
        }
        
        pipeline = ExamScorePipeline(
            model_params=custom_params,
            enable_mlflow=False
        )
        
        artifact = pipeline.train(X, y)
        
        # Should work correctly
        predictions = pipeline.predict(X)
        assert len(predictions) == len(X)
        
        # Verify custom params were used
        assert pipeline.model_params['n_estimators'] == 50
        assert pipeline.model_params['max_depth'] == 5
    
    def test_full_persistence_workflow(self, sample_exam_data, tmp_path):
        """
        Test complete save → load → predict workflow.
        
        INTEGRATION TEST:
        Model should work identically after save/load cycle.
        """
        X, y = sample_exam_data
        model_path = tmp_path / "full_workflow_model.joblib"
        
        # Train and save
        pipeline = ExamScorePipeline(enable_mlflow=False)
        pipeline.train(X, y)
        original_preds = pipeline.predict(X)
        pipeline.save(str(model_path))
        
        # Load and predict (simulating new session)
        loaded = ExamScorePipeline.load(str(model_path))
        loaded_preds = loaded.predict(X)
        
        # Should be identical
        np.testing.assert_array_almost_equal(original_preds, loaded_preds)
        
        # Should be able to evaluate
        metrics = loaded.evaluate(X, y)
        assert metrics['rmse'] > 0
