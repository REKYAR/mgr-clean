import os
import pprint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter, deque
from tqdm import tqdm
import json
import glob
import warnings
import zipfile
import argparse
from typing import List, Dict, Tuple, Optional, Union, Set, Any
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

np.random.seed(27)
import random
random.seed(27)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_)):
            return bool(obj)
        elif isinstance(obj, (set, tuple)):
            return list(obj)
        return super(NumpyEncoder, self).default(obj)


from river import metrics
from river import neighbors
    


class MonotonicityConstraint:
    """
    Defines a monotonicity constraint on a specific feature.
    """
    def __init__(self, feature, monotonicity):
        """
        Initialize a monotonicity constraint.
        
        Parameters:
        -----------
        feature : int or str
            Feature index or name on which to apply the constraint
        monotonicity : int
            Direction of monotonicity: 1 for positive, -1 for negative
        """
        self.feature = feature
        self.monotonicity = monotonicity  # 1 for positive, -1 for negative
    
    def __str__(self):
        direction = "positive" if self.monotonicity == 1 else "negative"
        return f"Feature {self.feature}, {direction} monotonicity"
    
    def __repr__(self):
        return self.__str__()


class EnhancedKNNMClassifierV2:
    def __init__(self, classes, k=10, constraints=[], window_size=100, 
                 constraint_weight=0.7, age_weight=0.2, random_weight=0.1,
                 minimum_new_sample_quota=0.05, constraint_threshold=0.8,
                 constraint_prediction_weight=0.5):
        """
        Enhanced KNN classifier with constraint-aware predictions.
        
        Parameters:
        -----------
        classes : list
            List of possible classes
        k : int
            Number of neighbors for KNN
        constraints : list
            List of MonotonicityConstraint objects
        window_size : int
            Fixed window size to maintain
        constraint_weight : float
            Weight for constraint satisfaction in replacement score (0-1)
        age_weight : float
            Weight for age in replacement score (0-1)
        random_weight : float
            Weight for randomness in replacement score (0-1)
        minimum_new_sample_quota : float
            Minimum percentage of window that should be reserved for new samples
        constraint_threshold : float
            Minimum constraint satisfaction threshold for preferential treatment
        constraint_prediction_weight : float
            Weight given to constraint-based prediction (0-1) vs standard KNN prediction
        """
        self.classes = classes
        self.k = k
        self.constraints = constraints
        self.window_size = window_size
        self.constraint_weight = constraint_weight
        self.age_weight = age_weight
        self.random_weight = random_weight
        self.minimum_new_sample_quota = minimum_new_sample_quota
        self.constraint_threshold = constraint_threshold
        self.constraint_prediction_weight = constraint_prediction_weight
        
        self.window = []  # List of (X, y, age, constraint_satisfaction) tuples
        
        self.classifier = neighbors.KNNClassifier(n_neighbors=self.k)
        
        self.counter = 0
        self.total_samples_seen = 0
        self.constraint_violations_avoided = 0
        self.constraint_violations_in_predictions = 0
        self.list_sizes_history = []
        self.max_list_size_history = []
        self.avg_list_size_history = []
        self.num_lists_history = []
        
    def _calculate_constraint_satisfaction(self, sample, sample_class):
        """
        Calculate how well a sample satisfies constraints with existing window.
        Returns a score between 0 and 1.
        """
        if not self.constraints or not self.window:
            return 1.0  # No constraints or empty window
        
        total_checks = 0
        satisfied_checks = 0
        
        for x, y, _, _ in self.window:
            for constraint in self.constraints:
                feature = constraint.feature
                monotonicity = constraint.monotonicity
                
                if feature not in sample or feature not in x:
                    continue
                
                total_checks += 1
                
                index_sample_class = self.classes.index(sample_class)
                index_y = self.classes.index(y)
                
                if monotonicity == 1:
                    if ((index_y > index_sample_class and x[feature] >= sample[feature]) or
                        (index_y < index_sample_class and x[feature] <= sample[feature]) or
                        (index_y == index_sample_class)):
                        satisfied_checks += 1
                
                elif monotonicity == -1:
                    if ((index_y > index_sample_class and x[feature] <= sample[feature]) or
                        (index_y < index_sample_class and x[feature] >= sample[feature]) or
                        (index_y == index_sample_class)):
                        satisfied_checks += 1
        

        if total_checks == 0:
            return 1.0
            
        return satisfied_checks / total_checks
        
    def _update_ages(self):
        """Increment the age of all samples in the window."""
        self.window = [(x, y, age + 1, cs) for x, y, age, cs in self.window]
    
    def _calculate_replacement_scores(self):
        """
        Calculate replacement scores for all samples in the window.
        Higher score means more likely to be replaced.
        """
        scores = []
        
        max_age = max([age for _, _, age, _ in self.window]) if self.window else 1
        
        for i, (_, _, age, constraint_satisfaction) in enumerate(self.window):
            normalized_age = age / max_age if max_age > 0 else 0
            
            random_component = np.random.random()
            
            constraint_score = 1 - constraint_satisfaction  # Higher score for lower satisfaction
            age_score = normalized_age  # Higher score for older samples
            
            new_sample_quota = int(self.window_size * self.minimum_new_sample_quota)
            is_oldest = sorted(range(len(self.window)), key=lambda j: self.window[j][2], reverse=True)[:new_sample_quota]
            
            if i in is_oldest:
                final_score = 0.9 + (0.1 * random_component)
            else:
                final_score = (self.constraint_weight * constraint_score +
                              self.age_weight * age_score +
                              self.random_weight * random_component)
            
            scores.append(final_score)
        
        return scores
    
    def _update_constraint_satisfactions(self):
        """Update the constraint satisfaction scores for all samples in the window."""
        updated_window = []
        
        for i, (x, y, age, _) in enumerate(self.window):
            constraint_satisfaction = self._calculate_constraint_satisfaction(x, y)
            updated_window.append((x, y, age, constraint_satisfaction))
        
        self.window = updated_window
    
    def _would_violate_constraints(self, X, predicted_class):
        """
        Check if the predicted class would violate monotonicity constraints.
        Returns True if it would violate constraints, False otherwise.
        """
        if not self.constraints:
            return False
            
        for x, y, _, _ in self.window:
            for constraint in self.constraints:
                feature = constraint.feature
                monotonicity = constraint.monotonicity
                
                if feature not in X or feature not in x:
                    continue
                
                index_pred_class = self.classes.index(predicted_class)
                index_y = self.classes.index(y)
                
                if monotonicity == 1:  # Positive monotonicity
                    if ((index_pred_class > index_y and X[feature] < x[feature]) or
                        (index_pred_class < index_y and X[feature] > x[feature])):
                        return True
                
                elif monotonicity == -1:  # Negative monotonicity
                    if ((index_pred_class > index_y and X[feature] > x[feature]) or
                        (index_pred_class < index_y and X[feature] < x[feature])):
                        return True
        
        return False
    
    def _get_constraint_based_prediction(self, X):
        """
        Generate a prediction based on monotonicity constraints.
        Returns the class that best satisfies constraints.
        """
        if not self.constraints or not self.window:
            return self.classifier.predict_one(X)
        class_satisfaction = {}
        for cls in self.classes:
            satisfaction = self._calculate_constraint_satisfaction(X, cls)
            class_satisfaction[cls] = satisfaction
        return max(class_satisfaction.items(), key=lambda x: x[1])[0]
    
    def learn_one(self, X, y):
        """
        Learn from one sample with constraint-aware approach.
        
        Parameters:
        -----------
        X : dict
            Features of the sample
        y : any
            Target class
        
        Returns:
        --------
        self
        """
        self.counter += 1
        self.total_samples_seen += 1
        
        self._update_ages()
        
        constraint_satisfaction = self._calculate_constraint_satisfaction(X, y)
        
    
        if len(self.window) < self.window_size:
            self.window.append((X, y, 0, constraint_satisfaction))
            self.classifier.learn_one(X, y)
        
        elif constraint_satisfaction < self.constraint_threshold:
            replacement_scores = self._calculate_replacement_scores()
            worst_idx = np.argmax(replacement_scores)
            worst_score = replacement_scores[worst_idx]
            
            new_score = self.constraint_weight * (1 - constraint_satisfaction)
            
            if new_score < worst_score:
                self.window[worst_idx] = (X, y, 0, constraint_satisfaction)
                self.classifier = neighbors.KNNClassifier(n_neighbors=self.k)
                for x, y, _, _ in self.window:
                    self.classifier.learn_one(x, y)
            else:
                self.constraint_violations_avoided += 1
            

        
        else:
            replacement_scores = self._calculate_replacement_scores()
            replace_idx = np.argmax(replacement_scores)
            self.window[replace_idx] = (X, y, 0, constraint_satisfaction)

            self.classifier = neighbors.KNNClassifier(n_neighbors=self.k)
            for x, y, _, _ in self.window:
                self.classifier.learn_one(x, y)
        
        if self.counter % 10 == 0:
            self._update_constraint_satisfactions()
        
        current_size = len(self.window)
        self.list_sizes_history.append([current_size])
        self.max_list_size_history.append(current_size)
        self.avg_list_size_history.append(current_size)
        self.num_lists_history.append(1)
        
        return self
    
    def predict_one(self, X):
        """
        Predict with constraint-awareness.
        The prediction is a weighted combination of:
        1. Standard KNN prediction 
        2. Constraint-based prediction
        """
        if not self.window:
            return self.classes[0]
        
        return self.classifier.predict_one(X)
        
    
    def predict_proba_one(self, X):
        """
        Predict probabilities with constraint-awareness.
        """
        if not self.window:
            return {cls: 1 / len(self.classes) for cls in self.classes}
        
        knn_probs = self.classifier.predict_proba_one(X)

        return knn_probs
        

    
    def get_stats(self):
        """Get statistics about the classifier."""
        if not self.window:
            return {
                "window_size": 0,
                "avg_age": 0,
                "avg_constraint_satisfaction": 0,
                "total_samples_seen": self.total_samples_seen,
                "constraint_violations_avoided": self.constraint_violations_avoided,
                "constraint_violations_in_predictions": self.constraint_violations_in_predictions
            }
            
        avg_age = np.mean([age for _, _, age, _ in self.window])
        avg_cs = np.mean([cs for _, _, _, cs in self.window])
        
        return {
            "window_size": len(self.window),
            "avg_age": avg_age,
            "avg_constraint_satisfaction": avg_cs,
            "total_samples_seen": self.total_samples_seen,
            "constraint_violations_avoided": self.constraint_violations_avoided,
            "constraint_violations_in_predictions": self.constraint_violations_in_predictions
        }
    


class UCIDatasetLoader:
    """
    Loader for UCI datasets with monotonicity constraints for experimentation.
    
    This class handles loading UCI datasets from a directory structure,
    applying appropriate preprocessing, and extracting features and targets
    according to specified monotonicity constraints.
    """
    
    def __init__(self, data_dir: str, constraints_file: Optional[str] = None):
        """
        Initialize the UCI dataset loader.
        
        Parameters:
        -----------
        data_dir : str
            Path to the directory containing UCI datasets
        constraints_file : str, optional
            Path to a CSV file containing constraint definitions
        """
        self.data_dir = data_dir
        self.default_constraints = self._define_default_constraints()
        
        self.custom_constraints = {}
        if constraints_file and os.path.exists(constraints_file):
            self._load_constraints_file(constraints_file)
    
    def _load_constraints_file(self, constraints_file: str):
        """Load constraints from CSV file with support for nested constraints."""
        try:
            constraints_df = pd.read_csv(constraints_file)
            custom_constraints = {}
            
            for _, row in constraints_df.iterrows():
                dataset = row['Dataset'].strip()
                
                positive_features = self._parse_list_field(row['Positive Monotonic Features'])
                negative_features = self._parse_list_field(row['Negative Monotonic Features'])
                
                custom_constraints[dataset] = {
                    'target': row['Target'].strip(),
                    'positive_monotonic': positive_features,
                    'negative_monotonic': negative_features,
                    'notes': row['Notes'] if 'Notes' in row else ''
                }
            
            self.custom_constraints = custom_constraints
        except Exception as e:
            warnings.warn(f"Error loading constraints file: {e}")
    
    def _parse_list_field(self, list_str):
        """
        Parse a string representation of a list into an actual list.
        This updated version handles nested lists.
        
        Examples:
        ---------
        "['a1', 'a2', ['a1', 'a2']]" -> ['a1', 'a2', ['a1', 'a2']]
        """
        if pd.isna(list_str) or not list_str:
            return []
        
        if isinstance(list_str, str):
            try:
                import ast
                return ast.literal_eval(list_str)
            except (SyntaxError, ValueError):
                cleaned = list_str.strip('[]').replace('"', '').replace("'", "")
                if cleaned:
                    items = []
                    for item in cleaned.split(','):
                        item = item.strip()
                        if item.startswith('[') and item.endswith(']'):
                            nested_items = item[1:-1].split()
                            items.append([i.strip() for i in nested_items])
                        else:
                            items.append(item)
                    return items
        
        return []
    
    def _define_default_constraints(self) -> Dict:
        """Define default monotonicity constraints for known UCI datasets."""
        return {
            "monks_1": {
                "target": "class",
                "positive_monotonic": ["a1", "a2", "a3", ["a1", "a2", "a3"]],
                "negative_monotonic": ["a4", "a5", "a6", ["a4", "a5", "a6"]],
                "notes": "Binary classification with logical rules"
            },
            "monks_2": {
                "target": "class",
                "positive_monotonic": ["a1", "a2", "a3", ["a1", "a2", "a3"]],
                "negative_monotonic": ["a4", "a5", "a6", ["a4", "a5", "a6"]],
                "notes": "Binary classification with logical rules"
            },
            "monks_3": {
                "target": "class",
                "positive_monotonic": ["a1", "a2", "a3", ["a1", "a2", "a3"]],
                "negative_monotonic": ["a4", "a5", "a6", ["a4", "a5", "a6"]],
                "notes": "Binary classification with logical rules"
            },
            "mammographic_mass": {
                "target": "Severity",
                "positive_monotonic": ["Age", ["Age"]],
                "negative_monotonic": ["Density", ["Density"]],
                "notes": "Binary classification, 1=benign, 0=malignant"
            },
            "iris": {
                "target": "Species",
                "positive_monotonic": ["petal length", "petal width", ["petal length", "petal width"]],
                "negative_monotonic": ["sepal width", ["sepal width"]],
                "notes": "Multiclass classification, ordering: setosa < versicolor < virginica"
            },
            "hepatitis": {
                "target": "Class",
                "positive_monotonic": ["Age", ["Age"]],
                "negative_monotonic": ["Albumin", ["Albumin"]],
                "notes": "Binary classification, 1=died, 0=lived (reversed for monotonicity)"
            },
            "wine": {
                "target": "class",
                "positive_monotonic": ["Alcohol", "Flavanoids", ["Alcohol", "Flavanoids"]],
                "negative_monotonic": ["Ash", ["Ash"]],
                "notes": "Multiclass classification, ordered by quality"
            },
            "heart_disease": {
                "target": "num",
                "positive_monotonic": ["thalach", ["thalach"]],  # Max Heart Rate
                "negative_monotonic": ["age", "chol", ["age", "chol"]],  # Age, Cholesterol
                "notes": "Binary classification, 0=no disease, 1=disease"
            },
            "adult": {
                "target": "income",
                "positive_monotonic": ["education-num", "hours-per-week", ["education-num", "hours-per-week"]],
                "negative_monotonic": ["age", ["age"]],
                "notes": "Binary classification, '>50K'-1 vs '<=50K'-0" 
            },
            "student_performance": {
                "target": "G3",  # Final grade
                "positive_monotonic": ["studytime", ["studytime"]],
                "negative_monotonic": ["absences", ["absences"]],
                "notes": "Regression/ordinal, final grade"
            },
            "diabetes": {
                "target": "Outcome",
                "positive_monotonic": ["Glucose", "BMI", ["Glucose", "BMI"]],
                "negative_monotonic": ["Age", ["Age"]],
                "notes": "Binary classification, 1=diabetes, 0=no diabetes"
            },
            "spambase": {
                "target": "class",
                "positive_monotonic": ["word_freq_free", "word_freq_money", ["word_freq_free", "word_freq_money"]], 
                "negative_monotonic": ["word_freq_hp", "word_freq_george", ["word_freq_hp", "word_freq_george"]],
                "notes": "Binary classification, 1=spam, 0=not spam"
            },
            "thoracic_surgery": {
                "target": "Risk1Yr",
                "positive_monotonic": ["FEV1", ["FEV1"]],
                "negative_monotonic": ["Age", ["Age"]],
                "notes": "Binary classification, 1=dead, 0=alive"
            },
            "air_quality": {
                "target": "CO_GT",  # CO level (discretized)
                "positive_monotonic": ["NOx", ["NOx"]],
                "negative_monotonic": ["O3", ["O3"]],
                "notes": "Regression/ordinal target"
            },
            "automobile": {
                "target": "price",
                "positive_monotonic": ["engine-size", ["engine-size"]],
                "negative_monotonic": ["highway-mpg", ["highway-mpg"]],
                "notes": "Regression/ordinal target"
            },
            "auto_mpg": {
                "target": "mpg",
                "positive_monotonic": [],
                "negative_monotonic": ["weight", "horsepower", ["weight", "horsepower"]],
                "notes": "Regression target (higher is better, so negative monotonicity)"
            },
            "ionosphere": {
                "target": "class",
                "positive_monotonic": [],
                "negative_monotonic": [],
                "notes": "Binary classification, 1=good, 0=bad"
            },
            "london": {
                "target": "travel_mode",
                "positive_monotonic": ["car_ownership", "distance", "driving_license", "pt_interchanges", ["car_ownership", "distance", "driving_license", "pt_interchanges"]],
                "negative_monotonic": ["car_ownership", "distance", "driving_license", "pt_interchanges", ["car_ownership", "distance", "driving_license", "pt_interchanges"]],
                "notes": "Binary classification, 1=car 0=else"
            },
            "students":{
                "target": "Car",  # Final grade
                "positive_monotonic": ["Distance","Season","CarAvail",["Distance","Season","CarAvail"]],
                "negative_monotonic": ["Age","Grade","Gender", ["Age","Grade","Gender"]],
                "notes": "Regression/ordinal, final grade"
            },
            "flights":{
                "target": "Cancelled",
                "positive_monotonic": ["Distance","Origin_Latitude","Month",["Distance","Airline","FlightNum"]],
                "negative_monotonic": ["Distance","Origin_Latitude",["DayOfWeek","Month","Year"]],
                "notes": "Binary classification, 1=cancelled, 0=not cancelled"
            },
            "ohio":{
                "target": "label",
                "positive_monotonic": ["drivers","workers","guest",[]],
                "negative_monotonic": ["income","students",[]],
                "notes": "Binary classification, 1=sole driver, 0=not a sole driver"
            }
        }
    
    def _get_constraints(self, dataset_name: str) -> Dict:
        """
        Get constraints for a specific dataset, prioritizing custom constraints if available.
        
        Parameters:
        -----------
        dataset_name : str
            Name of the dataset
            
        Returns:
        --------
        Dict with target and monotonicity constraints
        """
        for key in self.custom_constraints:
            if dataset_name.lower() in key.lower():
                return self.custom_constraints[key]
        print(f"Running {dataset_name}")
        if "monk" in dataset_name.lower():
            if "_1" in dataset_name or "-1" in dataset_name:
                return self.default_constraints["monks_1"]
            elif "_2" in dataset_name or "-2" in dataset_name:
                return self.default_constraints["monks_2"]
            elif "_3" in dataset_name or "-3" in dataset_name:
                return self.default_constraints["monks_3"]
            else:
                print(f"Using monks_1 constraints for {dataset_name}")
                return self.default_constraints["monks_1"]
        
        for key in self.default_constraints:
            if dataset_name.lower() in key.lower() or key.lower() in dataset_name.lower():
                print(f"Found match: {key} for {dataset_name}")
                return self.default_constraints[key]
        
        warnings.warn(f"No constraints found for dataset: {dataset_name}")
        return {
            "target": None,
            "positive_monotonic": [],
            "negative_monotonic": [],
            "notes": "No constraints defined"
        }
    
    def load_datasets(self, max_datasets: Optional[int] = None, 
                     dataset_filter: Optional[List[str]] = None) -> Dict[str, Tuple[pd.DataFrame, Dict]]:
        """
        Load UCI datasets with constraints.
        
        Parameters:
        -----------
        max_datasets : int, optional
            Maximum number of datasets to load
        dataset_filter : List[str], optional
            List of dataset names to load (if None, load all)
            
        Returns:
        --------
        Dict of {dataset_name: (DataFrame, constraints_dict)}
        """
        print(f"Loading datasets from {self.data_dir}")
        dataset_dirs = [d for d in os.listdir(self.data_dir) 
                      if os.path.isdir(os.path.join(self.data_dir, d))]
        print(f"Found datasets: {dataset_dirs}")
        if dataset_filter:
            dataset_dirs = [d for d in dataset_dirs 
                          if any(filter_name.lower() in d.lower() for filter_name in dataset_filter)]
        print(f"Filtered datasets: {dataset_dirs}")
        if max_datasets:
            dataset_dirs = dataset_dirs[:max_datasets]
        
        loaded_datasets = {}
        for dataset_dir in dataset_dirs:
            try:
                normalized_name = self._normalize_dataset_name(dataset_dir)
                df, constraints = self._load_specific_dataset(normalized_name)
                
                if df is not None:
                    loaded_datasets[normalized_name] = (df, constraints)
            except Exception as e:
                warnings.warn(f"Error loading dataset {dataset_dir}: {e}")
        
        return loaded_datasets
    
    def _normalize_dataset_name(self, dataset_name: str) -> str:
        """Normalize dataset name for consistent referencing."""
        name = dataset_name.replace('+', '_')
        name = name.replace(' ', '_')
        return name.lower()
    
    def _load_specific_dataset(self, dataset_name: str) -> Tuple[Optional[pd.DataFrame], Dict]:
        """
        Load a specific dataset with specialized handling based on dataset name.
        
        Parameters:
        -----------
        dataset_name : str
            Normalized name of the dataset
            
        Returns:
        --------
        Tuple of (DataFrame, constraints_dict)
        """
        constraints = self._get_constraints(dataset_name)
        
        if "monk" in dataset_name:
            return self._load_monks_dataset(dataset_name), constraints
        elif "mammographic_mass" in dataset_name:
            return self._load_mammographic_mass(), constraints
        elif "iris" in dataset_name:
            return self._load_iris(), constraints
        elif "hepatitis" in dataset_name:
            return self._load_hepatitis(), constraints
        elif "wine" in dataset_name:
            return self._load_wine(), constraints
        elif "heart_disease" in dataset_name:
            return self._load_heart_disease(), constraints
        elif "adult" in dataset_name:
            return self._load_adult(), constraints
        elif "student_performance" in dataset_name:
            return self._load_student_performance(), constraints
        elif "diabetes" in dataset_name:
            return self._load_diabetes(), constraints
        elif "spambase" in dataset_name:
            return self._load_spambase(), constraints
        elif "thoracic_surgery" in dataset_name:
            return self._load_thoracic_surgery(), constraints
        elif "air_quality" in dataset_name:
            return self._load_air_quality(), constraints
        elif "automobile" in dataset_name:
            return self._load_automobile(), constraints
        elif "auto_mpg" in dataset_name:
            return self._load_auto_mpg(), constraints
        elif "ionosphere" in dataset_name:
            return self._load_ionosphere(), constraints
        elif "london" in dataset_name:
            return self._load_london(), constraints
        elif "students" in dataset_name:
            return self._load_students(), constraints
        elif "flights" in dataset_name:
            return self._load_flights(), constraints
        elif "ohio" in dataset_name:
            return self._load_ohio(), constraints
        else:
            print(f"Loading generic dataset: {dataset_name}")
            try:
                original_dir = dataset_name.replace('_', '+')
                dataset_path = os.path.join(self.data_dir, original_dir)
                
                csv_files = glob.glob(os.path.join(dataset_path, "*.csv"))
                data_files = glob.glob(os.path.join(dataset_path, "*.data"))
                
                if csv_files:
                    df = pd.read_csv(csv_files[0])
                elif data_files:
                    df = pd.read_csv(data_files[0], header=None)
                else:
                    warnings.warn(f"No suitable data file found for {dataset_name}")
                    return None, constraints
                
                return df, constraints
            except Exception as e:
                warnings.warn(f"Error loading dataset {dataset_name}: {e}")
                return None, constraints
    
    def _extract_zip_if_needed(self, dataset_name: str) -> None:
        """Extract zip file if the dataset directory doesn't exist."""
        original_dir = dataset_name.replace('_', '+')
        dataset_path = os.path.join(self.data_dir, original_dir)
        
        if not os.path.exists(dataset_path):
            zip_path = os.path.join(self.data_dir, f"{original_dir}.zip")
            if os.path.exists(zip_path):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(dataset_path)
    
    def _load_monks_dataset(self, dataset_name: str) -> pd.DataFrame:
        """Load Monks Problems dataset."""
        self._extract_zip_if_needed("monk_s_problems")
        
        monks_num = None
        for i in range(1, 4):
            if f"monk_{i}" in dataset_name:
                monks_num = i
                break
        
        if monks_num is None:
            monks_num = 1  # Default to Monk 1
        
        dataset_path = os.path.join(self.data_dir, "monk+s+problems")
        train_file = os.path.join(dataset_path, f"monks-{monks_num}.train")
        
        if not os.path.exists(train_file):
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file == f"monks-{monks_num}.train":
                        train_file = os.path.join(root, file)
                        break
        
        columns = ['class', 'a1', 'a2', 'a3', 'a4', 'a5', 'a6', 'Id']
        df = pd.read_csv(train_file, sep=' ', names=columns)
        
        if 'Id' in df.columns:
            df = df.drop('Id', axis=1)
        
        return df
    
    def _load_mammographic_mass(self) -> pd.DataFrame:
        """Load Mammographic Mass dataset."""
        self._extract_zip_if_needed("mammographic_mass")
        
        dataset_path = os.path.join(self.data_dir, "mammographic+mass")
        data_file = os.path.join(dataset_path, "mammographic_masses.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for mammographic mass")
        
        columns = ['BI-RADS', 'Age', 'Shape', 'Margin', 'Density', 'Severity']
        df = pd.read_csv(data_file, na_values=['?'], names=columns)
        
        df = df.dropna()
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _load_iris(self) -> pd.DataFrame:
        """Load Iris dataset."""
        self._extract_zip_if_needed("iris")
        
        dataset_path = os.path.join(self.data_dir, "iris")
        data_file = os.path.join(dataset_path, "iris.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for iris")
        
        columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'Species']
        df = pd.read_csv(data_file, names=columns)
        
        species_mapping = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
        df['Species'] = df['Species'].map(species_mapping)
        
        return df
    
    def _load_hepatitis(self) -> pd.DataFrame:
        """Load Hepatitis dataset."""
        self._extract_zip_if_needed("hepatitis")
        
        dataset_path = os.path.join(self.data_dir, "hepatitis")
        data_file = os.path.join(dataset_path, "hepatitis.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for hepatitis")
        
        columns = [
            'Class', 'Age', 'Sex', 'Steroid', 'Antivirals', 'Fatigue', 'Malaise', 
            'Anorexia', 'Liver Big', 'Liver Firm', 'Spleen Palpable', 'Spiders', 
            'Ascites', 'Varices', 'Bilirubin', 'Alk Phosphate', 'SGOT', 'Albumin', 
            'Protime', 'Histology'
        ]
        
        df = pd.read_csv(data_file, na_values=['?'], names=columns)
        
        df = df.dropna()
        
        for col in df.columns:
            if col != 'Class':
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['Class'] = df['Class'].map({2: 0, 1: 1})
        
        return df
    
    def _load_wine(self) -> pd.DataFrame:
        """Load Wine dataset."""
        self._extract_zip_if_needed("wine")
        
        dataset_path = os.path.join(self.data_dir, "wine")
        data_file = os.path.join(dataset_path, "wine.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for wine")
        
        columns = [
            'class', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 
            'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavanoid phenols', 
            'Proanthocyanins', 'Color intensity', 'Hue', 'OD280/OD315 of diluted wines', 
            'Proline'
        ]
        
        df = pd.read_csv(data_file, names=columns)
        
        df['class'] = df['class'] - 1
        
        return df
    
    def _load_heart_disease(self) -> pd.DataFrame:
        """Load Heart Disease dataset."""
        self._extract_zip_if_needed("heart_disease")
        
        dataset_path = os.path.join(self.data_dir, "heart+disease")
        
        data_file = None
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if "processed.cleveland.data" in file:
                    data_file = os.path.join(root, file)
                    break
        
        if not data_file:
            data_files = glob.glob(os.path.join(dataset_path, "**/*.data"), recursive=True)
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for heart disease")
        
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach',
            'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num'
        ]
        
        df = pd.read_csv(data_file, na_values=['?'], names=columns)
        
        df = df.dropna()
        
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['num'] = df['num'].apply(lambda x: 0 if x == 0 else 1)
        
        return df
    
    def _load_ionosphere(self) -> pd.DataFrame:
        """Load Ionosphere dataset."""
        self._extract_zip_if_needed("ionosphere")
        
        dataset_path = os.path.join(self.data_dir, "ionosphere")
        data_file = os.path.join(dataset_path, "ionosphere.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for ionosphere")
        
        columns = [f'feature_{i}' for i in range(1, 35)] + ['class']
        df = pd.read_csv(data_file, names=columns)
        
        df['class'] = df['class'].map({'g': 1, 'b': 0})
        
        print(df.head())
        
        return df
    

    def _load_london(self) -> pd.DataFrame:
        """Load London dataset."""
        self._extract_zip_if_needed("london")
        
        dataset_path = os.path.join(self.data_dir, "london")
        data_file = os.path.join(dataset_path, "london_data_extended_by_mg.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for london")
        
        df = pd.read_csv(data_file)




        df['travel_mode'] = df['travel_mode'].apply(lambda x: 1 if x == 4 else 0)
        
        return df
    
    def _load_adult(self) -> pd.DataFrame:
        """Load Adult dataset."""
        self._extract_zip_if_needed("adult")
        
        dataset_path = os.path.join(self.data_dir, "adult")
        data_file = os.path.join(dataset_path, "adult.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for adult")
        
        columns = [
            'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status',
            'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss',
            'hours-per-week', 'native-country', 'income'
        ]
        
        df = pd.read_csv(data_file, names=columns, sep=', ', engine='python')
        
        df['income'] = df['income'].map({'<=50K': 0, '>50K': 1})
        
        categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 
                           'relationship', 'race', 'sex', 'native-country']
        
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def _load_student_performance(self) -> pd.DataFrame:
        """Load Student Performance dataset."""
        self._extract_zip_if_needed("student_performance")
        
        dataset_path = os.path.join(self.data_dir, "student+performance")
        
        data_file = os.path.join(dataset_path, "student-mat.csv")
        
        if not os.path.exists(data_file):
            data_file = os.path.join(dataset_path, "student-por.csv")
            
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for student performance")
        
        df = pd.read_csv(data_file, sep=';')
        
        categorical_cols = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 
                           'reason', 'guardian', 'schoolsup', 'famsup', 'paid', 'activities',
                           'nursery', 'higher', 'internet', 'romantic']
        
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def _load_diabetes(self) -> pd.DataFrame:
        """Load Pima Indians Diabetes dataset."""
        data_file = os.path.join(self.data_dir, "diabetes.csv")
        
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"Cannot find diabetes.csv file")
        
        df = pd.read_csv(data_file)
        
        return df
    
    def _load_spambase(self) -> pd.DataFrame:
        """Load Spambase dataset."""
        self._extract_zip_if_needed("spambase")
        
        dataset_path = os.path.join(self.data_dir, "spambase")
        data_file = os.path.join(dataset_path, "spambase.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for spambase")
        
        word_freq_cols = [f"word_freq_{w}" for w in [
            'make', 'address', 'all', '3d', 'our', 'over', 'remove', 'internet', 'order',
            'mail', 'receive', 'will', 'people', 'report', 'addresses', 'free', 'business',
            'email', 'you', 'credit', 'your', 'font', '000', 'money', 'hp', 'hpl', 'george',
            'lab', 'labs', 'telnet', '857', 'data', '415', '85', 'technology', '1999', 'parts',
            'pm', 'direct', 'cs', 'meeting', 'original', 'project', 're', 'edu', 'table',
            'conference'
        ]]
        
        char_freq_cols = [f"char_freq_{c}" for c in [';', '(', '[', '!', '$', '#']]
        capital_run_cols = ['capital_run_length_average', 'capital_run_length_longest', 'capital_run_length_total']
        
        columns = word_freq_cols + char_freq_cols + capital_run_cols + ['class']
        
        df = pd.read_csv(data_file, names=columns)
        
        return df
    
    def _load_thoracic_surgery(self) -> pd.DataFrame:
        """Load Thoracic Surgery dataset."""
        self._extract_zip_if_needed("thoracic_surgery_data")
        
        dataset_path = os.path.join(self.data_dir, "thoracic+surgery+data")
        data_file = os.path.join(dataset_path, "ThoraricSurgery.arff")
        
        if not os.path.exists(data_file):
            for ext in ['.arff', '.data', '.csv']:
                files = glob.glob(os.path.join(dataset_path, f"*{ext}"))
                if files:
                    data_file = files[0]
                    break
            else:
                raise FileNotFoundError(f"Cannot find data file for thoracic surgery")
        
        if data_file.endswith('.arff'):
            data_lines = []
            columns = []
            with open(data_file, 'r') as f:
                data_section = False
                for line in f:
                    line = line.strip()
                    if line.lower().startswith('@attribute'):
                        parts = line.split()
                        if len(parts) > 1:
                            attr_name = parts[1].strip("'")
                            columns.append(attr_name)
                    elif line.lower().startswith('@data'):
                        data_section = True
                    elif data_section and line and not line.startswith('%'):
                        data_lines.append(line)
            
            df = pd.DataFrame([line.split(',') for line in data_lines], columns=columns)
            print(df.head())
            
            binary_cols = [ 'PRE7', 'PRE8', 'PRE9', 'PRE10', 
                          'PRE11',  'PRE17', 'PRE19', 'PRE25', 'PRE30', 'PRE32']
            for col in binary_cols:
                if col in df.columns:
                    df[col] = df[col].map({'T': 1, 'F': 0})
            
            multivalue_cols = ['DGN', 'PRE6','PRE14']
            df["DGN"] = df["DGN"].map({'DGN1': 0, 'DGN2': 1, 'DGN3': 2, 'DGN4': 3, 'DGN5': 4, 'DGN6': 5, 'DGN8': 6})
            df["PRE6"] = df["PRE6"].map({'PRZ2': 0, 'PRZ1': 1, 'PRZ0': 2})
            df["PRE14"] = df["PRE14"].map({'OC11': 0, 'OC14': 1, 'OC12': 2, 'OC13': 3})
            
            numeric_cols = ['AGE','PRE4', 'PRE5']
            
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if 'Risk1Yr' in df.columns:
                df['Risk1Yr'] = df['Risk1Yr'].map({'T': 1, 'F': 0})
            elif 'Class' in df.columns:
                df['Risk1Yr'] = df['Class'].map({'T': 1, 'F': 0})
                df = df.drop('Class', axis=1)
        else:
            df = pd.read_csv(data_file)
        
        return df
    
    def _load_air_quality(self) -> pd.DataFrame:
        """Load Air Quality dataset."""
        self._extract_zip_if_needed("air_quality")
        
        dataset_path = os.path.join(self.data_dir, "air+quality")
        data_file = os.path.join(dataset_path, "AirQualityUCI.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for air quality")
        
        df = pd.read_csv(data_file, sep=';', decimal=',')
        print(f"Loaded air quality data from {data_file}")
        print(df.head())
        
        if 'Date' in df.columns:
            df = df.drop('Date', axis=1)
        if 'Time' in df.columns:
            df = df.drop('Time', axis=1)
        
        for col in df.columns:
            if df[col].dtype in [np.float64, np.int64]:
                df[col] = df[col].replace(-200, np.nan)
        
        if 'CO(GT)' in df.columns:
            df = df.dropna(subset=['CO(GT)'])
        
        rename_dict = {
            'CO(GT)': 'CO_GT',
            'PT08.S1(CO)': 'PT08_S1_CO',
            'NMHC(GT)': 'NMHC_GT',
            'C6H6(GT)': 'C6H6_GT',
            'PT08.S2(NMHC)': 'PT08_S2_NMHC',
            'NOx(GT)': 'NOx',
            'PT08.S3(NOx)': 'PT08_S3_NOx',
            'NO2(GT)': 'NO2',
            'PT08.S4(NO2)': 'PT08_S4_NO2',
            'PT08.S5(O3)': 'PT08_S5_O3',
            'T': 'Temperature',
            'RH': 'RelativeHumidity',
            'AH': 'AbsoluteHumidity'
        }
        
        df = df.rename(columns={c: rename_dict.get(c, c) for c in df.columns})

        unnamed_cols = [col for col in df.columns if 'Unnamed:' in col]
        if unnamed_cols:
            df = df.drop(unnamed_cols, axis=1)
        
        if 'CO_GT' in df.columns:
            df['CO_GT'] = df['CO_GT'].apply(lambda x: 1 if x > 2 else 0)
        
        return df
    
    def _load_automobile(self) -> pd.DataFrame:
        """Load Automobile dataset."""
        self._extract_zip_if_needed("automobile")
        
        dataset_path = os.path.join(self.data_dir, "automobile")
        data_file = os.path.join(dataset_path, "imports-85.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for automobile")
        
        columns = [
            'symboling', 'normalized-losses', 'make', 'fuel-type', 'aspiration', 
            'num-of-doors', 'body-style', 'drive-wheels', 'engine-location', 'wheel-base', 
            'length', 'width', 'height', 'curb-weight', 'engine-type', 'num-of-cylinders', 
            'engine-size', 'fuel-system', 'bore', 'stroke', 'compression-ratio', 'horsepower', 
            'peak-rpm', 'city-mpg', 'highway-mpg', 'price'
        ]
        
        df = pd.read_csv(data_file, na_values=['?'], names=columns)
        
        numeric_cols = ['symboling', 'normalized-losses', 'wheel-base', 'length', 'width', 
                       'height', 'curb-weight', 'engine-size', 'bore', 'stroke', 
                       'compression-ratio', 'horsepower', 'peak-rpm', 'city-mpg', 
                       'highway-mpg', 'price']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna(subset=['price'])  # Ensure target variable is present
        
        categorical_cols = ['make', 'fuel-type', 'aspiration', 'num-of-doors', 'body-style',
                           'drive-wheels', 'engine-location', 'engine-type', 'num-of-cylinders',
                           'fuel-system']
        
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def _load_auto_mpg(self) -> pd.DataFrame:
        """Load Auto MPG dataset."""
        self._extract_zip_if_needed("auto_mpg")
        
        dataset_path = os.path.join(self.data_dir, "auto+mpg")
        data_file = os.path.join(dataset_path, "auto-mpg.data")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.data"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for auto mpg")
        
        columns = [
            'mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
            'acceleration', 'model year', 'origin', 'car name'
        ]
        
        df = pd.read_fwf(data_file, names=columns, na_values=['?'])
        
        numeric_cols = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight',
                       'acceleration', 'model year', 'origin']
        
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df.dropna()
        
        if 'car name' in df.columns:
            df = df.drop('car name', axis=1)
        
        return df
    

    def _load_students(self) -> pd.DataFrame:
        """Load Students dataset."""
        self._extract_zip_if_needed("students")
        
        dataset_path = os.path.join(self.data_dir, "students")
        data_file = os.path.join(dataset_path, "students_binarized.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for students")
        
        df = pd.read_csv(data_file, sep=',')
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        df = df.dropna()
        
        return df


    def _load_ohio(self) -> pd.DataFrame:
        """Load Students dataset."""
        self._extract_zip_if_needed("ohio")
        
        dataset_path = os.path.join(self.data_dir, "ohio")
        data_file = os.path.join(dataset_path, "AllOhioDataSorted_cleaned.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for students")
            

        
        df = pd.read_csv(data_file, sep=',')
        if 'ID' in df.columns:
            df = df.drop('ID', axis=1)
        
        df = df.dropna()

        if 'ML_region' in df.columns:
            df = df.drop('ML_region', axis=1)
        
        return df
    

    def _load_flights(self) -> pd.DataFrame:
        self._extract_zip_if_needed("flights")
        dataset_path = os.path.join(self.data_dir, "flights")
        data_file = os.path.join(dataset_path, "flights_small.csv")
        
        if not os.path.exists(data_file):
            data_files = glob.glob(os.path.join(dataset_path, "*.csv"))
            if data_files:
                data_file = data_files[0]
            else:
                raise FileNotFoundError(f"Cannot find data file for students")
        
        df = pd.read_csv(data_file, sep=',')
        print(f"Loaded flights data from {data_file}")

        df['Cancelled'] = df['Cancelled'].apply(lambda x: 1 if x else 0)
        if 'FlightDate' in df.columns:
            df = df.drop('FlightDate', axis=1)
        if 'Diverted' in df.columns:
            df = df.drop('Diverted', axis=1)

        categorical_cols = ['Airline', 'Origin', 'Dest', 'Marketing_Airline_Network', 
                           'Operated_or_Branded_Code_Share_Partners', 'IATA_Code_Marketing_Airline', 'Operating_Airline', 'IATA_Code_Operating_Airline',
                           'OriginCityName', 'OriginState', 'OriginStateName', 'DestCityName',
                           'DestState', 'DestStateName', 'DepTimeBlk', 'ArrTimeBlk',]
        
        for col in categorical_cols:
            df[col] = df[col].astype('category').cat.codes
        
        return df
    
    def prepare_dataset_for_experiment(self, df: pd.DataFrame, 
                                      constraints: Dict,
                                      normalize: bool = True) -> Tuple[pd.DataFrame, Dict]:
        """
        Prepare a dataset for experimentation by extracting features, target,
        and formatting the constraints for the KNNMClassifier.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input DataFrame
        constraints : Dict
            Dictionary with target and monotonicity constraints
        normalize : bool
            Whether to normalize numeric features
            
        Returns:
        --------
        Tuple of (prepared_df, formatted_constraints_dict)
        """

        print(f"Preparing dataset with constraints")
        print(df.head())

        if df is None or constraints is None:
            return None, {}
        
        print(f"Constraints: {constraints}")
        target_col = constraints.get('target')
        print(f"Target column: {target_col}")
        if not target_col or target_col not in df.columns:
            for col in df.columns:
                print(f"Checking column: {col}, target_col: {target_col}")
                if col.lower() == target_col.lower():
                    target_col = col
                    break
            else:
                warnings.warn(f"Target column '{target_col}' not found in DataFrame")
                return None, {}
        
        pos_monotonic = constraints.get('positive_monotonic', [])
        neg_monotonic = constraints.get('negative_monotonic', [])
        
        valid_pos_monotonic = []
        valid_neg_monotonic = []
        
        for feature in pos_monotonic:
            if isinstance(feature, list):
                group = []
                for f in feature:
                    if f in df.columns:
                        group.append(f)
                    else:
                        for col in df.columns:
                            if col.lower() == f.lower():
                                group.append(col)
                                break
                if group:  # Only add if at least one feature in group is valid
                    valid_pos_monotonic.append(group)
            elif feature in df.columns:
                valid_pos_monotonic.append(feature)
            else:
                for col in df.columns:
                    if col.lower() == feature.lower():
                        valid_pos_monotonic.append(col)
                        break
                else:
                    warnings.warn(f"Positive monotonic feature '{feature}' not found in DataFrame")
        
        for feature in neg_monotonic:
            if isinstance(feature, list):
                group = []
                for f in feature:
                    if f in df.columns:
                        group.append(f)
                    else:
                        for col in df.columns:
                            if col.lower() == f.lower():
                                group.append(col)
                                break
                if group:  # Only add if at least one feature in group is valid
                    valid_neg_monotonic.append(group)
            elif feature in df.columns:
                valid_neg_monotonic.append(feature)
            else:
                for col in df.columns:
                    if col.lower() == feature.lower():
                        valid_neg_monotonic.append(col)
                        break
                else:
                    warnings.warn(f"Negative monotonic feature '{feature}' not found in DataFrame")
        
        prepared_df = df.copy()

        print(f"Prepared DataFrame with {len(prepared_df)} rows and {len(prepared_df.columns)} columns")
        print(df.head())
        prepared_df = prepared_df.dropna()

        print(f"Prepared DataFrame with {len(prepared_df)} rows and {len(prepared_df.columns)} columns")
        
        if normalize:
            print("Normalizing numeric features")
            scaler = MinMaxScaler()
            numeric_cols = prepared_df.select_dtypes(include=['float', 'int']).columns
            
            if target_col in numeric_cols:
                numeric_cols = [col for col in numeric_cols if col != target_col]
            
            if len(numeric_cols) > 0:
                prepared_df[numeric_cols] = scaler.fit_transform(prepared_df[numeric_cols])
        
        formatted_constraints = {
            'target': target_col,
            'positive_monotonic': valid_pos_monotonic,
            'negative_monotonic': valid_neg_monotonic
        }
        
        return prepared_df, formatted_constraints
    
    def convert_to_experiment_format(self, df: pd.DataFrame, 
                                    constraints: Dict) -> Tuple[List[Dict], List, List[Dict]]:
        """
        Convert DataFrame and constraints to a format compatible with the KNNMClassifier.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Prepared DataFrame
        constraints : Dict
            Dictionary with target and monotonicity constraints
            
        Returns:
        --------
        Tuple of (data_stream, classes, monotonicity_constraints)
        """
        if df is None or constraints is None:
            return [], [], []
        
        target_col = constraints.get('target')
        if not target_col or target_col not in df.columns:
            warnings.warn(f"Target column '{target_col}' not found in DataFrame")
            return [], [], []
        
        print(df.head())
        
        features = df.drop(columns=[target_col]).reset_index(drop=True)
        feature_names = features.columns.tolist()
        target = df[target_col].reset_index(drop=True)
        
        data_stream = []
        for i, row in features.iterrows():
            X = {j: row[feature_names[j]] for j in range(len(feature_names))}
            y = target.iloc[i]
            data_stream.append((X, y))
        
        print(f"Data stream created with {len(data_stream)} samples")
        
        classes = sorted(target.unique().tolist())

        print(f"Classes found: {classes}")
        
        monotonicity_constraints = []
        
        for feature in constraints.get('positive_monotonic', []):
            if isinstance(feature, list):  # Handle feature groups
                for f in feature:
                    if f in features.columns:
                        feature_idx = features.columns.get_loc(f)
                        monotonicity_constraints.append({'feature': feature_idx, 'monotonicity': 1})
            elif feature in features.columns:  # Handle single features
                feature_idx = features.columns.get_loc(feature)
                monotonicity_constraints.append({'feature': feature_idx, 'monotonicity': 1})
        
        for feature in constraints.get('negative_monotonic', []):
            if isinstance(feature, list):  # Handle feature groups
                for f in feature:
                    if f in features.columns:
                        feature_idx = features.columns.get_loc(f)
                        monotonicity_constraints.append({'feature': feature_idx, 'monotonicity': -1})
            elif feature in features.columns:  # Handle single features
                feature_idx = features.columns.get_loc(feature)
                monotonicity_constraints.append({'feature': feature_idx, 'monotonicity': -1})
        
        return data_stream, classes, monotonicity_constraints
    
    def create_constraints_csv(self, output_path: str, custom_constraints: Optional[Dict] = None):
        """
        Create a constraints CSV file with default or custom constraints.
        Updated to handle nested constraints.
        
        Parameters:
        -----------
        output_path : str
            Path to save the constraints CSV file
        custom_constraints : Dict, optional
            Custom constraints to override defaults
        """
        constraints = custom_constraints if custom_constraints else self.default_constraints
        
        rows = []
        for dataset, constraint in constraints.items():
            rows.append({
                'Dataset': dataset,
                'Target': constraint.get('target', ''),
                'Positive Monotonic Features': str(constraint.get('positive_monotonic', [])),
                'Negative Monotonic Features': str(constraint.get('negative_monotonic', [])),
                'Notes': constraint.get('notes', '')
            })
        
        constraints_df = pd.DataFrame(rows)
        print(output_path)
        constraints_df.to_csv(output_path, index=False)
        
        print(f"Constraints CSV saved to: {output_path}")



def expand_constraints(constraints):
    """
    Expands constraint combinations from nested lists.
    
    Parameters:
    -----------
    constraints : dict
        Dictionary with 'target', 'positive_monotonic', and 'negative_monotonic' keys
        where the monotonic lists can contain both single features and lists of features
    
    Returns:
    --------
    list : List of constraint dictionaries, one for each combination
    """
    expanded = []
    target = constraints.get('target')
    pos_monotonic = constraints.get('positive_monotonic', [])
    neg_monotonic = constraints.get('negative_monotonic', [])
    
    pos_singles = [item for item in pos_monotonic if not isinstance(item, list)]
    pos_groups = [item for item in pos_monotonic if isinstance(item, list)]
    
    neg_singles = [item for item in neg_monotonic if not isinstance(item, list)]
    neg_groups = [item for item in neg_monotonic if isinstance(item, list)]
    
    for feature in pos_singles:
        expanded.append({
            'target': target,
            'positive_monotonic': [feature],
            'negative_monotonic': [],
            'experiment_name': f"{feature}(+)"
        })
    
    for feature in neg_singles:
        expanded.append({
            'target': target,
            'positive_monotonic': [],
            'negative_monotonic': [feature],
            'experiment_name': f"{feature}(-)"
        })
    
    for group in pos_groups:
        expanded.append({
            'target': target,
            'positive_monotonic': group,
            'negative_monotonic': [],
            'experiment_name': f"{'+'.join(group)}(+)"
        })
    
    for group in neg_groups:
        expanded.append({
            'target': target,
            'positive_monotonic': [],
            'negative_monotonic': group,
            'experiment_name': f"{'+'.join(group)}(-)"
        })
    
    expanded.append({
        'target': target,
        'positive_monotonic': [],
        'negative_monotonic': [],
        'experiment_name': "No Constraints"
    })
    
    return expanded

def run_single_constraint_experiment(dataset_name, df, constraints, experiment_name, **kwargs):
    """
    Run an experiment with a single constraint configuration.
    This is a modified version of run_experiment_on_dataset that focuses on a single constraint set.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    df : pd.DataFrame
        Prepared DataFrame
    constraints : dict
        Dictionary with 'target', 'positive_monotonic', and 'negative_monotonic' keys
    experiment_name : str
        Name for this experiment configuration
    **kwargs :
        Additional parameters for the experiment
    
    Returns:
    --------
    dict : Results from this experiment
    """
    output_dir = kwargs.get('output_dir', 'results')
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    loader = UCIDatasetLoader("") # Dummy loader for utility functions
    do_normalize = False if dataset_name in ["monk_s_problems"] else True
    prepared_df, formatted_constraints = loader.prepare_dataset_for_experiment(df, constraints, do_normalize)
    
    if prepared_df is None:
        print(f"Could not prepare dataset {dataset_name} for experiment")
        return None
    
    data_stream, classes, monotonicity_constraint_dicts = loader.convert_to_experiment_format(
        prepared_df, formatted_constraints)
    
    monotonicity_constraints = [
        MonotonicityConstraint(c['feature'], c['monotonicity']) 
        for c in monotonicity_constraint_dicts
    ]
    
    window_size = kwargs.get('window_size', 100)
    constraint_weight = kwargs.get('constraint_weight', 0.7)
    age_weight = kwargs.get('age_weight', 0.2)
    random_weight = kwargs.get('random_replacement_prob', 0.1)
    minimum_new_sample_quota = kwargs.get('minimum_new_sample_quota', 0.05)
    constraint_threshold = kwargs.get('constraint_threshold', 0.8)
    constraint_prediction_weight = kwargs.get('constraint_prediction_weight', 0.5)
    
    knnm_with_constraints = EnhancedKNNMClassifierV2(
        classes=classes,
        k=10,
        constraints=monotonicity_constraints,
        window_size=window_size,
        constraint_weight=constraint_weight,
        age_weight=age_weight,
        random_weight=random_weight,
        minimum_new_sample_quota=minimum_new_sample_quota,
        constraint_threshold=constraint_threshold,
        constraint_prediction_weight=constraint_prediction_weight
    )
    
    metrics_dict = {
        'accuracy': metrics.Accuracy(),
        'cohenkappa': metrics.CohenKappa(),
        'recall': metrics.Recall(),
        'f1': metrics.F1(),
        'precision': metrics.Precision(),
        'macrof1': metrics.MacroF1(),
        'microf1': metrics.MicroF1(),
        'macroprecision': metrics.MacroPrecision(),
        'macrorecall': metrics.MacroRecall(),
        'microprecision': metrics.MicroPrecision(),
        'microrecall': metrics.MicroRecall(),
        'weightedf1': metrics.WeightedF1(),
        'weightedprecision': metrics.WeightedPrecision(),
        'weightedrecall': metrics.WeightedRecall()
    }
    
    results = {
        'dataset': dataset_name,
        'experiment_name': experiment_name,
        'window_size': window_size,
        'constraint_weight': constraint_weight,
        'age_weight': age_weight,
        'random_weight': random_weight,
        'minimum_new_sample_quota': minimum_new_sample_quota,
        'constraint_threshold': constraint_threshold,
        'constraint_prediction_weight': constraint_prediction_weight,
        'iterations': [],
        'models': {
            'with_constraints': {
                'name': f'{experiment_name}',
                'online_predictions': [],
                'online_truths': []
            }
        }
    }
    
    for model_key in results['models']:
        for metric_name in metrics_dict:
            results['models'][model_key][f'online_{metric_name}'] = []
            if metric_name == 'accuracy':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.Accuracy()
            elif metric_name == 'cohenkappa':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.CohenKappa()
            elif metric_name == 'recall':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.Recall()
            elif metric_name == 'f1':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.F1()
            elif metric_name == 'precision':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.Precision()
            elif metric_name == 'macrof1':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.MacroF1()
            elif metric_name == 'microf1':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.MicroF1()
            elif metric_name == 'macroprecision':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.MacroPrecision()
            elif metric_name == 'macrorecall':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.MacroRecall()
            elif metric_name == 'microprecision':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.MicroPrecision()
            elif metric_name == 'microrecall':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.MicroRecall()
            elif metric_name == 'weightedf1':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.WeightedF1()
            elif metric_name == 'weightedprecision':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.WeightedPrecision()
            elif metric_name == 'weightedrecall':
                results['models'][model_key][f'online_{metric_name}_obj'] = metrics.WeightedRecall()
    
    for i, (X, y) in enumerate(tqdm(data_stream)):
        results['iterations'].append(i)
        
        pred_with = knnm_with_constraints.predict_one(X)
        results['models']['with_constraints']['online_predictions'].append(pred_with)
        results['models']['with_constraints']['online_truths'].append(y)
        
        for metric_name, metric in metrics_dict.items():
            metric_key = f'online_{metric_name}_obj'
            results['models']['with_constraints'][metric_key].update(y, pred_with)
            results['models']['with_constraints'][f'online_{metric_name}'].append(
                results['models']['with_constraints'][metric_key].get()
            )
        
        knnm_with_constraints.learn_one(X, y)
    
    if hasattr(knnm_with_constraints, 'get_stats'):
        stats = knnm_with_constraints.get_stats()
        results['models']['with_constraints']['constraint_violations_avoided'] = stats.get('constraint_violations_avoided', 0)
        results['models']['with_constraints']['constraint_violations_in_predictions'] = stats.get('constraint_violations_in_predictions', 0)
        results['models']['with_constraints']['avg_constraint_satisfaction'] = stats.get('avg_constraint_satisfaction', 0)
    
    for model_key in results['models']:
        for metric_name in metrics_dict:
            del results['models'][model_key][f'online_{metric_name}_obj']
    
    return results

def run_multiple_experiments(dataset_name, df, constraints, **kwargs):
    """
    Run multiple experiments with different constraint combinations.
    
    Parameters:
    -----------
    dataset_name : str
        Name of the dataset
    df : pd.DataFrame
        Prepared DataFrame
    constraints : dict
        Dictionary with 'target', 'positive_monotonic', and 'negative_monotonic' keys
        where the monotonic lists can contain both single features and lists of features
    **kwargs : 
        Additional parameters for run_experiment_on_dataset
    
    Returns:
    --------
    dict : Combined results from all experiments
    """
    expanded_constraints = expand_constraints(constraints)
    
    combined_results = {
        'dataset': dataset_name,
        'window_size': kwargs.get('window_size', 100),
        'constraint_weight': kwargs.get('constraint_weight', 0.7),
        'age_weight': kwargs.get('age_weight', 0.2),
        'random_weight': kwargs.get('random_replacement_prob', 0.1),
        'minimum_new_sample_quota': kwargs.get('minimum_new_sample_quota', 0.05),
        'constraint_threshold': kwargs.get('constraint_threshold', 0.8),
        'constraint_prediction_weight': kwargs.get('constraint_prediction_weight', 0.5),
        'iterations': [],
        'models': {}
    }
    
    output_dir = kwargs.get('output_dir', 'results')
    
    for exp_constraints in expanded_constraints:
        print(f"\nRunning experiment on dataset '{dataset_name}' with constraints: {exp_constraints['experiment_name']}")
        
        current_constraints = {
            'target': exp_constraints['target'],
            'positive_monotonic': exp_constraints['positive_monotonic'],
            'negative_monotonic': exp_constraints['negative_monotonic']
        }
        
        result = run_single_constraint_experiment(
            dataset_name=dataset_name,
            df=df,
            constraints=current_constraints,
            experiment_name=exp_constraints['experiment_name'],
            **kwargs
        )
        
        if result:
            if not combined_results['iterations'] and 'iterations' in result:
                combined_results['iterations'] = result['iterations']
            
            for model_key, model_data in result['models'].items():
                constraint_key = f"{model_key}_{exp_constraints['experiment_name'].replace(' ', '_')}"
                combined_results['models'][constraint_key] = model_data
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)
    
    with open(os.path.join(output_dir, f"{dataset_name}_combined_results.json"), 'w') as f:
        json.dump(combined_results, f, cls=NumpyEncoder)
    
    visualize_combined_results(combined_results, os.path.join(output_dir, "plots"))
    
    return combined_results

def visualize_combined_results(results, plots_dir):
    """
    Visualize combined results from multiple constraint experiments.
    
    Parameters:
    -----------
    results : dict
        Combined results dictionary from run_multiple_experiments
    plots_dir : str
        Directory to save plots
    """
    dataset_name = results['dataset']
    
    metrics_to_plot = ['accuracy', 'f1', 'macrof1', 'recall', 'precision']
    
    for metric_name in metrics_to_plot:
        plt.figure(figsize=(14, 8))
        
        colors = plt.cm.tab10.colors
        color_idx = 0
        
        for model_key, model_data in results['models'].items():
            metric_key = f'online_{metric_name}'
            
            if metric_key in model_data and len(model_data[metric_key]) > 0:
                subsample_rate = max(1, len(results['iterations']) // 100)
                x_values = results['iterations'][::subsample_rate]
                y_values = model_data[metric_key][::subsample_rate]
                
                label = model_data['name']
                
                linestyle = '-'
                if '(-)' in label:
                    linestyle = '--'
                
                plt.plot(
                    x_values, 
                    y_values, 
                    label=label, 
                    linestyle=linestyle,
                    color=colors[color_idx % len(colors)], 
                    linewidth=2, 
                    alpha=0.8
                )
                color_idx += 1
        
        plt.title(f'Online {metric_name.capitalize()} - Dataset: {dataset_name}')
        plt.xlabel('Iterations (samples seen)')
        plt.ylabel(f'{metric_name.capitalize()}')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"combined_{metric_name}_{dataset_name}.png"))
        plt.close()
    
    plt.figure(figsize=(16, 10))
    
    metrics_for_comparison = ['accuracy', 'macrof1']
    model_keys = list(results['models'].keys())
    model_names = [results['models'][k]['name'] for k in model_keys]
    
    accuracy_values = []
    for k in model_keys:
        acc_key = 'online_accuracy'
        if acc_key in results['models'][k] and len(results['models'][k][acc_key]) > 0:
            accuracy_values.append(results['models'][k][acc_key][-1])
        else:
            accuracy_values.append(0)
    
    sorted_indices = np.argsort(accuracy_values)[::-1]  # Descending order
    model_keys = [model_keys[i] for i in sorted_indices]
    model_names = [model_names[i] for i in sorted_indices]
    
    bar_width = 0.35
    x = np.arange(len(model_keys))
    
    for i, metric_name in enumerate(metrics_for_comparison):
        metric_key = f'online_{metric_name}'
        values = []
        
        for k in model_keys:
            if metric_key in results['models'][k] and len(results['models'][k][metric_key]) > 0:
                values.append(results['models'][k][metric_key][-1])
            else:
                values.append(0)
        
        offset = i * bar_width - (len(metrics_for_comparison) * bar_width / 2) + bar_width/2
        bars = plt.bar(x + offset, values, bar_width, label=metric_name.capitalize())
        
        for j, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f'{height:.3f}',
                ha='center', 
                va='bottom',
                fontsize=8
            )
    
    plt.xticks(x, model_names, rotation=45, ha='right')
    plt.title(f'Final Metrics Comparison - {dataset_name}')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"combined_final_metrics_{dataset_name}.png"))
    plt.close()
    
    violation_metrics = ['constraint_violations_avoided', 'constraint_violations_in_predictions']
    if any(metric in results['models'][model_keys[0]] for metric in violation_metrics):
        plt.figure(figsize=(16, 10))
        
        violation_data = []
        for k in model_keys:
            if 'No_Constraints' in k:
                continue  # Skip no constraints model for this chart
                
            model_violations = []
            for metric in violation_metrics:
                if metric in results['models'][k]:
                    model_violations.append(results['models'][k][metric])
                else:
                    model_violations.append(0)
            
            if 'avg_constraint_satisfaction' in results['models'][k]:
                model_violations.append(results['models'][k]['avg_constraint_satisfaction'] * 100)
            else:
                model_violations.append(0)
                
            violation_data.append(model_violations)
        
        valid_model_keys = [k for k in model_keys if 'No_Constraints' not in k]
        valid_model_names = [results['models'][k]['name'] for k in valid_model_keys]
        
        bar_width = 0.25
        x = np.arange(len(valid_model_keys))
        
        metrics_labels = ['Violations Avoided', 'Violations in Predictions', 'Avg. Constraint Satisfaction (%)']
        
        for i, (metric_name, metric_label) in enumerate(zip(violation_metrics + ['avg_constraint_satisfaction'], metrics_labels)):
            values = [data[i] for data in violation_data]
            
            plt.bar(
                x + i * bar_width - bar_width,
                values,
                bar_width,
                label=metric_label
            )
        
        plt.xticks(x, valid_model_names, rotation=45, ha='right')
        plt.title(f'Constraint Statistics - {dataset_name}')
        plt.ylabel('Count / Percentage')
        plt.legend()
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"combined_constraint_stats_{dataset_name}.png"))
        plt.close()
    
def create_combined_summary(all_results, output_dir):
    """
    Create a summary of all experiments with multiple constraint combinations.
    
    Parameters:
    -----------
    all_results : Dict[str, Dict]
        Dictionary mapping dataset names to combined results
    output_dir : str
        Directory to save summary
    """
    if not all_results:
        print("No results to summarize")
        return
    
    rows = []
    metrics_of_interest = ['accuracy', 'macrof1']
    
    for dataset_name, results in all_results.items():
        for model_key, model_data in results['models'].items():
            row = {'dataset': dataset_name, 'constraint_config': model_data['name']}
            
            for metric_name in metrics_of_interest:
                metric_key = f'online_{metric_name}'
                if metric_key in model_data and len(model_data[metric_key]) > 0:
                    row[metric_name] = model_data[metric_key][-1]
            
            for metric_name in ['constraint_violations_avoided', 'avg_constraint_satisfaction']:
                if metric_name in model_data:
                    row[metric_name] = model_data[metric_name]
            
            rows.append(row)
    
    summary_df = pd.DataFrame(rows)
    
    summary_path = os.path.join(output_dir, "combined_experiment_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Combined summary saved to {summary_path}")
    
    plt.figure(figsize=(18, 12))
    
    datasets = summary_df['dataset'].unique()
    constraint_configs = summary_df['constraint_config'].unique()
    
    x = np.arange(len(datasets))
    width = 0.8 / len(constraint_configs)
    
    for i, config in enumerate(constraint_configs):
        config_data = summary_df[summary_df['constraint_config'] == config]
        accuracies = []
        
        for dataset in datasets:
            dataset_row = config_data[config_data['dataset'] == dataset]
            if not dataset_row.empty and 'accuracy' in dataset_row.columns:
                accuracies.append(dataset_row['accuracy'].iloc[0])
            else:
                accuracies.append(0)
        
        plt.bar(
            x + i * width - (len(constraint_configs) * width / 2) + width/2, 
            accuracies, 
            width, 
            label=config
        )
    
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across Datasets and Constraint Configurations')
    plt.xticks(x, datasets, rotation=45, ha='right')
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_accuracy_comparison.png"), bbox_inches='tight')
    plt.close()

def run_all_experiments(data_dir, 
                       constraints_file=None,
                       max_datasets=None,
                       dataset_filter=None,
                       window_size=100,
                       min_constraint_size=10,
                       random_replacement_prob=0.05,
                       constraint_weight=0.7,
                       age_weight=0.2,
                       constraint_threshold=0.8,
                       constraint_prediction_weight=0.5,
                       output_dir="results"):
    """
    Run experiments on multiple UCI datasets using EnhancedKNNMClassifier with expanded constraints.
    
    Parameters:
    -----------
    data_dir : str
        Path to the directory containing UCI datasets
    constraints_file : str, optional
        Path to a CSV file containing constraint definitions
    max_datasets : int, optional
        Maximum number of datasets to load
    dataset_filter : List[str], optional
        List of dataset names to load (if None, load all)
    window_size : int
        Size of the sliding window for KNNMClassifier
    min_constraint_size : int
        Used to calculate minimum_new_sample_quota
    random_replacement_prob : float
        Maps to random_weight in EnhancedKNNMClassifier
    constraint_weight : float
        Weight for constraint satisfaction in replacement score (0-1)
    age_weight : float
        Weight for age in replacement score (0-1)
    constraint_threshold : float
        Minimum constraint satisfaction threshold for preferential treatment
    constraint_prediction_weight : float
        Weight given to constraint-based prediction (0-1) vs standard KNN prediction
    output_dir : str
        Directory to save results
    """
    loader = UCIDatasetLoader(data_dir, constraints_file)
    
    if constraints_file is None:
        constraints_file = os.path.join(data_dir, "constraints.csv")
        if not os.path.exists(constraints_file):
            loader.create_constraints_csv(constraints_file)
    
    datasets = loader.load_datasets(max_datasets=max_datasets, dataset_filter=dataset_filter)
    
    print(f"Loaded {len(datasets)} datasets")
    
    all_results = {}
    for name, (df, constraints) in datasets.items():
        print(f"\nProcessing dataset: {name}")
        
        minimum_new_sample_quota = min(0.5, max(0.05, min_constraint_size / window_size))
        
        results = run_multiple_experiments(
            dataset_name=name,
            df=df,
            constraints=constraints,
            window_size=window_size,
            minimum_new_sample_quota=minimum_new_sample_quota,
            random_replacement_prob=random_replacement_prob,
            constraint_weight=constraint_weight,
            age_weight=age_weight,
            constraint_threshold=constraint_threshold,
            constraint_prediction_weight=constraint_prediction_weight,
            output_dir=output_dir
        )
        
        if results:
            all_results[name] = results
    
    create_combined_summary(all_results, output_dir)
    
    return all_results


def example_usage_with_nested_constraints():
    """
    Example usage of the UCI dataset loader with nested monotonicity constraints.
    """
    data_dir = "./data"
    
    print("=== UCI Dataset Loader with Nested Monotonicity Constraints ===")
    print(f"Looking for datasets in: {data_dir}")
    
    loader = UCIDatasetLoader(data_dir)
    
    constraints_file = os.path.join(data_dir, "custom_constraints.csv")
    
    custom_constraints = {
        "iris": {
            "target": "Species",
            "positive_monotonic": ["petal length", "petal width", ["petal length", "petal width"]],
            "negative_monotonic": ["sepal width"],
            "notes": "Testing multiple constraint combinations"
        }
    }
    
    loader.create_constraints_csv(constraints_file, custom_constraints)
    print(f"Custom constraints file created at: {constraints_file}")
    
    example_dataset = "iris"
    print(f"\nLoading example dataset: {example_dataset}")
    
    iris_datasets = loader.load_datasets(dataset_filter=[example_dataset])
    if iris_datasets:
        iris_df, iris_constraints = next(iter(iris_datasets.values()))
        print(f"  Shape: {iris_df.shape}")
        print(f"  Columns: {', '.join(iris_df.columns)}")
        print(f"  Constraints: {iris_constraints}")
        
        print("\nRunning multiple experiments with different constraint combinations...")
        
        results = run_multiple_experiments(
            dataset_name=example_dataset,
            df=iris_df,
            constraints=iris_constraints,
            window_size=20,
            min_constraint_size=5,
            random_replacement_prob=0.05,
            constraint_weight=0.7,
            age_weight=0.2,
            constraint_threshold=0.8,
            constraint_prediction_weight=0.5,
            output_dir="example_results"
        )
        
        print("\nExperiments completed!")
        
        print("\nFinal metrics for each constraint configuration:")
        
        for model_key, model_data in results['models'].items():
            print(f"\n  Configuration: {model_data['name']}")
            
            metrics = {
                k.replace('online_', ''): v[-1] 
                for k, v in model_data.items()
                if k.startswith('online_') and isinstance(v, list) 
            }
            
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
            
            if 'constraint_violations_avoided' in model_data:
                print(f"    Constraint violations avoided: {model_data['constraint_violations_avoided']}")
            
            if 'avg_constraint_satisfaction' in model_data:
                print(f"    Avg constraint satisfaction: {model_data['avg_constraint_satisfaction']:.4f}")
        
        print("\nVisualization plots saved to: example_results/plots/")
    else:
        print(f"Dataset {example_dataset} not found")


def example_usage_with_nested_constraints():
    """
    Example usage of the UCI dataset loader with nested monotonicity constraints.
    """
    data_dir = "./data"
    
    print("=== UCI Dataset Loader with Nested Monotonicity Constraints ===")
    print(f"Looking for datasets in: {data_dir}")
    
    loader = UCIDatasetLoader(data_dir)
    
    constraints_file = os.path.join(data_dir, "custom_constraints.csv")
    
    custom_constraints = {
        "iris": {
            "target": "Species",
            "positive_monotonic": ["petal length", "petal width", ["petal length", "petal width"]],
            "negative_monotonic": ["sepal width"],
            "notes": "Testing multiple constraint combinations"
        }
    }
    
    loader.create_constraints_csv(constraints_file, custom_constraints)
    print(f"Custom constraints file created at: {constraints_file}")
    
    example_dataset = "iris"
    print(f"\nLoading example dataset: {example_dataset}")
    
    iris_datasets = loader.load_datasets(dataset_filter=[example_dataset])
    if iris_datasets:
        iris_df, iris_constraints = next(iter(iris_datasets.values()))
        print(f"  Shape: {iris_df.shape}")
        print(f"  Columns: {', '.join(iris_df.columns)}")
        print(f"  Constraints: {iris_constraints}")
        
        print("\nRunning multiple experiments with different constraint combinations...")
        
        results = run_multiple_experiments(
            dataset_name=example_dataset,
            df=iris_df,
            constraints=iris_constraints,
            window_size=20,
            min_constraint_size=5,
            random_replacement_prob=0.05,
            constraint_weight=0.7,
            age_weight=0.2,
            constraint_threshold=0.8,
            constraint_prediction_weight=0.5,
            output_dir="example_results"
        )
        
        print("\nExperiments completed!")
        
        print("\nFinal metrics for each constraint configuration:")
        
        for model_key, model_data in results['models'].items():
            print(f"\n  Configuration: {model_data['name']}")
            
            metrics = {
                k.replace('online_', ''): v[-1] 
                for k, v in model_data.items()
                if k.startswith('online_') and isinstance(v, list) 
            }
            
            for metric, value in metrics.items():
                print(f"    {metric}: {value:.4f}")
            
            if 'constraint_violations_avoided' in model_data:
                print(f"    Constraint violations avoided: {model_data['constraint_violations_avoided']}")
            
            if 'avg_constraint_satisfaction' in model_data:
                print(f"    Avg constraint satisfaction: {model_data['avg_constraint_satisfaction']:.4f}")
        
        print("\nVisualization plots saved to: example_results/plots/")
    else:
        print(f"Dataset {example_dataset} not found")


def main():
    """
    Main function with updated argument parsing for nested constraints.
    """
    parser = argparse.ArgumentParser(description='Run monotonicity constraint experiments on UCI datasets')
    parser.add_argument('--data_dir', type=str, default='D:/mgr/mgrdeg/project/masters/data_dump',
                      help='Directory containing UCI datasets')
    parser.add_argument('--constraints_file', type=str, default=None,
                      help='Path to constraints CSV file')
    parser.add_argument('--max_datasets', type=int, default=None,
                      help='Maximum number of datasets to process')
    parser.add_argument('--dataset_filter', type=str, nargs='+', default=None,
                      help='Only process datasets matching these names')
    parser.add_argument('--window_size', type=int, default=100,
                      help='Window size for KNN')
    parser.add_argument('--min_constraint_size', type=int, default=10,
                      help='Minimum size before enforcing constraints')
    parser.add_argument('--random_replacement_prob', type=float, default=0.05,
                      help='Probability of random replacement (maps to random_weight)')
    parser.add_argument('--constraint_weight', type=float, default=0.7,
                      help='Weight for constraint satisfaction in replacement score (0-1)')
    parser.add_argument('--age_weight', type=float, default=0.2,
                      help='Weight for age in replacement score (0-1)')
    parser.add_argument('--constraint_threshold', type=float, default=0.8,
                      help='Minimum constraint satisfaction threshold for preferential treatment')
    parser.add_argument('--constraint_prediction_weight', type=float, default=0.5,
                      help='Weight given to constraint-based prediction vs standard KNN')
    parser.add_argument('--output_dir', type=str, default='balanced_results_multi_v2',
                      help='Directory to save results')
    parser.add_argument('--example', action='store_true',
                      help='Run example usage instead of full experiments')
    
    args = parser.parse_args()
    
    if args.example:
        example_usage_with_nested_constraints()
    else:
        run_all_experiments(
            data_dir=args.data_dir,
            constraints_file=args.constraints_file,
            max_datasets=args.max_datasets,
            dataset_filter=args.dataset_filter,
            window_size=args.window_size,
            min_constraint_size=args.min_constraint_size,
            random_replacement_prob=args.random_replacement_prob,
            constraint_weight=args.constraint_weight,
            age_weight=args.age_weight,
            constraint_threshold=args.constraint_threshold,
            constraint_prediction_weight=args.constraint_prediction_weight,
            output_dir=args.output_dir
        )


if __name__ == "__main__":
    pd.set_option('display.max_columns', None)
    main()