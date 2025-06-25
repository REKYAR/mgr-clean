import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, deque
from river import metrics
import os
import random
from river import neighbors


output_dir = "mass_stm_monotonicity_experiments_knn"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, "plots"), exist_ok=True)


def generate_synthetic_data(case_type, seed, n_samples=5000):
    """
    Generate synthetic data for testing monotonicity constraints.

    Parameters:
    -----------
    case_type : str
        Type of data to generate:
        - "strict_positive": Feature has strictly nondecreasing relationship with target
        - "strict_negative": Feature has strictly nonincreasing relationship with target
        - "positive_noisy": Feature has positive relationship with noise
        - "opposing_trend": Feature has negative relationship (opposite of positive constraint)
    n_samples : int
        Number of samples to generate

    Returns:
    --------
    main_df: DataFrame with integer-indexed columns
    """
    np.random.seed(seed)
    random.seed(seed)
    X_primary = np.random.uniform(0, 10, n_samples)

    X_random1 = np.random.normal(5, 2, n_samples)
    X_random2 = np.random.exponential(2, n_samples)
    X_random3 = np.random.uniform(-3, 3, n_samples)

    if case_type == "strict_positive":
        sort_idx = np.argsort(X_primary)
        sorted_primary = X_primary[sort_idx]

        sorted_y = np.zeros(n_samples, dtype=int)

        midpoint = 5.0
        transition_width = 4.0

        for i in range(n_samples):
            if sorted_primary[i] < midpoint - transition_width / 2:
                sorted_y[i] = 0
            elif sorted_primary[i] > midpoint + transition_width / 2:
                sorted_y[i] = 1
            else:
                position = (
                    sorted_primary[i] - (midpoint - transition_width / 2)
                ) / transition_width
                sorted_y[i] = 1 if np.random.random() < position else 0

        for i in range(1, n_samples):
            if sorted_y[i] < sorted_y[i - 1]:
                sorted_y[i] = sorted_y[i - 1]

        bump_probability = 0.01
        for i in range(n_samples):
            if sorted_y[i] == 0 and np.random.random() < bump_probability:
                sorted_y[i] = 1
                for j in range(i + 1, n_samples):
                    sorted_y[j] = 1
                break

        y = np.zeros_like(sorted_y)
        y[sort_idx] = sorted_y

    elif case_type == "strict_negative":
        sort_idx = np.argsort(X_primary)
        sorted_primary = X_primary[sort_idx]

        sorted_y = np.ones(n_samples, dtype=int)

        midpoint = 5.0
        transition_width = 4.0

        for i in range(n_samples):
            if sorted_primary[i] < midpoint - transition_width / 2:
                sorted_y[i] = 1
            elif sorted_primary[i] > midpoint + transition_width / 2:
                sorted_y[i] = 0
            else:
                position = (
                    sorted_primary[i] - (midpoint - transition_width / 2)
                ) / transition_width
                sorted_y[i] = 0 if np.random.random() < position else 1

        for i in range(1, n_samples):
            if sorted_y[i] > sorted_y[i - 1]:
                sorted_y[i] = sorted_y[i - 1]

        bump_probability = 0.01
        for i in range(n_samples):
            if sorted_y[i] == 1 and np.random.random() < bump_probability:
                sorted_y[i] = 0
                for j in range(i + 1, n_samples):
                    sorted_y[j] = 0
                break

        y = np.zeros_like(sorted_y)
        y[sort_idx] = sorted_y

    elif case_type == "positive_noisy":
        bins = 10
        bin_edges = np.linspace(0, 10, bins + 1)
        bin_indices = np.digitize(X_primary, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)

        bin_probabilities = np.linspace(0.1, 0.9, bins)
        base_prob = bin_probabilities[bin_indices]

        noise = np.random.normal(0, 0.2, n_samples)
        prob = np.clip(base_prob + noise, 0.05, 0.95)

        y = np.random.binomial(1, prob)

    elif case_type == "opposing_trend":
        prob = 0.9 - 0.08 * X_primary
        prob = np.clip(prob, 0.05, 0.95)

        y = np.random.binomial(1, prob)

    else:
        raise ValueError(f"Unknown case type: {case_type}")

    df = pd.DataFrame(
        {0: X_primary, 1: X_random1, 2: X_random2, 3: X_random3, "target": y}
    )

    main_df = df.reset_index(drop=True)

    return main_df


class MonotonicityConstraint:
    def __init__(self, feature, monotonicity):
        """Initialize a monotonicity constraint."""
        self.feature = feature
        self.monotonicity = monotonicity

    def __str__(self):
        direction = "positive" if self.monotonicity == 1 else "negative"
        return f"Feature {self.feature}, {direction} monotonicity"

    def __repr__(self):
        return self.__str__()


class BufferedMajorityClassClassifier:
    def __init__(self, classes, buffer_size=10):
        """Majority Class Classifier with a fixed-size buffer.

        Parameters:
        -----------
        classes : list
            List of possible classes
        buffer_size : int
            Maximum number of recent examples to store in the buffer
        """
        self.classes = classes
        self.buffer_size = buffer_size
        self.counter = Counter()
        self.majority_class = None
        self.buffer = deque(maxlen=buffer_size)

        self.list_sizes_history = []
        self.max_list_size_history = []
        self.avg_list_size_history = []
        self.num_lists_history = []

    def learn_one(self, X, y):
        if len(self.buffer) == self.buffer_size:
            old_y = self.buffer.popleft()
            self.counter[old_y] -= 1
            if self.counter[old_y] == 0:
                del self.counter[old_y]

        self.buffer.append(y)
        self.counter[y] += 1

        if self.counter:
            self.majority_class = self.counter.most_common(1)[0][0]

        self.list_sizes_history.append([len(self.buffer)])
        self.max_list_size_history.append(len(self.buffer))
        self.avg_list_size_history.append(len(self.buffer))
        self.num_lists_history.append(1)

        return self

    def predict_one(self, X):
        if not self.counter:
            return self.classes[0]
        return self.majority_class

    def predict_proba_one(self, X):
        """Predict probabilities for one sample."""
        if not self.counter:
            return {cls: 1 / len(self.classes) for cls in self.classes}

        total = sum(self.counter.values())
        return {cls: count / total for cls, count in self.counter.items()}


class EnhancedKNNMClassifierV2:
    def __init__(
        self,
        classes,
        k=10,
        constraints=[],
        window_size=100,
        constraint_weight=0.7,
        age_weight=0.2,
        random_weight=0.1,
        minimum_new_sample_quota=0.05,
        constraint_threshold=0.8,
        constraint_prediction_weight=0.5,
    ):
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

        self.window = []

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
            return 1.0

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
                    if (
                        (index_y > index_sample_class and x[feature] >= sample[feature])
                        or (
                            index_y < index_sample_class
                            and x[feature] <= sample[feature]
                        )
                        or (index_y == index_sample_class)
                    ):
                        satisfied_checks += 1

                elif monotonicity == -1:
                    if (
                        (index_y > index_sample_class and x[feature] <= sample[feature])
                        or (
                            index_y < index_sample_class
                            and x[feature] >= sample[feature]
                        )
                        or (index_y == index_sample_class)
                    ):
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

            constraint_score = 1 - constraint_satisfaction
            age_score = normalized_age

            new_sample_quota = int(self.window_size * self.minimum_new_sample_quota)
            is_oldest = sorted(
                range(len(self.window)), key=lambda j: self.window[j][2], reverse=True
            )[:new_sample_quota]

            if i in is_oldest:
                final_score = 0.9 + (0.1 * random_component)
            else:
                final_score = (
                    self.constraint_weight * constraint_score
                    + self.age_weight * age_score
                    + self.random_weight * random_component
                )

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

                if monotonicity == 1:
                    if (index_pred_class > index_y and X[feature] < x[feature]) or (
                        index_pred_class < index_y and X[feature] > x[feature]
                    ):
                        return True

                elif monotonicity == -1:
                    if (index_pred_class > index_y and X[feature] > x[feature]) or (
                        index_pred_class < index_y and X[feature] < x[feature]
                    ):
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
                "constraint_violations_in_predictions": self.constraint_violations_in_predictions,
            }

        avg_age = np.mean([age for _, _, age, _ in self.window])
        avg_cs = np.mean([cs for _, _, _, cs in self.window])

        return {
            "window_size": len(self.window),
            "avg_age": avg_age,
            "avg_constraint_satisfaction": avg_cs,
            "total_samples_seen": self.total_samples_seen,
            "constraint_violations_avoided": self.constraint_violations_avoided,
            "constraint_violations_in_predictions": self.constraint_violations_in_predictions,
        }


def run_experiment(
    case_name,
    constraint_feature=None,
    constraint_direction=None,
    use_majority=False,
    buffer_size=10,
    rho=1e-5,
    stdevmul=2.0,
    grace_period=500,
    seed=42,
):
    """
    Run experiment with sequential test-then-train approach.

    Parameters:
    -----------
    case_name : str
        Name of the case to run
    constraint_feature : int or None
        Feature index to apply monotonicity constraint on, or None for no constraint
    constraint_direction : int or None
        1 for positive, -1 for negative monotonicity, None for no constraint
    use_majority : bool
        Whether to use majority classifier instead of Hoeffding Tree
    buffer_size : int
        Buffer size for majority classifier
    rho : float
        Monotonicity penalty parameter for Hoeffding Tree
    stdevmul : float
        Standard deviation multiplier for Hoeffding Tree
    grace_period : int
        Grace period for the Hoeffding Tree
    seed : int
        Random seed for data generation
    """
    main_df = generate_synthetic_data(case_name, n_samples=5000, seed=seed)

    if constraint_feature is not None:
        constraints = [MonotonicityConstraint(constraint_feature, constraint_direction)]
    else:
        constraints = []

    if use_majority:
        model = BufferedMajorityClassClassifier(classes=[0, 1], buffer_size=buffer_size)
    else:
        model = EnhancedKNNMClassifierV2(
            constraints=constraints,
            classes=[0, 1],
        )

    online_metrics = {
        "accuracy": metrics.Accuracy(),
        "cohenkappa": metrics.CohenKappa(),
        "macrof1": metrics.MacroF1(),
        "macroprecision": metrics.MacroPrecision(),
        "macrorecall": metrics.MacroRecall(),
        "microf1": metrics.MicroF1(),
        "microprecision": metrics.MicroPrecision(),
        "microrecall": metrics.MicroRecall(),
        "weightedf1": metrics.WeightedF1(),
        "weightedprecision": metrics.WeightedPrecision(),
        "weightedrecall": metrics.WeightedRecall(),
    }

    main_stream = []
    for _, row in main_df.iterrows():
        X = {i: row[i] for i in range(4)}
        y = row["target"]
        main_stream.append((X, y))

    results = {
        "case": case_name,
        "constraint_feature": constraint_feature,
        "constraint_direction": constraint_direction,
        "use_majority": use_majority,
        "buffer_size": buffer_size,
        "rho": rho,
        "stdevmul": stdevmul,
        "grace_period": grace_period,
        "seed": seed,
        "iterations": [],
        "online_predictions": [],
        "online_truths": [],
    }

    for name in online_metrics.keys():
        results[f"online_{name}"] = []

    results["max_list_size"] = []
    results["avg_list_size"] = []
    results["num_lists"] = []

    print(
        f"Running experiment on case '{case_name}', "
        f"constraint: {'None' if not constraints else constraints[0]}, "
        f"seed: {seed}"
    )

    for i, (X, y) in enumerate(main_stream):
        pred = model.predict_one(X)

        results["online_predictions"].append(pred)
        results["online_truths"].append(y)

        for metric_name, metric in online_metrics.items():
            metric.update(y, pred)

        results["iterations"].append(i)

        for name, metric in online_metrics.items():
            results[f"online_{name}"].append(metric.get())

        model.learn_one(X, y)

    for name in online_metrics.keys():
        results[f"final_online_{name}"] = results[f"online_{name}"][-1]

    print(
        f"Seed {seed} complete: final online accuracy = {results['final_online_accuracy']:.4f}"
    )

    return results


def run_experiment_with_multiple_seeds(config, seeds=range(100, 200)):
    """Run the same experiment with multiple seeds and return aggregated results."""
    print(f"Running experiment with {len(seeds)} seeds: {config}")

    all_results = []

    for seed in seeds:
        experiment_config = config.copy()
        experiment_config["seed"] = seed
        result = run_experiment(**experiment_config)
        all_results.append(result)

    final_metrics = [
        "final_online_accuracy",
        "final_online_macrof1",
        "final_online_weightedf1",
        "final_online_cohenkappa",
    ]

    aggregated = {
        "case": config["case_name"],
        "constraint_feature": config["constraint_feature"],
        "constraint_direction": config["constraint_direction"],
        "use_majority": config.get("use_majority", False),
        "buffer_size": config.get("buffer_size", None),
        "rho": config.get("rho", None),
        "stdevmul": config.get("stdevmul", None),
        "grace_period": config.get("grace_period", None),
        "seeds": seeds,
    }

    for metric in final_metrics:
        values = [result[metric] for result in all_results]
        aggregated[metric] = np.mean(values)
        aggregated[f"{metric}_std"] = np.std(values)
        aggregated[f"{metric}_min"] = np.min(values)
        aggregated[f"{metric}_max"] = np.max(values)
        aggregated[f"{metric}_values"] = values

    return aggregated


def create_boxplot_summary(aggregated_results, metric="final_online_accuracy"):
    """Create boxplots comparing performance across multiple seeds."""
    plots_dir = os.path.join(output_dir, "plots")

    plt.figure(figsize=(18, 10))

    cases = sorted(list({exp["case"] for exp in aggregated_results}))

    experiment_categories = [
        (
            "No Constraint",
            lambda e: not e.get("use_majority", False)
            and e["constraint_feature"] is None,
        ),
        (
            "Pos Constraint",
            lambda e: not e.get("use_majority", False)
            and e.get("constraint_feature") == 0
            and e.get("constraint_direction") == 1,
        ),
        (
            "Neg Constraint",
            lambda e: not e.get("use_majority", False)
            and e.get("constraint_feature") == 0
            and e.get("constraint_direction") == -1,
        ),
        (
            "Majority (5)",
            lambda e: e.get("use_majority", True) and e.get("buffer_size") == 5,
        ),
        (
            "Majority (20)",
            lambda e: e.get("use_majority", True) and e.get("buffer_size") == 20,
        ),
        (
            "Majority (100)",
            lambda e: e.get("use_majority", True) and e.get("buffer_size") == 100,
        ),
    ]

    boxplot_data = []
    labels = []
    positions = []
    colors = []

    color_map = {
        "No Constraint": "#1f77b4",
        "Pos Constraint": "#2ca02c",
        "Neg Constraint": "#d62728",
        "Majority (5)": "#9467bd",
        "Majority (20)": "#8c564b",
        "Majority (100)": "#e377c2",
    }

    pos = 1
    for i, case in enumerate(cases):
        for j, (category_name, condition) in enumerate(experiment_categories):
            exps = [
                exp
                for exp in aggregated_results
                if exp["case"] == case and condition(exp)
            ]

            if exps:
                values = [exp[f"{metric}_values"] for exp in exps]
                boxplot_data.append(values[0])
                labels.append(f"{category_name}\n{case}")
                positions.append(pos)
                colors.append(color_map[category_name])
                pos += 1

        pos += 1.5

    box = plt.boxplot(
        boxplot_data,
        positions=positions,
        patch_artist=True,
        widths=0.6,
        showfliers=True,
    )

    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.grid(True, axis="y", linestyle="--", alpha=0.7)

    metric_name = metric.replace("final_online_", "").capitalize()
    plt.ylabel(metric_name, fontsize=14)

    plt.xticks(positions, labels, rotation=45, ha="right")

    legend_elements = [
        plt.Line2D([0], [0], color=color, lw=10, alpha=0.6, label=name)
        for name, color in color_map.items()
    ]
    plt.legend(
        handles=legend_elements, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=3
    )

    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, f"boxplot_{metric}.png"), bbox_inches="tight", dpi=300
    )
    plt.close()


def create_violin_plot(aggregated_results, metric="final_online_accuracy"):
    """Create violin plots for more detailed distribution visualization."""
    plots_dir = os.path.join(output_dir, "plots")

    plt.figure(figsize=(20, 10))

    cases = sorted(list({exp["case"] for exp in aggregated_results}))

    experiment_categories = [
        (
            "No Constraint",
            lambda e: not e.get("use_majority", False)
            and e["constraint_feature"] is None,
        ),
        (
            "Pos Constraint",
            lambda e: not e.get("use_majority", False)
            and e.get("constraint_feature") == 0
            and e.get("constraint_direction") == 1,
        ),
        (
            "Neg Constraint",
            lambda e: not e.get("use_majority", False)
            and e.get("constraint_feature") == 0
            and e.get("constraint_direction") == -1,
        ),
        (
            "Majority (5)",
            lambda e: e.get("use_majority", True) and e.get("buffer_size") == 5,
        ),
        (
            "Majority (20)",
            lambda e: e.get("use_majority", True) and e.get("buffer_size") == 20,
        ),
        (
            "Majority (100)",
            lambda e: e.get("use_majority", True) and e.get("buffer_size") == 100,
        ),
    ]

    all_data = []

    for case in cases:
        for category_name, condition in experiment_categories:
            exps = [
                exp
                for exp in aggregated_results
                if exp["case"] == case and condition(exp)
            ]

            if exps:
                exp = exps[0]
                values = exp[f"{metric}_values"]

                for value in values:
                    all_data.append(
                        {"Case": case, "Model": category_name, metric: value}
                    )

    df = pd.DataFrame(all_data)

    ax = sns.violinplot(
        x="Model", y=metric, hue="Case", data=df, split=True, inner="quart", linewidth=1
    )

    metric_name = metric.replace("final_online_", "").capitalize()
    plt.title(f"{metric_name} Distribution by Model and Case", fontsize=16)
    plt.xlabel("Model", fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, f"violin_{metric}.png"), bbox_inches="tight", dpi=300
    )
    plt.close()


def create_summary_plot(results_by_case):
    """Create a summary plot comparing different models across all cases with improved visibility."""
    plots_dir = os.path.join(output_dir, "plots")

    cases = ["strict_positive", "positive_noisy", "strict_negative", "opposing_trend"]
    case_positions = np.arange(len(cases))

    ht_experiment_types = [
        (
            "No Constraint",
            lambda r: not r.get("use_majority", False)
            and r["constraint_feature"] is None,
        ),
        (
            "Pos Constraint (ρ=1e-5)",
            lambda r: not r.get("use_majority", False)
            and r.get("constraint_feature") == 0
            and r.get("constraint_direction") == 1
            and r.get("rho") == 1e-5,
        ),
        (
            "Pos Constraint (ρ=1e-6)",
            lambda r: not r.get("use_majority", False)
            and r.get("constraint_feature") == 0
            and r.get("constraint_direction") == 1
            and r.get("rho") == 1e-6,
        ),
        (
            "Pos Constraint (ρ=1e-4)",
            lambda r: not r.get("use_majority", False)
            and r.get("constraint_feature") == 0
            and r.get("constraint_direction") == 1
            and r.get("rho") == 1e-4,
        ),
        (
            "Neg Constraint",
            lambda r: not r.get("use_majority", False)
            and r.get("constraint_feature") == 0
            and r.get("constraint_direction") == -1,
        ),
    ]

    majority_experiment_types = [
        (
            "Majority (5)",
            lambda r: r.get("use_majority", False) and r.get("buffer_size") == 5,
        ),
        (
            "Majority (20)",
            lambda r: r.get("use_majority", False) and r.get("buffer_size") == 20,
        ),
        (
            "Majority (100)",
            lambda r: r.get("use_majority", False) and r.get("buffer_size") == 100,
        ),
    ]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), sharex=True)

    ht_barWidth = 0.75 / len(ht_experiment_types)
    majority_barWidth = 0.75 / len(majority_experiment_types)

    bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(ht_experiment_types)))

    for i, (label, condition) in enumerate(ht_experiment_types):
        accuracies = []

        for case in cases:
            case_results = results_by_case[case]
            matching_results = [r for r in case_results if condition(r)]

            if matching_results:
                accuracies.append(matching_results[0]["final_online_accuracy"])
            else:
                accuracies.append(0)

        offset = (
            i * ht_barWidth
            - (len(ht_experiment_types) * ht_barWidth / 2)
            + ht_barWidth / 2
        )
        ax1.bar(
            case_positions + offset,
            accuracies,
            width=ht_barWidth,
            label=f"{label}",
            alpha=0.3,
            color=bar_colors[i],
            edgecolor="black",
            linewidth=1,
        )

    ax1.set_ylabel("Online Accuracy", fontsize=12)
    ax1.set_title("Hoeffding Tree Models - Final Online Accuracy", fontsize=14)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)

    bar_colors = plt.cm.plasma(np.linspace(0, 0.8, len(majority_experiment_types)))

    for i, (label, condition) in enumerate(majority_experiment_types):
        accuracies = []

        for case in cases:
            case_results = results_by_case[case]
            matching_results = [r for r in case_results if condition(r)]

            if matching_results:
                accuracies.append(matching_results[0]["final_online_accuracy"])
            else:
                accuracies.append(0)

        offset = (
            i * majority_barWidth
            - (len(majority_experiment_types) * majority_barWidth / 2)
            + majority_barWidth / 2
        )
        ax2.bar(
            case_positions + offset,
            accuracies,
            width=majority_barWidth,
            label=f"{label}",
            alpha=0.8,
            color=bar_colors[i],
            edgecolor="black",
            linewidth=1,
        )

    ax2.set_xlabel("Case Type", fontsize=12)
    ax2.set_ylabel("Online Accuracy", fontsize=12)
    ax2.set_title("Majority Classifiers - Final Online Accuracy", fontsize=14)
    ax2.set_xticks(case_positions)
    ax2.set_xticklabels(cases, fontsize=11)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, "final_online_accuracy_comparison.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), sharex=True)

    bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(ht_experiment_types)))

    for i, (label, condition) in enumerate(ht_experiment_types):
        macrof1s = []

        for case in cases:
            case_results = results_by_case[case]
            matching_results = [r for r in case_results if condition(r)]

            if matching_results:
                macrof1s.append(matching_results[0]["final_online_macrof1"])
            else:
                macrof1s.append(0)

        offset = (
            i * ht_barWidth
            - (len(ht_experiment_types) * ht_barWidth / 2)
            + ht_barWidth / 2
        )
        ax1.bar(
            case_positions + offset,
            macrof1s,
            width=ht_barWidth,
            label=f"{label}",
            alpha=0.8,
            color=bar_colors[i],
            edgecolor="black",
            linewidth=1,
        )

    ax1.set_ylabel("Online MacroF1", fontsize=12)
    ax1.set_title("Hoeffding Tree Models - Final Online MacroF1", fontsize=14)
    ax1.grid(axis="y", alpha=0.3, linestyle="--")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2)

    bar_colors = plt.cm.plasma(np.linspace(0, 0.8, len(majority_experiment_types)))

    for i, (label, condition) in enumerate(majority_experiment_types):
        macrof1s = []

        for case in cases:
            case_results = results_by_case[case]
            matching_results = [r for r in case_results if condition(r)]

            if matching_results:
                macrof1s.append(matching_results[0]["final_online_macrof1"])
            else:
                macrof1s.append(0)

        offset = (
            i * majority_barWidth
            - (len(majority_experiment_types) * majority_barWidth / 2)
            + majority_barWidth / 2
        )
        ax2.bar(
            case_positions + offset,
            macrof1s,
            width=majority_barWidth,
            label=f"{label}",
            alpha=0.8,
            color=bar_colors[i],
            edgecolor="black",
            linewidth=1,
        )

    ax2.set_xlabel("Case Type", fontsize=12)
    ax2.set_ylabel("Online MacroF1", fontsize=12)
    ax2.set_title("Majority Classifiers - Final Online MacroF1", fontsize=14)
    ax2.set_xticks(case_positions)
    ax2.set_xticklabels(cases, fontsize=11)
    ax2.grid(axis="y", alpha=0.3, linestyle="--")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(plots_dir, "final_online_macrof1_comparison.png"),
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()


def create_experiment_configs_multi_seed():
    """Create a comprehensive set of experiment configurations."""
    experiments_config = []

    cases = ["strict_positive", "positive_noisy", "strict_negative", "opposing_trend"]

    for case in cases:
        experiments_config.append(
            {
                "case_name": case,
                "constraint_feature": None,
                "constraint_direction": None,
                "use_majority": False,
                "rho": 1e-5,
                "stdevmul": 2.0,
                "grace_period": 50,
            }
        )

    for case in cases:
        experiments_config.append(
            {
                "case_name": case,
                "constraint_feature": 0,
                "constraint_direction": 1,
                "use_majority": False,
                "rho": 1e-5,
                "stdevmul": 2.0,
                "grace_period": 50,
            }
        )

    for case in cases:
        experiments_config.append(
            {
                "case_name": case,
                "constraint_feature": 0,
                "constraint_direction": -1,
                "use_majority": False,
                "rho": 1e-5,
                "stdevmul": 2.0,
                "grace_period": 50,
            }
        )

    for case in cases:
        for buffer_size in [5, 20, 100]:
            experiments_config.append(
                {
                    "case_name": case,
                    "constraint_feature": None,
                    "constraint_direction": None,
                    "use_majority": True,
                    "buffer_size": buffer_size,
                }
            )

    return experiments_config


def run_multi_seed_experiments(num_seeds=100, start_seed=100):
    """Run all experiments with multiple seeds and visualize results."""
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    seeds = range(start_seed, start_seed + num_seeds)
    print(
        f"Running experiments with seeds {start_seed} to {start_seed + num_seeds - 1}"
    )

    experiments_config = create_experiment_configs_multi_seed()

    aggregated_results = []
    for config in experiments_config:
        agg_result = run_experiment_with_multiple_seeds(config, seeds)
        aggregated_results.append(agg_result)

        config_name = f"{config['case_name']}_"
        if config.get("use_majority", False):
            config_name += f"majority_{config['buffer_size']}"
        else:
            constraint = (
                "none"
                if config["constraint_feature"] is None
                else ("pos" if config["constraint_direction"] == 1 else "neg")
            )
            config_name += f"ht_{constraint}_rho{config['rho']}"

        result_file = os.path.join(results_dir, f"{config_name}.pkl")

    for metric in [
        "final_online_accuracy",
        "final_online_macrof1",
        "final_online_weightedf1",
        "final_online_cohenkappa",
    ]:
        create_boxplot_summary(aggregated_results, metric)
        create_violin_plot(aggregated_results, metric)

    summary_data = []
    for result in aggregated_results:
        row = {
            "Case": result["case"],
            "Model": "Majority" if result["use_majority"] else "HoeffdingTree",
            "Constraint": "None"
            if result["constraint_feature"] is None
            else ("Positive" if result["constraint_direction"] == 1 else "Negative"),
        }

        if result["use_majority"]:
            row["Parameter"] = f"Buffer={result['buffer_size']}"
        else:
            row["Parameter"] = f"rho={result['rho']}, stdevmul={result['stdevmul']}"

        for metric in [
            "final_online_accuracy",
            "final_online_macrof1",
            "final_online_weightedf1",
            "final_online_cohenkappa",
        ]:
            row[f"{metric}_mean"] = result[metric]
            row[f"{metric}_std"] = result[f"{metric}_std"]
            row[f"{metric}_min"] = result[f"{metric}_min"]
            row[f"{metric}_max"] = result[f"{metric}_max"]

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(os.path.join(results_dir, "summary_statistics.csv"), index=False)

    print("All multi-seed experiments completed successfully!")
    return aggregated_results


if __name__ == "__main__":
    run_multi_seed_experiments()
