import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict, Counter, deque
import abc
import numbers
import typing
import math
from river import metrics
from river.tree.base import Leaf
from river.tree.utils import BranchFactory
from river.tree.hoeffding_tree import HoeffdingTree
from river.tree.nodes.branch import DTBranch
from river.tree.nodes.htc_nodes import LeafMajorityClass, LeafNaiveBayes, LeafNaiveBayesAdaptive
from river.tree.nodes.leaf import HTLeaf
from river.tree.split_criterion import GiniSplitCriterion, HellingerDistanceCriterion, InfoGainSplitCriterion
from river.tree.splitter import GaussianSplitter, Splitter
from river.tree.splitter.nominal_splitter_classif import NominalSplitterClassif
from river.utils.norm import normalize_values_in_dict
from river.tree.utils import do_naive_bayes_prediction, round_sig_fig
from river.base.typing import ClfTarget
from river import base
from tqdm import tqdm
import os
import random
import heapq
from sklearn.model_selection import train_test_split
from river import neighbors


from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.metrics import accuracy_score



output_dir = "mass_stm_monotonicity_experiments_sktree"
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
            if sorted_primary[i] < midpoint - transition_width/2:
                sorted_y[i] = 0
            elif sorted_primary[i] > midpoint + transition_width/2:
                sorted_y[i] = 1
            else:
                position = (sorted_primary[i] - (midpoint - transition_width/2)) / transition_width
                sorted_y[i] = 1 if np.random.random() < position else 0
        
        for i in range(1, n_samples):
            if sorted_y[i] < sorted_y[i-1]:
                sorted_y[i] = sorted_y[i-1]
        
        bump_probability = 0.01
        for i in range(n_samples):
            if sorted_y[i] == 0 and np.random.random() < bump_probability:
                sorted_y[i] = 1
                for j in range(i+1, n_samples):
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
            if sorted_primary[i] < midpoint - transition_width/2:
                sorted_y[i] = 1
            elif sorted_primary[i] > midpoint + transition_width/2:
                sorted_y[i] = 0
            else:
                position = (sorted_primary[i] - (midpoint - transition_width/2)) / transition_width
                sorted_y[i] = 0 if np.random.random() < position else 1
        
        for i in range(1, n_samples):
            if sorted_y[i] > sorted_y[i-1]:
                sorted_y[i] = sorted_y[i-1]
        
        bump_probability = 0.01
        for i in range(n_samples):
            if sorted_y[i] == 1 and np.random.random() < bump_probability:
                sorted_y[i] = 0
                for j in range(i+1, n_samples):
                    sorted_y[j] = 0
                break
        
        y = np.zeros_like(sorted_y)
        y[sort_idx] = sorted_y
    
    elif case_type == "positive_noisy":
        bins = 10
        bin_edges = np.linspace(0, 10, bins+1)
        bin_indices = np.digitize(X_primary, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins-1)
        
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
    
    df = pd.DataFrame({
        0: X_primary,
        1: X_random1,
        2: X_random2, 
        3: X_random3,
        'target': y
    })
    
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



class TreeBaggingSamplingClassifier:
    def __init__(self, max_sample_bank_size, nr_clf_to_probe, classes, feature_cnt, rho = 1e-1,  max_sample_proportion=0.9, minibatch_size=1, monotonicity_constraints = [], save_prop_mono_Trees=False):
        self.sample_bank_size = max_sample_bank_size
        self.nr_clf_to_probe = nr_clf_to_probe
        self.max_sample_proportion = max_sample_proportion
        self.minibatch_size = minibatch_size
        self.monotonicity_constraints = monotonicity_constraints
        self.classes = classes
        self.model: DecisionTreeClassifier = None
        self.sample_bank = []
        self.save_prop_mono_Trees = save_prop_mono_Trees
        self.feature_cnt = feature_cnt
        self_mono_proportion = []
        self.rho = rho
        self.minibatch_ctr=0

    def learn_one(self, X, y):
        if len(self.sample_bank) < self.sample_bank_size:
            self.sample_bank.append((X, y))
        if len(self.sample_bank) == self.sample_bank_size:
            self.sample_bank.pop(0)
            self.sample_bank.append((X, y))

        self.minibatch_ctr = (self.minibatch_ctr + 1) % self.minibatch_size
        if self.minibatch_ctr == 0 and len(self.sample_bank) >= self.sample_bank_size:
            self.sample_trees()
        
    def sample_trees(self):
        best_model = None
        best_score = -1

        for _ in range(self.nr_clf_to_probe):
            sample_size = int(len(self.sample_bank) * self.max_sample_proportion)
            sample_indices = np.random.choice(len(self.sample_bank), sample_size, replace=False)
            train_samples = [self.sample_bank[i] for i in sample_indices]
            test_samples = [self.sample_bank[i] for i in range(len(self.sample_bank)) if i not in sample_indices]


            X_train, y_train = zip(*train_samples)
            X_test, y_test = zip(*test_samples)

            X_train_np = np.array([[d[i] for i in range(self.feature_cnt)] for d in X_train])
            y_train_np = np.array(y_train)

            if X_test:
                X_test_np = np.array([[d[i] for i in range(self.feature_cnt)] for d in X_test])
                y_test_np = np.array(y_test)
            else:
                X_test_np = np.array([]).reshape(0, self.feature_cnt)
                y_test_np = np.array([])


            model = DecisionTreeClassifier()
            model.fit(X_train_np, y_train_np)

            if X_test:
                y_pred = model.predict(X_test_np)
                score = accuracy_score(y_test_np, y_pred)
            else:
                if X_train_np.size > 0:
                    y_train_pred = model.predict(X_train_np)
                    score = accuracy_score(y_train_np, y_train_pred)
                else:
                    score = 0

            if self.monotonicity_constraints:
                monotonicity = -1 * self.evaluate_tree_monotonicity(model)
                score += monotonicity

            if score > best_score:
                best_score = score
                best_model = model

        self.model = best_model
    
    def evaluate_tree_monotonicity(self, tree: DecisionTreeClassifier):
        np.set_printoptions(threshold=np.inf)
        n_nodes = tree.tree_.node_count
        children_left = tree.tree_.children_left
        children_right = tree.tree_.children_right
        feature = tree.tree_.feature
        threshold = tree.tree_.threshold

        def get_mx_class(node_idx):
            mxv = -1
            mxc = -1
            n_classes_in_tree = tree.tree_.value.shape[2]
            for k in range(min(len(self.classes), n_classes_in_tree)):
                if tree.tree_.value[node_idx, :, k]*tree.tree_.n_node_samples[node_idx] > mxv:
                    mxv = tree.tree_.value[node_idx, :, k]*tree.tree_.n_node_samples[node_idx]
                    mxc = k
            return mxc
        
        def A(I):
            if I == 0:
                return 0
            else:
                return -1 / np.log2(I)
        
            
        minmax_values = np.full((n_nodes, self.feature_cnt, 2), np.inf)
        minmax_values[:, :, 0] *= -1
        stack = [(0, None)]
        is_leaves = np.zeros(shape=tree.tree_.node_count, dtype=bool)
        while len(stack) > 0:
            node_id, parent_mtx = stack.pop()
            is_leaves[node_id] = False

            if parent_mtx is not None:
                minmax_values[node_id] = np.copy(parent_mtx)


            is_split_node = children_left[node_id] != children_right[node_id]
            if is_split_node:
                feature_id = feature[node_id]
                threshold_value = threshold[node_id]

                lpmx = np.copy(minmax_values[node_id])
                rpmx = np.copy(minmax_values[node_id])
                lpmx[feature_id, 1] = threshold_value
                rpmx[feature_id, 0] = threshold_value

                stack.append((children_left[node_id], lpmx))
                stack.append((children_right[node_id], rpmx))
            else:
                is_leaves[node_id] = True
    
        max_classes = []

        leaf_cnt = 0
        for index, is_leaf in enumerate(is_leaves):
            leaf_cnt += 1
            if is_leaf:
                max_class = get_mx_class(index)
                max_classes.append(max_class)
            else:
                max_classes.append(None)
        
        arr = np.array(max_classes)

        matrix = np.full((self.feature_cnt, n_nodes, n_nodes), None)

        
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i==j and is_leaves[i]:
                    matrix[:, i, j] = 0
                    continue
                for constraint in self.monotonicity_constraints:#foreach constraint check monotonicity
                    feature = constraint.feature
                    monotonicity = constraint.monotonicity
                    if is_leaves[i] and is_leaves[j]:
                        if monotonicity == 1:
                            if arr[i] > arr[j] and minmax_values[i,feature_id,0] < minmax_values[j,feature_id,1]:
                                matrix[feature_id, i, j] = 1
                            else:
                                matrix[feature_id, i, j] = 0
                        else:
                            if arr[i] < arr[j] and minmax_values[i,feature_id,0] > minmax_values[j,feature_id,1]:
                                matrix[feature_id, i, j] = 1
                            else:
                                matrix[feature_id, i, j] = 0
        

        denom = leaf_cnt * (leaf_cnt - 1) if leaf_cnt > 1 else 1

        cache_mtx = np.full((self.feature_cnt, n_nodes, n_nodes), 1)
        for constraint in self.monotonicity_constraints:
            feature_id = constraint.feature
            feature_mtx = matrix[feature_id]
            feature_mtx = np.where(feature_mtx == None, 0, feature_mtx)
            cache_mtx = np.logical_and(cache_mtx, feature_mtx)

        
        
        W = np.sum(cache_mtx)
        I = W/denom
        Av = A(I)
        return Av * self.rho

    def predict_proba_one(self, X):
        X_np = np.array([[X[i] for i in range(self.feature_cnt)]])

        model_probas_for_known_classes = self.model.predict_proba(X_np)[0]

        current_model_class_probas = dict(zip(self.model.classes_, model_probas_for_known_classes))

        final_probas_dict = {cls_label: 0.0 for cls_label in self.classes}

        final_probas_dict.update(current_model_class_probas)

        ordered_probabilities = [final_probas_dict[cls_label] for cls_label in self.classes]

        return ordered_probabilities


    def predict_one(self, X):
        if self.model is None:
            if self.classes:
                return self.classes[0]
            return None

        predictions = self.predict_proba_one(X)
        predictions_np = np.array(predictions)
        if predictions_np.size == 0:
            return None

        max_idx = predictions_np.argmax()
        max_class = self.classes[max_idx]
        return max_class


def run_experiment(case_name, constraint_feature=None, constraint_direction=None, 
                  use_majority=False, buffer_size=10, rho=1e-5, stdevmul=2.0,
                  grace_period=500, seed=42):
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
    main_df = generate_synthetic_data(case_name, n_samples=1000, seed=seed)
    
    if constraint_feature is not None:
        constraints = [MonotonicityConstraint(constraint_feature, constraint_direction)]
    else:
        constraints = []
    
    if use_majority:
        model = BufferedMajorityClassClassifier(
            classes=[0, 1],
            buffer_size=buffer_size
        )
    else:
        model = TreeBaggingSamplingClassifier(
            monotonicity_constraints=constraints,
            classes=[0, 1],
            max_sample_bank_size=100,
            nr_clf_to_probe=10,
            feature_cnt=4,
        )
    
    online_metrics = {
        'accuracy': metrics.Accuracy(),
        'cohenkappa': metrics.CohenKappa(),
        'macrof1': metrics.MacroF1(),
        'macroprecision': metrics.MacroPrecision(),
        'macrorecall': metrics.MacroRecall(),
        'microf1': metrics.MicroF1(),
        'microprecision': metrics.MicroPrecision(),
        'microrecall': metrics.MicroRecall(),
        'weightedf1': metrics.WeightedF1(),
        'weightedprecision': metrics.WeightedPrecision(),
        'weightedrecall': metrics.WeightedRecall()
    }
    
    main_stream = []
    for _, row in main_df.iterrows():
        X = {i: row[i] for i in range(4)}
        y = row['target']
        main_stream.append((X, y))
    
    results = {
        'case': case_name,
        'constraint_feature': constraint_feature,
        'constraint_direction': constraint_direction,
        'use_majority': use_majority,
        'buffer_size': buffer_size,
        'rho': rho,
        'stdevmul': stdevmul,
        'grace_period': grace_period,
        'seed': seed,
        'iterations': [],
        'online_predictions': [],
        'online_truths': [],
    }
    
    for name in online_metrics.keys():
        results[f'online_{name}'] = []
    
    results['max_list_size'] = []
    results['avg_list_size'] = []
    results['num_lists'] = []
    
    print(f"Running experiment on case '{case_name}', "
          f"constraint: {'None' if not constraints else constraints[0]}, "
          f"seed: {seed}")
    
    for i, (X, y) in enumerate(main_stream):
        pred = model.predict_one(X)
        
        results['online_predictions'].append(pred)
        results['online_truths'].append(y)
        
        for metric_name, metric in online_metrics.items():
            metric.update(y, pred)
        
        results['iterations'].append(i)
        
        for name, metric in online_metrics.items():
            results[f'online_{name}'].append(metric.get())
        
        
        model.learn_one(X, y)
    
    for name in online_metrics.keys():
        results[f'final_online_{name}'] = results[f'online_{name}'][-1]
    
    print(f"Seed {seed} complete: final online accuracy = {results['final_online_accuracy']:.4f}")
    
    return results

def run_experiment_with_multiple_seeds(config, seeds=range(100, 200)):
    """Run the same experiment with multiple seeds and return aggregated results."""
    print(f"Running experiment with {len(seeds)} seeds: {config}")
    
    all_results = []
    
    for seed in seeds:
        experiment_config = config.copy()
        experiment_config['seed'] = seed
        result = run_experiment(**experiment_config)
        all_results.append(result)
    
    final_metrics = ['final_online_accuracy', 'final_online_macrof1', 
                     'final_online_weightedf1', 'final_online_cohenkappa']
    
    aggregated = {
        'case': config['case_name'],
        'constraint_feature': config['constraint_feature'],
        'constraint_direction': config['constraint_direction'],
        'use_majority': config.get('use_majority', False),
        'buffer_size': config.get('buffer_size', None),
        'rho': config.get('rho', None),
        'stdevmul': config.get('stdevmul', None),
        'grace_period': config.get('grace_period', None),
        'seeds': seeds,
    }
    
    for metric in final_metrics:
        values = [result[metric] for result in all_results]
        aggregated[metric] = np.mean(values)
        aggregated[f"{metric}_std"] = np.std(values)
        aggregated[f"{metric}_min"] = np.min(values)
        aggregated[f"{metric}_max"] = np.max(values)
        aggregated[f"{metric}_values"] = values
    
    return aggregated


def create_boxplot_summary(aggregated_results, metric='final_online_accuracy'):
    """Create boxplots comparing performance across multiple seeds."""
    plots_dir = os.path.join(output_dir, "plots")
    
    plt.figure(figsize=(18, 10))
    
    cases = sorted(list({exp["case"] for exp in aggregated_results}))
    
    experiment_categories = [
        ('No Constraint', lambda e: not e.get('use_majority', False) and e['constraint_feature'] is None),
        ('Pos Constraint', lambda e: not e.get('use_majority', False) and e.get('constraint_feature') == 0 
                                       and e.get('constraint_direction') == 1),
        ('Neg Constraint', lambda e: not e.get('use_majority', False) and e.get('constraint_feature') == 0 
                                      and e.get('constraint_direction') == -1),
        ('Majority (5)', lambda e: e.get('use_majority', True) and e.get('buffer_size') == 5),
        ('Majority (20)', lambda e: e.get('use_majority', True) and e.get('buffer_size') == 20),
        ('Majority (100)', lambda e: e.get('use_majority', True) and e.get('buffer_size') == 100)
    ]
    
    boxplot_data = []
    labels = []
    positions = []
    colors = []
    
    color_map = {
        'No Constraint': '#1f77b4',
        'Pos Constraint': '#2ca02c',
        'Neg Constraint': '#d62728',
        'Majority (5)': '#9467bd',
        'Majority (20)': '#8c564b',
        'Majority (100)': '#e377c2'
    }
    
    pos = 1
    for i, case in enumerate(cases):
        for j, (category_name, condition) in enumerate(experiment_categories):
            exps = [exp for exp in aggregated_results 
                   if exp['case'] == case and condition(exp)]
            
            if exps:
                values = [exp[f"{metric}_values"] for exp in exps]
                boxplot_data.append(values[0])
                labels.append(f"{category_name}\n{case}")
                positions.append(pos)
                colors.append(color_map[category_name])
                pos += 1
        
        pos += 1.5
    
    box = plt.boxplot(boxplot_data, positions=positions, patch_artist=True, 
                      widths=0.6, showfliers=True)
    
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    
    metric_name = metric.replace('final_online_', '').capitalize()
    plt.ylabel(metric_name, fontsize=14)
    
    plt.xticks(positions, labels, rotation=45, ha='right')
    
    legend_elements = [plt.Line2D([0], [0], color=color, lw=10, alpha=0.6, label=name) 
                       for name, color in color_map.items()]
    plt.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"boxplot_{metric}.png"), bbox_inches='tight', dpi=300)
    plt.close()

def create_violin_plot(aggregated_results, metric='final_online_accuracy'):
    """Create violin plots for more detailed distribution visualization."""
    plots_dir = os.path.join(output_dir, "plots")
    
    plt.figure(figsize=(20, 10))
    
    cases = sorted(list({exp["case"] for exp in aggregated_results}))
    
    experiment_categories = [
        ('No Constraint', lambda e: not e.get('use_majority', False) and e['constraint_feature'] is None),
        ('Pos Constraint', lambda e: not e.get('use_majority', False) and e.get('constraint_feature') == 0 
                                       and e.get('constraint_direction') == 1),
        ('Neg Constraint', lambda e: not e.get('use_majority', False) and e.get('constraint_feature') == 0 
                                      and e.get('constraint_direction') == -1),
        ('Majority (5)', lambda e: e.get('use_majority', True) and e.get('buffer_size') == 5),
        ('Majority (20)', lambda e: e.get('use_majority', True) and e.get('buffer_size') == 20),
        ('Majority (100)', lambda e: e.get('use_majority', True) and e.get('buffer_size') == 100)
    ]
    
    all_data = []
    
    for case in cases:
        for category_name, condition in experiment_categories:
            exps = [exp for exp in aggregated_results 
                   if exp['case'] == case and condition(exp)]
            
            if exps:
                exp = exps[0]
                values = exp[f"{metric}_values"]
                
                for value in values:
                    all_data.append({
                        'Case': case,
                        'Model': category_name,
                        metric: value
                    })
    
    df = pd.DataFrame(all_data)
    
    ax = sns.violinplot(x='Model', y=metric, hue='Case', data=df, 
                       split=True, inner="quart", linewidth=1)
    
    metric_name = metric.replace('final_online_', '').capitalize()
    plt.title(f'{metric_name} Distribution by Model and Case', fontsize=16)
    plt.xlabel('Model', fontsize=14)
    plt.ylabel(metric_name, fontsize=14)
    plt.xticks(rotation=30, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, f"violin_{metric}.png"), bbox_inches='tight', dpi=300)
    plt.close()

def create_summary_plot(results_by_case):
    """Create a summary plot comparing different models across all cases with improved visibility."""
    plots_dir = os.path.join(output_dir, "plots")
    
    cases = ["strict_positive", "positive_noisy", "strict_negative", "opposing_trend"]
    case_positions = np.arange(len(cases))
    
    ht_experiment_types = [
        ('No Constraint', lambda r: not r.get('use_majority', False) and r['constraint_feature'] is None),
        ('Pos Constraint (ρ=1e-5)', lambda r: not r.get('use_majority', False) and r.get('constraint_feature') == 0
         and r.get('constraint_direction') == 1 and r.get('rho') == 1e-5),
        ('Pos Constraint (ρ=1e-6)', lambda r: not r.get('use_majority', False) and r.get('constraint_feature') == 0
         and r.get('constraint_direction') == 1 and r.get('rho') == 1e-6),
        ('Pos Constraint (ρ=1e-4)', lambda r: not r.get('use_majority', False) and r.get('constraint_feature') == 0
         and r.get('constraint_direction') == 1 and r.get('rho') == 1e-4),
        ('Neg Constraint', lambda r: not r.get('use_majority', False) and r.get('constraint_feature') == 0
         and r.get('constraint_direction') == -1)
    ]
    
    majority_experiment_types = [
        ('Majority (5)', lambda r: r.get('use_majority', False) and r.get('buffer_size') == 5),
        ('Majority (20)', lambda r: r.get('use_majority', False) and r.get('buffer_size') == 20),
        ('Majority (100)', lambda r: r.get('use_majority', False) and r.get('buffer_size') == 100)
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
                accuracies.append(matching_results[0]['final_online_accuracy'])
            else:
                accuracies.append(0)
        
        offset = i * ht_barWidth - (len(ht_experiment_types) * ht_barWidth / 2) + ht_barWidth/2
        ax1.bar(case_positions + offset, accuracies, width=ht_barWidth, 
                label=f"{label}", alpha=0.3, color=bar_colors[i],
                edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Online Accuracy', fontsize=12)
    ax1.set_title('Hoeffding Tree Models - Final Online Accuracy', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    bar_colors = plt.cm.plasma(np.linspace(0, 0.8, len(majority_experiment_types)))
    
    for i, (label, condition) in enumerate(majority_experiment_types):
        accuracies = []
        
        for case in cases:
            case_results = results_by_case[case]
            matching_results = [r for r in case_results if condition(r)]
            
            if matching_results:
                accuracies.append(matching_results[0]['final_online_accuracy'])
            else:
                accuracies.append(0)
        
        offset = i * majority_barWidth - (len(majority_experiment_types) * majority_barWidth / 2) + majority_barWidth/2
        ax2.bar(case_positions + offset, accuracies, width=majority_barWidth, 
                label=f"{label}", alpha=0.8, color=bar_colors[i],
                edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Case Type', fontsize=12)
    ax2.set_ylabel('Online Accuracy', fontsize=12)
    ax2.set_title('Majority Classifiers - Final Online Accuracy', fontsize=14)
    ax2.set_xticks(case_positions)
    ax2.set_xticklabels(cases, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "final_online_accuracy_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 15), sharex=True)
    
    bar_colors = plt.cm.viridis(np.linspace(0, 0.8, len(ht_experiment_types)))
    
    for i, (label, condition) in enumerate(ht_experiment_types):
        macrof1s = []
        
        for case in cases:
            case_results = results_by_case[case]
            matching_results = [r for r in case_results if condition(r)]
            
            if matching_results:
                macrof1s.append(matching_results[0]['final_online_macrof1'])
            else:
                macrof1s.append(0)
        
        offset = i * ht_barWidth - (len(ht_experiment_types) * ht_barWidth / 2) + ht_barWidth/2
        ax1.bar(case_positions + offset, macrof1s, width=ht_barWidth, 
                label=f"{label}", alpha=0.8, color=bar_colors[i],
                edgecolor='black', linewidth=1)
    
    ax1.set_ylabel('Online MacroF1', fontsize=12)
    ax1.set_title('Hoeffding Tree Models - Final Online MacroF1', fontsize=14)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
    
    bar_colors = plt.cm.plasma(np.linspace(0, 0.8, len(majority_experiment_types)))
    
    for i, (label, condition) in enumerate(majority_experiment_types):
        macrof1s = []
        
        for case in cases:
            case_results = results_by_case[case]
            matching_results = [r for r in case_results if condition(r)]
            
            if matching_results:
                macrof1s.append(matching_results[0]['final_online_macrof1'])
            else:
                macrof1s.append(0)
        
        offset = i * majority_barWidth - (len(majority_experiment_types) * majority_barWidth / 2) + majority_barWidth/2
        ax2.bar(case_positions + offset, macrof1s, width=majority_barWidth, 
                label=f"{label}", alpha=0.8, color=bar_colors[i],
                edgecolor='black', linewidth=1)
    
    ax2.set_xlabel('Case Type', fontsize=12)
    ax2.set_ylabel('Online MacroF1', fontsize=12)
    ax2.set_title('Majority Classifiers - Final Online MacroF1', fontsize=14)
    ax2.set_xticks(case_positions)
    ax2.set_xticklabels(cases, fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "final_online_macrof1_comparison.png"), bbox_inches='tight', dpi=300)
    plt.close()


def create_experiment_configs_multi_seed():
    """Create a comprehensive set of experiment configurations."""
    experiments_config = []
    
    cases = ["strict_positive", "positive_noisy", "strict_negative", "opposing_trend"]
    
    for case in cases:
        experiments_config.append({
            'case_name': case,
            'constraint_feature': None,
            'constraint_direction': None,
            'use_majority': False,
            'rho': 1e-5,
            'stdevmul': 2.0,
            'grace_period': 50
        })
    
    for case in cases:
        experiments_config.append({
            'case_name': case,
            'constraint_feature': 0,
            'constraint_direction': 1,
            'use_majority': False,
            'rho': 1e-5,
            'stdevmul': 2.0,
            'grace_period': 50
        })
    
    for case in cases:
        experiments_config.append({
            'case_name': case,
            'constraint_feature': 0,
            'constraint_direction': -1,
            'use_majority': False,
            'rho': 1e-5,
            'stdevmul': 2.0,
            'grace_period': 50
        })
    
    for case in cases:
        for buffer_size in [5, 20, 100]:
            experiments_config.append({
                'case_name': case,
                'constraint_feature': None,
                'constraint_direction': None,
                'use_majority': True,
                'buffer_size': buffer_size
            })
    
    return experiments_config


def run_multi_seed_experiments(num_seeds=100, start_seed=100):
    """Run all experiments with multiple seeds and visualize results."""
    results_dir = os.path.join(output_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    seeds = range(start_seed, start_seed + num_seeds)
    print(f"Running experiments with seeds {start_seed} to {start_seed + num_seeds - 1}")
    
    experiments_config = create_experiment_configs_multi_seed()
    
    aggregated_results = []
    for config in experiments_config:
        agg_result = run_experiment_with_multiple_seeds(config, seeds)
        aggregated_results.append(agg_result)
        
        config_name = f"{config['case_name']}_"
        if config.get('use_majority', False):
            config_name += f"majority_{config['buffer_size']}"
        else:
            constraint = "none" if config['constraint_feature'] is None else (
                "pos" if config['constraint_direction'] == 1 else "neg")
            config_name += f"ht_{constraint}_rho{config['rho']}"
        
        result_file = os.path.join(results_dir, f"{config_name}.pkl")
    
    
    for metric in ['final_online_accuracy', 'final_online_macrof1', 
                  'final_online_weightedf1', 'final_online_cohenkappa']:
        create_boxplot_summary(aggregated_results, metric)
        create_violin_plot(aggregated_results, metric)
    
    summary_data = []
    for result in aggregated_results:
        row = {
            'Case': result['case'],
            'Model': 'Majority' if result['use_majority'] else 'HoeffdingTree',
            'Constraint': 'None' if result['constraint_feature'] is None else (
                'Positive' if result['constraint_direction'] == 1 else 'Negative'
            ),
        }
        
        if result['use_majority']:
            row['Parameter'] = f"Buffer={result['buffer_size']}"
        else:
            row['Parameter'] = f"rho={result['rho']}, stdevmul={result['stdevmul']}"
        
        for metric in ['final_online_accuracy', 'final_online_macrof1', 
                      'final_online_weightedf1', 'final_online_cohenkappa']:
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