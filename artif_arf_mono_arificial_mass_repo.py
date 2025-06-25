import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, deque
import abc
import numbers
import typing
import math
from river import metrics
from river.tree.base import Leaf
from river.tree.utils import BranchFactory
from river.tree.hoeffding_tree import HoeffdingTree
from river.tree.nodes.branch import DTBranch
from river.tree.nodes.htc_nodes import LeafNaiveBayes, LeafNaiveBayesAdaptive
from river.tree.nodes.leaf import HTLeaf
from river.tree.split_criterion import (
    GiniSplitCriterion,
    HellingerDistanceCriterion,
    InfoGainSplitCriterion,
)
from river.tree.splitter import GaussianSplitter, Splitter
from river.tree.splitter.nominal_splitter_classif import NominalSplitterClassif
from river.utils.norm import normalize_values_in_dict
from river.tree.utils import round_sig_fig
from river import base
import os
import random


output_dir = "mass_stm_monotonicity_experiments_arf_exp"
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


class HTLeafMono(Leaf, abc.ABC):
    """Base leaf class to be used in Hoeffding Trees.

    Parameters
    ----------
    stats
        Target statistics (they differ in classification and regression tasks).
    depth
        The depth of the node
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    attributes
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(**kwargs)
        self.stats = stats
        self.depth = depth

        self.splitter = splitter

        self.splitters = {}
        self._disabled_attrs = set()
        self._last_split_attempt_at = self.total_weight

    @property
    @abc.abstractmethod
    def total_weight(self) -> float:
        pass

    def is_active(self):
        return self.splitters is not None

    def activate(self):
        if not self.is_active():
            self.splitters = {}

    def deactivate(self):
        self.splitters = None

    @property
    def last_split_attempt_at(self) -> float:
        """The weight seen at last split evaluation.

        Returns
        -------
        Weight seen at last split evaluation.
        """
        return self._last_split_attempt_at

    @last_split_attempt_at.setter
    def last_split_attempt_at(self, weight):
        """Set the weight seen at last split evaluation.

        Parameters
        ----------
        weight
            Weight seen at last split evaluation.
        """
        self._last_split_attempt_at = weight

    @staticmethod
    @abc.abstractmethod
    def new_nominal_splitter():
        pass

    @abc.abstractmethod
    def update_stats(self, y, w):
        pass

    def _iter_features(self, x) -> typing.Iterable:
        """Determine how the input instance is looped through when updating the splitters.

        Parameters
        ----------
        x
            The input instance.
        """
        yield from x.items()

    def update_splitters(self, x, y, w, nominal_attributes):
        for att_id, att_val in self._iter_features(x):
            if att_id in self._disabled_attrs:
                continue

            try:
                splitter = self.splitters[att_id]
            except KeyError:
                if (
                    nominal_attributes is not None and att_id in nominal_attributes
                ) or not isinstance(att_val, numbers.Number):
                    splitter = self.new_nominal_splitter()
                else:
                    splitter = self.splitter.clone()

                self.splitters[att_id] = splitter
            splitter.update(att_val, y, w)

    def best_split_suggestions(self, criterion, tree) -> list[BranchFactory]:
        """Find possible split candidates.

        Parameters
        ----------
        criterion
            The splitting criterion to be used.
        tree
            Decision tree.

        Returns
        -------
        Split candidates.
        """
        leafs = tree._find_leaves()
        clean_leafs = [leaf for leaf in leafs if leaf is not self]

        best_suggestions = []
        pre_split_dist = self.stats
        if tree.merit_preprune:
            null_split = BranchFactory()
            best_suggestions.append(null_split)
        for att_id, splitter in self.splitters.items():
            best_suggestion = splitter.best_evaluated_split_suggestion(
                criterion, pre_split_dist, att_id, tree.binary_split, clean_leafs
            )
            best_suggestions.append(best_suggestion)

        return best_suggestions

    def disable_attribute(self, att_id):
        """Disable an attribute observer.

        Parameters
        ----------
        att_id
            Attribute index.

        """
        if att_id in self.splitters:
            del self.splitters[att_id]
            self._disabled_attrs.add(att_id)

    def learn_one(self, x, y, *, w=1.0, tree=None):
        """Update the node with the provided sample.

        Parameters
        ----------
        x
            Sample attributes for updating the node.
        y
            Target value.
        w
            Sample weight.
        tree
            Tree to update.

        Notes
        -----
        This base implementation defines the basic functioning of a learning node.
        All classes overriding this method should include a call to `super().learn_one`
        to guarantee the learning process happens consistently.
        """
        self.update_stats(y, w)
        if self.is_active():
            self.update_splitters(x, y, w, tree.nominal_attributes)

    @abc.abstractmethod
    def prediction(self, x, *, tree=None) -> dict:
        pass

    @abc.abstractmethod
    def calculate_promise(self) -> int:
        """Calculate node's promise.

        Returns
        -------
        int
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """


class LeafMajorityClassMono(HTLeafMono):
    """Leaf that always predicts the majority class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)

    @staticmethod
    def new_nominal_splitter():
        return NominalSplitterClassif()

    def update_stats(self, y, w):
        try:
            self.stats[y] += w
        except KeyError:
            self.stats[y] = w

    def prediction(self, x, *, tree=None):
        return normalize_values_in_dict(self.stats, inplace=False)

    @property
    def total_weight(self):
        """Calculate the total weight seen by the node.

        Returns
        -------
            Total weight seen.

        """
        return sum(self.stats.values()) if self.stats else 0

    def best_split_suggestions(self, criterion, tree) -> list[BranchFactory]:
        maj_class = max(self.stats.values())
        if maj_class and maj_class / self.total_weight > tree.max_share_to_split:
            return [BranchFactory()]

        return super().best_split_suggestions(criterion, tree)

    def aggregated_monotonicty_metadata(self):
        return self.splitter.split_monotonicty_metadata()

    def calculate_promise(self):
        """Calculate how likely a node is going to be split.

        A node with a (close to) pure class distribution will less likely be split.

        Returns
        -------
            A small value indicates that the node has seen more samples of a
            given class than the other classes.

        """
        total_seen = sum(self.stats.values())
        if total_seen > 0:
            return total_seen - max(self.stats.values())
        else:
            return 0

    def observed_class_distribution_is_pure(self):
        """Check if observed class distribution is pure, i.e. if all samples
        belong to the same class.

        Returns
        -------
            True if observed number of classes is less than 2, False otherwise.
        """
        count = 0
        for weight in self.stats.values():
            if weight != 0:
                count += 1
                if count == 2:
                    break
        return count < 2

    def __repr__(self):
        if not self.stats:
            return ""

        text = f"Class {max(self.stats, key=self.stats.get)}:"
        for label, proba in sorted(
            normalize_values_in_dict(self.stats, inplace=False).items()
        ):
            text += f"\n\tP({label}) = {round_sig_fig(proba)}"

        return text


from river.tree.splitter import GaussianSplitter
from river.tree.utils import BranchFactory
import numpy as np
import math


class DummyDist:
    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


class DummySplitter(GaussianSplitter):
    def __init__(self, n_splits: int = 10):
        super().__init__(n_splits)
        self._att_dist_per_class = {}

    def build_distributions(self, distributions, split_value, side, minpc, maxpc):
        for class_idx, dist in distributions.items():
            self._att_dist_per_class[class_idx] = DummyDist(dist.mu, dist.sigma)

            if side == 0:
                span_left = minpc[class_idx]
                span_right = split_value
            else:
                span_left = split_value
                span_right = maxpc[class_idx]

            self._att_dist_per_class[class_idx].mu = (span_right - span_left) / 2
            self._att_dist_per_class[class_idx].sigma = (span_right - span_left) / 6
            if self._att_dist_per_class[class_idx].sigma < 0:
                self._att_dist_per_class[class_idx].sigma = 0


class CustomGaussianSplitter(GaussianSplitter):
    def __init__(
        self,
        n_splits: int = 10,
        monotonicity_constraints: list[MonotonicityConstraint] = [],
        rho: float = 1e-5,
        stdevmul: float = 2.0,
        max_correction_factor: int = 10000,
    ):  # , fresh_fill_strategy=0):
        super().__init__(n_splits=n_splits)
        self.monotonicity_constraints = monotonicity_constraints
        self.rho = rho
        self.stdevmul = stdevmul
        self.max_correction_factor = max_correction_factor
        self.monotonic_correction_dict = {}

    def split_monotonicty_metadata(self):
        return self.monotonic_correction_dict

    def best_evaluated_split_suggestion(
        self, criterion, pre_split_dist, att_idx, binary_only, leafs
    ):
        best_suggestion = BranchFactory()
        suggested_split_values = self._split_point_suggestions()

        for split_value in suggested_split_values:
            post_split_dist = self._class_dists_from_binary_split(split_value)

            base_merit = criterion.merit_of_split(pre_split_dist, post_split_dist)

            correction = self._calculate_monotonicity_correction_factor(
                att_idx, split_value, pre_split_dist, post_split_dist, leafs
            )

            if correction > 0 and base_merit > 0:
                scaled_correction = min(
                    base_merit, self.max_correction_factor * correction
                )

                merit = max(0, base_merit + scaled_correction)
            else:
                merit = base_merit

            if merit > best_suggestion.merit:
                best_suggestion = BranchFactory(
                    merit, att_idx, split_value, post_split_dist
                )
                self.monotonic_correction_dict["best_suggestion"] = correction
                self.monotonic_correction_dict["greatest"] = (
                    self.monotonic_correction_dict["greatest"]
                    if "greatest" in self.monotonic_correction_dict
                    and self.monotonic_correction_dict["greatest"] > correction
                    else correction
                )

        return best_suggestion

    def _calculate_monotonicity_correction_factor(
        self, att_idx, split_value, pre_split_dist, post_split_dist, leafs
    ):
        simulatedBranchFactory = BranchFactory(
            -math.inf, att_idx, split_value, post_split_dist
        )
        simleafs = []
        if len(simulatedBranchFactory.children_stats) == 0:
            return 0

        self_clone = self.clone()
        self_clone._att_dist_per_class = self._att_dist_per_class

        cnt = 0
        for psd in simulatedBranchFactory.children_stats:
            ds = DummySplitter(self.n_splits)
            ds.build_distributions(
                self._att_dist_per_class,
                split_value,
                cnt,
                self._min_per_class,
                self._max_per_class,
            )
            simleafs.append(LeafMajorityClassMono(psd, 1, ds))
            cnt += 1

        total_leaves = len(leafs) + len(simleafs)
        matrix = np.zeros((total_leaves, total_leaves))
        denom = total_leaves * (total_leaves - 1)
        if denom == 0:
            return 0
        for i, leaf in enumerate(leafs + simleafs):
            for j, leaf2 in enumerate(leafs + simleafs):
                if i == j or i > j:
                    continue
                if self._is_monotonic(leaf, leaf2, att_idx):
                    matrix[i, j] = 1
        W = np.sum(matrix)
        I = W / denom
        Av = self._calculate_A(I)
        return self.rho * Av

    def _is_monotonic(self, leaf1, leaf2, att_idx):
        val = 1
        mcs = self.monotonicity_constraints
        for mc in mcs:
            if mc.feature == att_idx:
                if mc.monotonicity == 1:
                    val = val and self._is_increasing(leaf1, leaf2)
                    if val == 0:
                        return 1
                else:
                    val = val and self._is_decreasing(leaf1, leaf2)
                    if val == 0:
                        return 1
        return 1 - val

    def _is_increasing(self, leaf1, leaf2):
        if (
            len(leaf1.splitter._att_dist_per_class) == 0
            or len(leaf2.splitter._att_dist_per_class) == 0
        ):
            return 0
        l1p = leaf1.prediction(None)
        l2p = leaf2.prediction(None)
        max_l1p = max(l1p, key=lambda k: float(l1p[k]))
        max_l2p = max(l2p, key=lambda k: float(l2p[k]))

        l1distr = leaf1.splitter._att_dist_per_class[max_l1p]
        l2distr = leaf2.splitter._att_dist_per_class[max_l2p]

        return max_l1p > max_l2p and (l1distr.mu - l1distr.sigma * self.stdevmul) < (
            l2distr.mu + l2distr.sigma * self.stdevmul
        )

    def _is_decreasing(self, leaf1, leaf2):
        if (
            len(leaf1.splitter._att_dist_per_class) == 0
            or len(leaf2.splitter._att_dist_per_class) == 0
        ):
            return 0
        l1p = leaf1.prediction(None)
        l2p = leaf2.prediction(None)
        max_l1p = max(l1p, key=lambda k: float(l1p[k]))
        max_l2p = max(l2p, key=lambda k: float(l2p[k]))
        l1distr = leaf1.splitter._att_dist_per_class[max_l1p]
        l2distr = leaf2.splitter._att_dist_per_class[max_l2p]
        return max_l1p < max_l2p and (l1distr.mu + l1distr.sigma * self.stdevmul) > (
            l2distr.mu - l2distr.sigma * self.stdevmul
        )

    def _calculate_A(self, I):
        if I == 1:
            return 0
        else:
            return -1 / np.log2(I)


from river import base

from river.tree.hoeffding_tree import HoeffdingTree
from river.tree.nodes.branch import DTBranch
from river.tree.nodes.htc_nodes import LeafNaiveBayes, LeafNaiveBayesAdaptive
from river.tree.nodes.leaf import HTLeaf
from river.tree.split_criterion import (
    GiniSplitCriterion,
    HellingerDistanceCriterion,
    InfoGainSplitCriterion,
)
from river.tree.splitter import GaussianSplitter, Splitter
import random


class HoeffdingTreeClassifierMono(HoeffdingTree, base.Classifier):
    """Hoeffding Tree or Very Fast Decision Tree classifier.

    Parameters
    ----------
    grace_period
        Number of instances a leaf should observe between split attempts.
    max_depth
        The maximum depth a tree can reach. If `None`, the tree will grow indefinitely.
    split_criterion
        Split criterion to use.</br>
        - 'gini' - Gini</br>
        - 'info_gain' - Information Gain</br>
        - 'hellinger' - Helinger Distance</br>
    delta
        Significance level to calculate the Hoeffding bound. The significance level is given by
        `1 - delta`. Values closer to zero imply longer split decision delays.
    tau
        Threshold below which a split will be forced to break ties.
    leaf_prediction
        Prediction mechanism used at leafs.</br>
        - 'mc' - Majority Class</br>
        - 'nb' - Naive Bayes</br>
        - 'nba' - Naive Bayes Adaptive</br>
    nb_threshold
        Number of instances a leaf should observe before allowing Naive Bayes.
    nominal_attributes
        List of Nominal attributes identifiers. If empty, then assume that all numeric
        attributes should be treated as continuous.
    splitter
        The Splitter or Attribute Observer (AO) used to monitor the class statistics of numeric
        features and perform splits. Splitters are available in the `tree.splitter` module.
        Different splitters are available for classification and regression tasks. Classification
        and regression splitters can be distinguished by their property `is_target_class`.
        This is an advanced option. Special care must be taken when choosing different splitters.
        By default, `tree.splitter.GaussianSplitter` is used if `splitter` is `None`.
    binary_split
        If True, only allow binary splits.
    min_branch_fraction
        The minimum percentage of observed data required for branches resulting from split
        candidates. To validate a split candidate, at least two resulting branches must have
        a percentage of samples greater than `min_branch_fraction`. This criterion prevents
        unnecessary splits when the majority of instances are concentrated in a single branch.
    max_share_to_split
        Only perform a split in a leaf if the proportion of elements in the majority class is
        smaller than this parameter value. This parameter avoids performing splits when most
        of the data belongs to a single class.
    max_size
        The max size of the tree, in Megabytes (MB).
    memory_estimate_period
        Interval (number of processed instances) between memory consumption checks.
    stop_mem_management
        If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        If True, disable poor attributes to reduce memory usage.
    merit_preprune
        If True, enable merit-based tree pre-pruning.


    """

    _GINI_SPLIT = "gini"
    _INFO_GAIN_SPLIT = "info_gain"
    _HELLINGER_SPLIT = "hellinger"
    _VALID_SPLIT_CRITERIA = [_GINI_SPLIT, _INFO_GAIN_SPLIT, _HELLINGER_SPLIT]

    _MAJORITY_CLASS = "mc"
    _NAIVE_BAYES = "nb"
    _NAIVE_BAYES_ADAPTIVE = "nba"
    _VALID_LEAF_PREDICTION = [_MAJORITY_CLASS, _NAIVE_BAYES, _NAIVE_BAYES_ADAPTIVE]

    def __init__(
        self,
        monotonic_constrains: list[MonotonicityConstraint],
        rho: float = 1e-5,
        stdevmul: float = 2.0,
        grace_period: int = 200,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
    ):
        super().__init__(
            max_depth=max_depth,
            binary_split=binary_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )
        self.rho = rho
        self.stdevmul = stdevmul
        self.tree_indecies = []
        self.tree_leafs = []
        self.constrains = monotonic_constrains
        self.grace_period = grace_period
        self.split_criterion = split_criterion
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes

        if splitter is None:
            print("constraints")
            print(self.constrains)
            self.splitter = CustomGaussianSplitter(
                n_splits=10,
                monotonicity_constraints=self.constrains,
                rho=self.rho,
                stdevmul=self.stdevmul,
            )
        else:
            if not splitter.is_target_class:
                raise ValueError(
                    "The chosen splitter cannot be used in classification tasks."
                )
            self.splitter = splitter

        self.min_branch_fraction = min_branch_fraction
        self.max_share_to_split = max_share_to_split

        self.classes: set = set()

    @property
    def _mutable_attributes(self):
        return {"grace_period", "delta", "tau"}

    @HoeffdingTree.split_criterion.setter
    def split_criterion(self, split_criterion):
        if split_criterion not in self._VALID_SPLIT_CRITERIA:
            print(
                "Invalid split_criterion option {}', will use default '{}'".format(
                    split_criterion, self._INFO_GAIN_SPLIT
                )
            )
            self._split_criterion = self._INFO_GAIN_SPLIT
        else:
            self._split_criterion = split_criterion

    @HoeffdingTree.leaf_prediction.setter
    def leaf_prediction(self, leaf_prediction):
        if leaf_prediction not in self._VALID_LEAF_PREDICTION:
            print(
                "Invalid leaf_prediction option {}', will use default '{}'".format(
                    leaf_prediction, self._NAIVE_BAYES_ADAPTIVE
                )
            )
            self._leaf_prediction = self._NAIVE_BAYES_ADAPTIVE
        else:
            self._leaf_prediction = leaf_prediction

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}
        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return LeafMajorityClassMono(initial_stats, depth, self.splitter)
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return LeafNaiveBayes(initial_stats, depth, self.splitter)
        else:
            return LeafNaiveBayesAdaptive(initial_stats, depth, self.splitter)

    def _new_split_criterion(self):
        if self._split_criterion == self._GINI_SPLIT:
            split_criterion = GiniSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._INFO_GAIN_SPLIT:
            split_criterion = InfoGainSplitCriterion(self.min_branch_fraction)
        elif self._split_criterion == self._HELLINGER_SPLIT:
            split_criterion = HellingerDistanceCriterion(self.min_branch_fraction)
        else:
            split_criterion = InfoGainSplitCriterion(self.min_branch_fraction)

        return split_criterion

    def _attempt_to_split(
        self, leaf: HTLeaf | HTLeafMono, parent: DTBranch, parent_branch: int, **kwargs
    ):
        """Attempt to split a leaf."""
        if not leaf.observed_class_distribution_is_pure():
            split_criterion = self._new_split_criterion()

            best_split_suggestions = leaf.best_split_suggestions(split_criterion, self)
            best_split_suggestions.sort()
            should_split = False
            if len(best_split_suggestions) < 2:
                should_split = len(best_split_suggestions) > 0
            else:
                hoeffding_bound = self._hoeffding_bound(
                    split_criterion.range_of_merit(leaf.stats),
                    self.delta,
                    leaf.total_weight,
                )
                best_suggestion = best_split_suggestions[-1]
                second_best_suggestion = best_split_suggestions[-2]
                if (
                    best_suggestion.merit - second_best_suggestion.merit
                    > hoeffding_bound
                    or hoeffding_bound < self.tau
                ):
                    should_split = True
                if self.remove_poor_attrs:
                    poor_atts = set()
                    for suggestion in best_split_suggestions:
                        if (
                            suggestion.feature
                            and best_suggestion.merit - suggestion.merit
                            > hoeffding_bound
                        ):
                            poor_atts.add(suggestion.feature)
                    for poor_att in poor_atts:
                        leaf.disable_attribute(poor_att)
            if should_split:
                split_decision = best_split_suggestions[-1]
                if split_decision.feature is None:
                    leaf.deactivate()
                    self._n_inactive_leaves += 1
                    self._n_active_leaves -= 1
                else:
                    branch = self._branch_selector(
                        split_decision.numerical_feature, split_decision.multiway_split
                    )
                    leaves = tuple(
                        self._new_leaf(initial_stats, parent=leaf)
                        for initial_stats in split_decision.children_stats
                    )

                    new_split = split_decision.assemble(
                        branch, leaf.stats, leaf.depth, *leaves, **kwargs
                    )

                    self._n_active_leaves -= 1
                    self._n_active_leaves += len(leaves)
                    if parent is None:
                        self._root = new_split
                    else:
                        parent.children[parent_branch] = new_split

                self._enforce_size_limit()

    def learn_one(self, x, y, *, w=1.0):
        """Train the model on instance x and corresponding target y.

        Parameters
        ----------
        x
            Instance attributes.
        y
            Class label for sample x.
        w
            Sample weight.

        Notes
        -----
        Training tasks:

        * If the tree is empty, create a leaf node as the root.
        * If the tree is already initialized, find the corresponding leaf for
          the instance and update the leaf node statistics.
        * If growth is allowed and the number of instances that the leaf has
          observed between split attempts exceed the grace period then attempt
          to split.
        """

        self.classes.add(y)

        self._train_weight_seen_by_model += w

        if self._root is None:
            self._root = self._new_leaf()
            self._n_active_leaves = 1

        p_node = None
        node = None
        if isinstance(self._root, DTBranch):
            path = iter(self._root.walk(x, until_leaf=False))
            while True:
                aux = next(path, None)
                if aux is None:
                    break
                p_node = node
                node = aux
        else:
            node = self._root

        if isinstance(node, HTLeaf) or isinstance(node, HTLeafMono):
            node.learn_one(x, y, w=w, tree=self)
            if self._growth_allowed and node.is_active():
                if node.depth >= self.max_depth:
                    node.deactivate()
                    self._n_active_leaves -= 1
                    self._n_inactive_leaves += 1
                else:
                    weight_seen = node.total_weight
                    weight_diff = weight_seen - node.last_split_attempt_at
                    if weight_diff >= self.grace_period:
                        p_branch = (
                            p_node.branch_no(x)
                            if isinstance(p_node, DTBranch)
                            else None
                        )
                        self._attempt_to_split(node, p_node, p_branch)
                        node.last_split_attempt_at = weight_seen
        else:
            while True:
                if node.max_branches() == -1 and node.feature in x:
                    leaf = self._new_leaf(parent=node)
                    node.add_child(x[node.feature], leaf)
                    self._n_active_leaves += 1
                    node = leaf
                else:
                    _, node = node.most_common_path()
                    if isinstance(node, DTBranch):
                        node = node.traverse(x, until_leaf=False)
                if isinstance(node, HTLeaf):
                    break
            node.learn_one(x, y, w=w, tree=self)

        if self._train_weight_seen_by_model % self.memory_estimate_period == 0:
            self._estimate_model_size()

    def predict_proba_one(self, x):
        proba = {c: 0.0 for c in sorted(self.classes)}
        if self._root is not None:
            if isinstance(self._root, DTBranch):
                leaf = self._root.traverse(x, until_leaf=True)
            else:
                leaf = self._root

            proba.update(leaf.prediction(x, tree=self))
        return proba

    @property
    def _multiclass(self):
        return True

    def monotonicty_data(self) -> dict:
        """Return the aggregated monotonicity violations data from the tree."""
        if self._root is None:
            return {"greatest": 0, "best_suggestion": 0}

        leaves = self._find_leaves()

        aggregated_data = {"greatest": 0, "best_suggestion": 0}
        for leaf in leaves:
            if hasattr(leaf, "splitter") and hasattr(
                leaf.splitter, "monotonic_correction_dict"
            ):
                mono_data = leaf.splitter.monotonic_correction_dict
                if (
                    "greatest" in mono_data
                    and mono_data["greatest"] > aggregated_data["greatest"]
                ):
                    aggregated_data["greatest"] = mono_data["greatest"]
                if (
                    "best_suggestion" in mono_data
                    and mono_data["best_suggestion"]
                    > aggregated_data["best_suggestion"]
                ):
                    aggregated_data["best_suggestion"] = mono_data["best_suggestion"]

        return aggregated_data


from river.forest.adaptive_random_forest import BaseForest

import collections

import numpy as np

from river import base, metrics
from river.drift import ADWIN
from river.tree.nodes.arf_htc_nodes import (
    RandomLeafNaiveBayes,
    RandomLeafNaiveBayesAdaptive,
)


class BaseRandomLeafMono(HTLeafMono):
    """The Random Learning Node changes the way in which the attribute observers
    are updated (by using subsets of features).

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    max_features
        Number of attributes per subset for each node split.
    rng
        Random number generator.
    kwargs
        Other parameters passed to the learning node.
    """

    def __init__(self, stats, depth, splitter, max_features, rng, **kwargs):
        super().__init__(stats, depth, splitter, **kwargs)
        self.max_features = max_features
        self.rng = rng
        self.feature_indices = []

    def _iter_features(self, x) -> typing.Iterable:
        if len(self.feature_indices) == 0:
            self.feature_indices = self._sample_features(x, self.max_features)

        for att_id in self.feature_indices:
            if att_id in x:
                yield att_id, x[att_id]

    def _sample_features(self, x, max_features):
        return self.rng.sample(sorted(x.keys()), k=max_features)


class RandomLeafMajorityClassMono(BaseRandomLeafMono, LeafMajorityClassMono):
    """ARF learning node that always predicts the majority class.

    Parameters
    ----------
    stats
        Initial class observations.
    depth
        The depth of the node.
    splitter
        The numeric attribute observer algorithm used to monitor target statistics
        and perform split attempts.
    max_features
        Number of attributes per subset for each node split.
    rng
        Random number generator.
    kwargs
        Other parameters passed to the learning node.

    """

    def __init__(self, stats, depth, splitter, max_features, rng, **kwargs):
        super().__init__(stats, depth, splitter, max_features, rng, **kwargs)


class BaseTreeClassifier(HoeffdingTreeClassifierMono):
    """Adaptive Random Forest Hoeffding Tree Classifier.

    This is the base-estimator of the Adaptive Random Forest classifier.
    This variant of the Hoeffding Tree classifier includes the `max_features`
    parameter, which defines the number of randomly selected features to be
    considered at each split.

    """

    def __init__(
        self,
        max_features: int = 2,
        grace_period: int = 200,
        monotonic_constrains: list[MonotonicityConstraint] = None,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1000000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        rng: random.Random | None = None,
    ):
        super().__init__(
            monotonic_constrains=monotonic_constrains,
            grace_period=grace_period,
            max_depth=max_depth,
            split_criterion=split_criterion,
            delta=delta,
            tau=tau,
            leaf_prediction=leaf_prediction,
            nb_threshold=nb_threshold,
            nominal_attributes=nominal_attributes,
            splitter=splitter,
            binary_split=binary_split,
            min_branch_fraction=min_branch_fraction,
            max_share_to_split=max_share_to_split,
            max_size=max_size,
            memory_estimate_period=memory_estimate_period,
            stop_mem_management=stop_mem_management,
            remove_poor_attrs=remove_poor_attrs,
            merit_preprune=merit_preprune,
        )

        self.max_features = max_features
        self.rng = rng

    def _new_leaf(self, initial_stats=None, parent=None):
        if initial_stats is None:
            initial_stats = {}

        if parent is None:
            depth = 0
        else:
            depth = parent.depth + 1

        if self._leaf_prediction == self._MAJORITY_CLASS:
            return RandomLeafMajorityClassMono(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )
        elif self._leaf_prediction == self._NAIVE_BAYES:
            return RandomLeafNaiveBayes(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )
        else:
            return RandomLeafNaiveBayesAdaptive(
                initial_stats,
                depth,
                self.splitter,
                self.max_features,
                self.rng,
            )

    def get_monotonicity_data(self):
        """Wrapper to access monotonicity data from the tree."""
        return self.monotonicty_data()


class ARFClassifierMono(BaseForest, base.Classifier):
    """Adaptive Random Forest classifier.

    The 3 most important aspects of Adaptive Random Forest [^1] are:

    1. inducing diversity through re-sampling

    2. inducing diversity through randomly selecting subsets of features for
       node splits

    3. drift detectors per base tree, which cause selective resets in response
       to drifts

    It also allows training background trees, which start training if a
    warning is detected and replace the active tree if the warning escalates
    to a drift.

    Parameters
    ----------
    n_models
        Number of trees in the ensemble.
    max_features
        Max number of attributes for each node split.<br/>
        - If `int`, then consider `max_features` at each split.<br/>
        - If `float`, then `max_features` is a percentage and
          `int(max_features * n_features)` features are considered per split.<br/>
        - If "sqrt", then `max_features=sqrt(n_features)`.<br/>
        - If "log2", then `max_features=log2(n_features)`.<br/>
        - If None, then ``max_features=n_features``.
    lambda_value
        The lambda value for bagging (lambda=6 corresponds to Leveraging Bagging).
    metric
        Metric used to track trees performance within the ensemble.
        Defaults to `metrics.Accuracy()`.
    disable_weighted_vote
        If `True`, disables the weighted vote prediction.
    drift_detector
        Drift Detection method. Set to None to disable Drift detection.
        Defaults to `drift.ADWIN(delta=0.001)`.
    warning_detector
        Warning Detection method. Set to None to disable warning detection.
        Defaults to `drift.ADWIN(delta=0.01)`.
    grace_period
        [*Tree parameter*] Number of instances a leaf should observe between
        split attempts.
    max_depth
        [*Tree parameter*] The maximum depth a tree can reach. If `None`, the
        tree will grow indefinitely.
    split_criterion
        [*Tree parameter*] Split criterion to use.<br/>
        - 'gini' - Gini<br/>
        - 'info_gain' - Information Gain<br/>
        - 'hellinger' - Hellinger Distance
    delta
        [*Tree parameter*] Allowed error in split decision, a value closer to 0
        takes longer to decide.
    tau
        [*Tree parameter*] Threshold below which a split will be forced to break
        ties.
    leaf_prediction
        [*Tree parameter*] Prediction mechanism used at leafs.<br/>
        - 'mc' - Majority Class<br/>
        - 'nb' - Naive Bayes<br/>
        - 'nba' - Naive Bayes Adaptive
    nb_threshold
        [*Tree parameter*] Number of instances a leaf should observe before
        allowing Naive Bayes.
    nominal_attributes
        [*Tree parameter*] List of Nominal attributes. If empty, then assume that
        all attributes are numerical.
    splitter
        [*Tree parameter*] The Splitter or Attribute Observer (AO) used to monitor the class
        statistics of numeric features and perform splits. Splitters are available in the
        `tree.splitter` module. Different splitters are available for classification and
        regression tasks. Classification and regression splitters can be distinguished by their
        property `is_target_class`. This is an advanced option. Special care must be taken when
        choosing different splitters. By default, `tree.splitter.GaussianSplitter` is used
        if `splitter` is `None`.
    binary_split
        [*Tree parameter*] If True, only allow binary splits.
    min_branch_fraction
        [*Tree parameter*] The minimum percentage of observed data required for branches
        resulting from split candidates. To validate a split candidate, at least two resulting
        branches must have a percentage of samples greater than `min_branch_fraction`. This
        criterion prevents unnecessary splits when the majority of instances are concentrated
        in a single branch.
    max_share_to_split
        [*Tree parameter*] Only perform a split in a leaf if the proportion of elements
        in the majority class is smaller than this parameter value. This parameter avoids
        performing splits when most of the data belongs to a single class.
    max_size
        [*Tree parameter*] Maximum memory (MB) consumed by the tree.
    memory_estimate_period
        [*Tree parameter*] Number of instances between memory consumption checks.
    stop_mem_management
        [*Tree parameter*] If True, stop growing as soon as memory limit is hit.
    remove_poor_attrs
        [*Tree parameter*] If True, disable poor attributes to reduce memory usage.
    merit_preprune
        [*Tree parameter*] If True, enable merit-based tree pre-pruning.
    seed
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_models: int = 10,
        monotonic_constrains: list[MonotonicityConstraint] = None,
        max_features: bool | str | int = "sqrt",
        lambda_value: int = 6,
        metric: metrics.base.MultiClassMetric | None = None,
        disable_weighted_vote=False,
        drift_detector: base.DriftDetector | None = None,
        warning_detector: base.DriftDetector | None = None,
        grace_period: int = 200,
        max_depth: int | None = None,
        split_criterion: str = "info_gain",
        delta: float = 1e-7,
        tau: float = 0.05,
        leaf_prediction: str = "nba",
        nb_threshold: int = 0,
        nominal_attributes: list | None = None,
        splitter: Splitter | None = None,
        binary_split: bool = False,
        min_branch_fraction: float = 0.01,
        max_share_to_split: float = 0.99,
        max_size: float = 100.0,
        memory_estimate_period: int = 1_000_000,
        stop_mem_management: bool = False,
        remove_poor_attrs: bool = False,
        merit_preprune: bool = True,
        seed: int | None = None,
    ):
        super().__init__(
            n_models=n_models,
            max_features=max_features,
            lambda_value=lambda_value,
            metric=metric or metrics.Accuracy(),
            disable_weighted_vote=disable_weighted_vote,
            drift_detector=drift_detector or ADWIN(delta=0.001),
            warning_detector=warning_detector or ADWIN(delta=0.01),
            seed=seed,
        )

        self.monotonic_constrains = monotonic_constrains
        self.grace_period = grace_period
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self.delta = delta
        self.tau = tau
        self.leaf_prediction = leaf_prediction
        self.nb_threshold = nb_threshold
        self.nominal_attributes = nominal_attributes
        self.splitter = splitter
        self.binary_split = binary_split
        self.min_branch_fraction = min_branch_fraction
        self.max_share_to_split = max_share_to_split
        self.max_size = max_size
        self.memory_estimate_period = memory_estimate_period
        self.stop_mem_management = stop_mem_management
        self.remove_poor_attrs = remove_poor_attrs
        self.merit_preprune = merit_preprune

    @property
    def _mutable_attributes(self):
        return {
            "max_features",
            "lambda_value",
            "grace_period",
            "delta",
            "tau",
        }

    @property
    def _multiclass(self):
        return True

    def predict_proba_one(self, x: dict) -> dict[base.typing.ClfTarget, float]:
        y_pred: typing.Counter = collections.Counter()

        if len(self) == 0:
            self._init_ensemble(sorted(x.keys()))
            return y_pred

        for i, model in enumerate(self):
            y_proba_temp = model.predict_proba_one(x)
            metric_value = self._metrics[i].get()
            if not self.disable_weighted_vote and metric_value > 0.0:
                y_proba_temp = {
                    k: val * metric_value for k, val in y_proba_temp.items()
                }
            y_pred.update(y_proba_temp)

        total = sum(y_pred.values())
        if total > 0:
            return {label: proba / total for label, proba in y_pred.items()}
        return y_pred

    def _new_base_model(self):
        return BaseTreeClassifier(
            monotonic_constrains=self.monotonic_constrains,
            max_features=self.max_features,
            grace_period=self.grace_period,
            split_criterion=self.split_criterion,
            delta=self.delta,
            tau=self.tau,
            leaf_prediction=self.leaf_prediction,
            nb_threshold=self.nb_threshold,
            nominal_attributes=self.nominal_attributes,
            splitter=self.splitter,
            max_depth=self.max_depth,
            binary_split=self.binary_split,
            min_branch_fraction=self.min_branch_fraction,
            max_share_to_split=self.max_share_to_split,
            max_size=self.max_size,
            memory_estimate_period=self.memory_estimate_period,
            stop_mem_management=self.stop_mem_management,
            remove_poor_attrs=self.remove_poor_attrs,
            merit_preprune=self.merit_preprune,
            rng=self._rng,
        )

    def _drift_detector_input(
        self, tree_id: int, y_true: base.typing.ClfTarget, y_pred: base.typing.ClfTarget
    ) -> int | float:
        tree = self.models[tree_id]

        prediction_error = int(not y_true == y_pred)

        mono_data = tree.get_monotonicity_data()

        monotonicity_threshold = 0.05
        monotonicity_violation = 0

        if "best_suggestion" in mono_data:
            monotonicity_violation = int(
                mono_data["best_suggestion"] > monotonicity_threshold
            )

        weight_pred = 0.7
        weight_mono = 0.3
        combined_signal = (
            weight_pred * prediction_error + weight_mono * monotonicity_violation
        )

        return combined_signal


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
        model = ARFClassifierMono(
            monotonic_constrains=constraints,
            leaf_prediction="mc",
            split_criterion="info_gain",
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
