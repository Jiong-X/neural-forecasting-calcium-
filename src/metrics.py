
import re

import torch
import torch.nn as nn
import numpy as np

from torch.distributions import Normal, StudentT
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple
from abc import ABC, abstractmethod

@dataclass
class Prediction:
    mean: Optional[torch.Tensor] = field(default=None)
    logvar: Optional[torch.Tensor] = field(default=None) # log-variance
    df: Optional[torch.Tensor] = field(default=None) # degrees of freedom

    def is_valid(self, requirements: list) -> bool:
        for req in requirements:
            if getattr(self, req) is None:
                return False
        return True
    
    @property
    def sigma(self) -> Optional[torch.Tensor]:
        if self.logvar is not None:
            return (0.5 * self.logvar).exp().clamp(min=1e-4)
        else:
            return None

    @property
    def variance(self) -> Optional[torch.Tensor]:
        if self.sigma is not None:
            return self.sigma ** 2
        else:
            return None

class _MetricBase(ABC):
    """
    API for callable criterion to unify interface for Score object, and in training/eval loop
    """
    
    def __call__(self, prediction:Prediction, targets:Any) -> Tuple[torch.tensor, dict[str, Optional[float]]]:
        """
        interface to call criterion, adds total to the score dict
        """
        loss, score = self._hidden_call_(prediction=prediction, targets=targets)
        score["total"] = len(targets)
        return loss, score
    
    @property
    @abstractmethod
    def monitor_name(self) -> str: ...

    @abstractmethod
    def _hidden_call_(self, prediction:Prediction, targets:Any) -> Tuple[torch.tensor, dict[str, Optional[float]]]: ...

    @abstractmethod
    def _get_names(self) -> list[str]: ...

@dataclass
class Score:
    criterion: _MetricBase = field()
    total: int = field(default=0, init=False)
    scores: dict[str, Optional[float]] = field(default_factory=dict)

    def update(self, kwargs) -> None:

        total = kwargs.pop("total", None)
        self.total += total
        for key, value in kwargs.items():
            if value is None:
                continue
            self.scores[key] += (value * total)

    def get_scores(self) -> 'Score':
        """
        normalise scores over total training and return cleaned Score instance
        """
        normalised_scores = self._normalise()
        normalised_scores["total"] = 1
        ret_instance = self.create(self.criterion)
        ret_instance.update(normalised_scores)
        return ret_instance
    
    def _normalise(self) -> dict[str, Optional[float]]:
        """
        normalise scores, return dictionary of normalised scores
        """
        normalised_scores = {}
        if self.total is not None and self.total > 0:
            for key in self.scores.keys():
                value = self.scores.get(key, None)
                if value is not None:
                    normalised_scores[key] = value / self.total
        
        return normalised_scores

    def get_metric(self, name: str) -> Optional[float]:
        return self.scores.get(name, None)
    
    @classmethod
    def create(cls, criterion:_MetricBase) -> 'Score':
        """class factory method to create a Scores instance from a _MetricBase object
        allows correct tracking of metrics, as well as hooking onto derived metrics i.e. RMSE"
        """
        instance = cls(criterion=criterion)
        for name in criterion._get_names():
            instance.scores[name] = 0.0
        return instance

    def __str__(self) -> str:
        """Pretty print scores in aligned columns"""
        
        name_width = 4
        value_width = 6
        precision = 3

        parts = []
        for name, value in self.scores.items():
            parts.append(
                f"{name:<{name_width}} : {value:>{value_width}.{precision}f}"
            )
        
        return " | ".join(parts)
    
@dataclass
class ScoreTracker:
    
    criterion: _MetricBase = field()

    val_scores: dict[str, Optional[list[float]]] = field(default_factory=dict)
    train_scores: dict[str, Optional[list[float]]] = field(default_factory=dict)
    test_scores: dict[str, Optional[float]] = field(default_factory=dict)

    epoch_width: int = field(init=False)
    metric_widths: dict[str, int] = field(init=False)
    metric_names: list[str] = field(init=False)

    @classmethod
    def create(cls, criterion:_MetricBase) -> 'ScoreTracker':
        """class factory method to create a Scores instance from a _MetricBase object
        allows correct tracking of metrics, as well as hooking onto derived metrics i.e. RMSE"
        """
        instance = cls(criterion=criterion)
        instance.metric_names = criterion._get_names()
        for name in instance.metric_names:
            instance.val_scores[name] = []
            instance.train_scores[name] = []
        
        instance.epoch_width, instance.metric_widths = instance._column_specs()
        return instance

    def update(self, score:Score, flag:str) -> None:

        train_flags = ("train", "training")
        val_flags = ("val", "validation", "eval", "evaluation")
        test_flags = ("test", "testing")

        if score.total != 1:
            score = score.get_scores()

        flag = flag.lower()

        if flag in train_flags:
            target_scores = self.train_scores
            for key, value in score.scores.items():
                target_scores[key].append(value)

        elif flag in val_flags:
            target_scores = self.val_scores
            for key, value in score.scores.items():
                target_scores[key].append(value)

        elif flag in test_flags:
            for key, value in score.scores.items():
                self.test_scores[key] = value
        else:
            raise ValueError(
                f"Invalid flag '{flag}' for ScoreTracker.update, "
                f"must be one of {train_flags + val_flags + test_flags}"
            )
        
    def to_save_dict(self) -> dict[str, np.ndarray]:
        save_dict = {}
        loss_key = self.criterion.monitor_name

        for key, values in self.train_scores.items():
            if key == loss_key:
                save_dict["train_losses"] = np.array(values, dtype=float)
            save_dict[f"train_{key}"] = np.array(values, dtype=float)

        for key, values in self.val_scores.items():
            if key == loss_key:
                save_dict["val_losses"] = np.array(values, dtype=float)
            save_dict[f"val_{key}"] = np.array(values, dtype=float)

        for key, value in self.test_scores.items():
            save_dict[f"test_{key}"] = np.array(np.nan if value is None else value, dtype=float)

        return save_dict

    def _column_specs(self) -> tuple[int, dict[str, int]]:
        """
        Return:
            epoch_width: width of epoch column
            metric_widths: width for each metric column, keyed by metric name

        Width is chosen to fit both:
        - the headline label, e.g. 'Train NLL'
        - the printed numeric value, e.g. '123.4567' or 'n/a'
        """
        epoch_width = max(len("Epoch"), 5)

        metric_widths = {}
        for name in self.metric_names:
            header_train = f"Train {name}"
            header_val = f"Val {name}"

            # room for typical printed numeric values
            # e.g. '-123.4567' is length 9
            numeric_width = 10
            metric_widths[name] = max(len(header_train), len(header_val), numeric_width)

        return epoch_width, metric_widths

    @staticmethod
    def _format_value(value: Optional[float], width: int, precision: int = 4) -> str:
        if value is None:
            return f"{'n/a':>{width}}"
        return f"{value:>{width}.{precision}f}"

    def print_headline(self) -> None:
        parts = [f"{'Epoch':>{self.epoch_width}}"]

        for name in self.metric_names:
            parts.append(f"{('Train ' + name):>{self.metric_widths[name]}}")

        for name in self.metric_names:
            parts.append(f"{('Val ' + name):>{self.metric_widths[name]}}")

        line = " | ".join(parts)
        print(line)
        print("-" * len(line))

    def print_latest(self, epoch: int, tag = "") -> None:
        parts = [f"{epoch:>{self.epoch_width}d}"]

        for name in self.metric_names:
            parts.append(
                self._format_value(self.train_scores[name][-1], self.metric_widths[name])
            )

        for name in self.metric_names:
            parts.append(
                self._format_value(self.val_scores[name][-1], self.metric_widths[name])
            )

        msg = " | ".join(parts)
        print(f"{msg}{tag}")

class _CriterionBase(_MetricBase):
    requirements:list
    name: str
    acronym: str

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Require subclasses to define name
        if not hasattr(cls, "name"):
            raise TypeError(f"{cls.__name__} must define a 'name' class attribute")

        # Compute acronym once at class creation
        words = re.split(r'[^A-Za-z]+', cls.name)
        cls.acronym = ''.join(w[0].upper() for w in words if w)

    def __init__(self):
        self.derived = False

    def _hidden_call_(self, prediction:Prediction, targets:Any) -> Tuple[torch.tensor, dict[str, Optional[float]]]:
        loss = None
        score = {self.acronym: None}

        #soft = True
        if prediction.is_valid(requirements=self.requirements):
            loss = self.compute(prediction, targets)
            score[self.acronym] = loss.item()
            if self.derived:
                score.update(self._compute_derived_metrics(loss))
        # elif soft == False:
        #    raise ValueError(f"{self.name}: Expected output type needs {self.requirements}, got {prediction}")

        return loss, score

    @property
    def monitor_name(self) -> str:
        return self.acronym

    def _get_names(self) -> list[str]:
        names = [self.acronym]
        if self.derived:
            names.extend(self._derived_metrics)
        return names

    def _derived_metrics(self) -> Optional[list[str]]:
        return []

    def _compute_derived_metrics(self) -> Optional[dict[str, torch.Tensor]]:
        return None

    @abstractmethod
    def compute(self, prediction:Prediction, targets:Any) -> torch.Tensor: ...

class _NllLoss(_CriterionBase):
    requirements = ["mean", "sigma"]
    name = "Negative Log-Likelihood"

    @staticmethod
    def nll_loss(dists: Normal, targets: torch.Tensor) -> torch.Tensor:
        """Mean Gaussian NLL across all sessions."""
        total = -dists.log_prob(targets).mean()
        return total

    @abstractmethod
    def calc_distr(self, prediction:Prediction) -> torch.distributions.Distribution: ...

    def compute(self, prediction:Prediction, targets:torch.Tensor) -> torch.Tensor:
        dists = self.calc_distr(prediction)
        return self.nll_loss(dists, targets)

class GaussianNllLoss(_NllLoss):
    name = "Gaussian Negative Log-Likelihood"
    def calc_distr(self, prediction:Prediction) -> torch.distributions.Distribution:
        return Normal(prediction.mean, prediction.sigma)

class StudentTNllLoss(_NllLoss):
    requirements = ["mean", "sigma", "df"]
    name = "Student-T Negative Log-Likelihood"
    def calc_distr(self, prediction:Prediction) -> torch.distributions.Distribution:
        return StudentT(prediction.df, prediction.mean, prediction.sigma)
    
class MSELoss(_CriterionBase):
    requirements = ["mean"]
    name = "Mean Squared Error"

    def __init__(self, RMSE:bool=False):
        """
        if RMSE=True, also computes Root Mean Squared Error as a derived metric, 
        accessible via Scores; otherwise only MSE is computed.
        """
        super().__init__()
        self.derived = RMSE

    def _derived_metrics(self) -> Optional[list[str]]:
        return ["RMSE"]

    @staticmethod    
    def _compute_rmse(mse:torch.Tensor) -> torch.Tensor:
        return torch.sqrt(mse)

    def _compute_derived_metrics(self) -> dict[str, torch.Tensor]:
        return {"RMSE": self._compute_rmse().item()}

    def compute(self, prediction:Prediction, targets:torch.Tensor) -> torch.Tensor:
        return nn.MSELoss(prediction.mean, targets)

class MAELoss(_CriterionBase):
    requirements = ["mean"]
    name = "Mean Absolute Error"

    @staticmethod
    def MAE_loss(pred:torch.Tensor, targets:torch.Tensor) -> torch.Tensor:
        return (pred - targets).abs().mean()

    def compute(self, prediction:Prediction, targets:torch.Tensor) -> torch.Tensor:
        return self.MAE_loss(prediction.mean, targets)

class MetricSuite(_MetricBase):
    def __init__(self, metrics:list[_CriterionBase], primary:_CriterionBase=None):
        """
        suite to enable comparison for a list of different criterion functions.

        inputs:
            metrics - list[_CriterionBase]: list of criterion to execute
            primary - _CriterionBase: Primary criterion to set if you want a returned loss tensor to train on
        """
        self.metrics = metrics
        self.primary = primary
        if primary in metrics:
            self.metrics.remove(primary)

    

    def _hidden_call_(self, prediction:Prediction, targets:Any) -> Tuple[torch.tensor, dict[str, Optional[float]]]:
        """
        call to execute suite of criterion

        inputs:
            prediction - Prediction: data regarding model outputs to give to criterion
            targets - Any: target value to compare against
        
        outputs:
            Tuple[Loss, scores]
                loss - torch.tensor: loss tensor for .backwards if primary metric is given to instance
                scores - dict[str, Optional[float]]: dictionary to pass to Score object, contains results for all run criterion
        """
        results = {}
        loss = None
        for metric in self.metrics:
            _, score = metric._hidden_call_(prediction, targets)
            results.update(score)
        if self.primary:
            loss, score = self.primary._hidden_call_(prediction, targets)
            results.update(score)

        return loss, results

    @property
    def monitor_name(self) -> str:
        if self.primary is None:
            raise ValueError("MetricSuite has no primary metric, so no monitor_name is defined")
        return self.primary.monitor_name

    def _get_names(self) -> list[str]:
        names = set()
        for metric in self.metrics:
            names.update(metric._get_names())
        if self.primary:
            names.update(self.primary._get_names())
        return names

        