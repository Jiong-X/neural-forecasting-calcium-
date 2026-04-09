
import re

import torch
import torch.nn as nn

from torch.distributions import Normal
from dataclasses import dataclass, field
from typing import Optional, Any, Tuple
from abc import ABC, abstractmethod

@dataclass
class Prediction:
    mean: Optional[torch.Tensor] = None
    sigma: Optional[torch.Tensor] = None # standard deviation

    @property
    def output_type(self) -> str:
        cur_type = None
        if self.mean is not None:  
            if self.sigma is not None:
                cur_type = "probabilistic"
            else:
                cur_type = "point"

        return cur_type

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

    @abstractmethod
    def _hidden_call_(self, prediction:Prediction, targets:Any) -> Tuple[torch.tensor, dict[str, Optional[float]]]: ...

    @abstractmethod
    def _get_names(self) -> list[str]: ...

@dataclass
class Score:
    total: int = 0
    criterion: _MetricBase = None
    _scores: dict[str, Optional[float]] = field(default_factory=dict)

    def update(self, kwargs) -> None:

        total = kwargs.pop("total", None)
        self.total += total
        for key, value in kwargs.items():
            if value is None:
                continue
            self._scores[key] += (value * total)

    def get_scores(self) -> dict[str, Optional[float]]:
        return self._normalise()
    
    def _normalise(self) -> 'Score':
        normalised_scores = {"total":1}
        if self.total is not None and self.total > 0:
            for key in self._scores.keys():
                value = self._scores.get(key, None)
                if value is not None:
                    normalised_scores[key] = value / self.total
        
        ret_instance = self.create(self.criterion)
        ret_instance.update(normalised_scores)
        return ret_instance

    @classmethod
    def create(cls, criterion:_MetricBase) -> 'Score':
        """class factory method to create a Scores instance from a _MetricBase object
        allows correct tracking of metrics, as well as hooking onto derived metrics i.e. RMSE"
        """
        instance = cls(criterion=criterion)
        for name in criterion._get_names():
            instance._scores[name] = 0.0
        return instance

    def __str__(self) -> str:
        """Pretty print scores in aligned columns"""
        
        name_width = 4
        value_width = 6
        precision = 3

        parts = []
        for name, value in self._scores.items():
            parts.append(
                f"{name:<{name_width}} : {value:>{value_width}.{precision}f}"
            )
        
        return " | ".join(parts)

class _CriterionBase(_MetricBase):
    name: str
    acronym: str
    output_type = None

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

        if self.check_type(prediction, soft=True):
            loss = self.compute(prediction, targets)
            score[self.acronym] = loss.item()
            if self.derived:
                score.update(self._compute_derived_metrics(loss))

        return loss, score

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

    def check_type(self, prediction:Prediction, soft:bool=False) -> bool:
        """Check if prediction output type matches metric expectation to allow computation. If soft=False, raises ValueError; if soft=True, returns False."""
        if self.output_type:
            if prediction.output_type != self.output_type:
                if not soft:
                    raise ValueError(f"{self.name}: Expected output type '{self.output_type}', got '{prediction.output_type}'")
                else:
                    return False
        return True

class NllLoss(_CriterionBase):
    output_type = "probabilistic"
    name = "Negative Log-Likelihood"

    @staticmethod
    def nll_loss(dists: Normal, targets: list[torch.Tensor]) -> torch.Tensor:
        """Mean Gaussian NLL across all sessions."""
        total = -dists.log_prob(targets).mean()
        return total

    def compute(self, prediction:Prediction, targets:torch.Tensor) -> torch.Tensor:
        dists = Normal(prediction.mean, prediction.sigma)
        return self.nll_loss(dists, targets)

class MSELoss(_CriterionBase):
    output_type = None
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
    output_type = None
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

    def _get_names(self) -> list[str]:
        names = set()
        for metric in self.metrics:
            names.update(metric._get_names())
        if self.primary:
            names.update(self.primary._get_names())
        return names