"""Retrieval options and named results shared by the public search methods.

Backend query settings belong to built indexes; store methods remain exact.
"""
from dataclasses import dataclass
from math import isfinite
from typing import Optional


@dataclass(frozen=True)
class SearchOptions:
    k: int = 5
    candidates: Optional[int] = None
    rerank_exact: bool = True
    backend: object = None

    def __post_init__(self):
        if type(self.k) is not int or self.k < 0:
            raise ValueError("k must be a nonnegative integer")
        if self.candidates is None:
            object.__setattr__(self, "candidates", self.k)
        if type(self.candidates) is not int or self.candidates < self.k:
            raise ValueError("candidates must be an integer greater than or equal to k")
        if type(self.rerank_exact) is not bool:
            raise ValueError("rerank_exact must be a bool")


@dataclass(frozen=True)
class MmrOptions:
    lambda_mult: float = 0.5

    def __post_init__(self):
        _unit_interval(self.lambda_mult, "lambda_mult")


@dataclass(frozen=True)
class HybridOptions:
    vector_weight: float = 0.5
    rrf_k: int = 60
    keyword_candidates: Optional[int] = None

    def __post_init__(self):
        _unit_interval(self.vector_weight, "vector_weight")
        if type(self.rrf_k) is not int or self.rrf_k <= 0:
            raise ValueError("rrf_k must be a positive integer")
        if self.keyword_candidates is not None and (
            type(self.keyword_candidates) is not int or self.keyword_candidates < 0
        ):
            raise ValueError("keyword_candidates must be a nonnegative integer or None")


@dataclass(frozen=True)
class SearchHit:
    id: str
    distance: float
    distance_kind: str


@dataclass(frozen=True)
class FusionHit:
    id: str
    fusion_score: float


def _unit_interval(value, name):
    if type(value) not in (int, float) or not isfinite(value) or not 0 <= value <= 1:
        raise ValueError(name + " must be finite and between 0 and 1")
