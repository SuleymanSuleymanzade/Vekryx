"""Typed index construction and query settings, plus immutable built snapshots."""
from dataclasses import asdict, dataclass, field
import json

from .embeddings import _vector
from .search import FusionHit, HybridOptions, MmrOptions, SearchHit, SearchOptions


def _integer(value, name, minimum=1, maximum=None):
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(name + " is outside its supported integer range")


@dataclass(frozen=True)
class HnswConfig:
    m: int = 16
    ef_construction: int = 100
    seed: int = 42

    def __post_init__(self):
        _integer(self.m, "m", 2)
        _integer(self.ef_construction, "ef_construction", self.m)
        _integer(self.seed, "seed", 0, 2**64 - 1)


@dataclass(frozen=True)
class HnswSearchOptions:
    ef_search: int = 64

    def __post_init__(self):
        _integer(self.ef_search, "ef_search")


@dataclass(frozen=True)
class IvfConfig:
    nlist: int = 16
    iterations: int = 20
    seed: int = 42

    def __post_init__(self):
        _integer(self.nlist, "nlist")
        _integer(self.iterations, "iterations")
        _integer(self.seed, "seed", 0, 2**64 - 1)


@dataclass(frozen=True)
class IvfSearchOptions:
    nprobe: int = 1

    def __post_init__(self):
        _integer(self.nprobe, "nprobe")


@dataclass(frozen=True)
class PqConfig:
    subquantizers: int = 4
    bits: int = 4
    iterations: int = 20
    seed: int = 42

    def __post_init__(self):
        _integer(self.subquantizers, "subquantizers")
        _integer(self.bits, "bits", 1, 8)
        _integer(self.iterations, "iterations")
        _integer(self.seed, "seed", 0, 2**64 - 1)


@dataclass(frozen=True)
class IvfPqConfig:
    ivf: IvfConfig = field(default_factory=IvfConfig)
    pq: PqConfig = field(default_factory=PqConfig)

    def __post_init__(self):
        if not isinstance(self.ivf, IvfConfig) or not isinstance(self.pq, PqConfig):
            raise TypeError("ivf and pq must be IvfConfig and PqConfig")


_CONFIGS = {HnswConfig: "hnsw", IvfConfig: "ivf", PqConfig: "pq", IvfPqConfig: "ivf_pq"}
_OPTIONS = {"hnsw": HnswSearchOptions, "ivf": IvfSearchOptions,
            "ivf_pq": IvfSearchOptions, "pq": type(None)}


def _config_json(config):
    if type(config) not in _CONFIGS:
        raise TypeError("config must be HnswConfig, IvfConfig, PqConfig, or IvfPqConfig")
    return json.dumps({"algorithm": _CONFIGS[type(config)], "config": asdict(config)}, allow_nan=False)


class SearchIndex:
    """Immutable snapshot of eligible records, bound metric, and trained backend.

    Changes to the source store do not affect this snapshot. Build a new index
    after updates. The embedding model is retained for query encoding; index and
    model state are not part of store snapshots.
    """
    def __init__(self, native, embedding_model, algorithm):
        self._native = native
        self._embedding_model = embedding_model
        self._algorithm = algorithm

    def __len__(self):
        return len(self._native)

    @property
    def guarantee(self):
        return self._native.guarantee

    def _query(self, query, options, method):
        if not isinstance(options, SearchOptions):
            raise TypeError("options must be SearchOptions")
        expected = _OPTIONS[self._algorithm]
        backend = options.backend
        if backend is None and expected is not type(None):
            backend = expected()
        if type(backend) is not expected:
            raise ValueError("query options do not match the built index backend")
        payload = asdict(options)
        payload["backend"] = None if backend is None else asdict(backend)
        if isinstance(query, str):
            if self._embedding_model is None:
                raise ValueError("text input requires an embedding model")
            query = _vector(self._embedding_model.embed_query(query))
        return self._native.query(query, json.dumps(payload, allow_nan=False),
                                  json.dumps(method, allow_nan=False))

    @staticmethod
    def _hits(rows):
        return [SearchHit(id, value, "exact" if exact else "approximate")
                for id, value, exact in rows]

    def search(self, query, options=None):
        options = SearchOptions() if options is None else options
        return self._hits(self._query(query, options, {"method": "vector"}))

    def mmr_search(self, query, options=None, *, mmr=None):
        options = SearchOptions(candidates=20) if options is None else options
        mmr = MmrOptions() if mmr is None else mmr
        return self._hits(self._query(query, options, dict(method="mmr", **asdict(mmr))))

    def hybrid_search(self, query, options=None, *, hybrid=None):
        if not isinstance(query, str):
            raise TypeError("hybrid_search query must be text")
        options = SearchOptions() if options is None else options
        hybrid = HybridOptions() if hybrid is None else hybrid
        rows = self._query(query, options, dict(method="hybrid", text=query, **asdict(hybrid)))
        return [FusionHit(id, value) for id, value, _ in rows]
