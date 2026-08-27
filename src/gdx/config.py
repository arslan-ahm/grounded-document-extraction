"""Typed configuration: dataclasses, YAML with ``_base_`` inheritance, ``--set``.

Three properties are deliberate.

**Unknown keys are rejected.** A silently-swallowed typo
(``model.use_2d_pos`` vs ``model.use_2d_position``) would void an ablation while
producing a plausible-looking number. :func:`from_dict` raises instead.

**Every leaf is overridable from the command line** through ``--set
a.b.c=value``, with YAML-style value parsing so ``--set verify.enabled=false``
gives a bool and not the string ``"false"``.

**A run saves its own resolved config.** ``results/runs/<name>/config.yaml`` is
the fully-composed dictionary, not the file that was passed, so an experiment
can be replayed without knowing which ``_base_`` chain produced it.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from functools import cache
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin, get_type_hints

import yaml

T = TypeVar("T")


@dataclass
class DataConfig:
    """Synthetic document generator and split sizes."""

    n_train: int = 1200
    n_val: int = 300
    n_test: int = 400
    max_tokens: int = 192
    max_pages: int = 3
    multi_page_prob: float = 0.35
    # Difficulty knobs. Each one exists so a failure mode can be turned off and
    # its contribution measured, rather than being baked into "the dataset".
    box_jitter: float = 0.006
    rotation_deg: float = 1.2
    distractor_prob: float = 0.85
    duplicate_total_prob: float = 0.45
    missing_field_prob: float = 0.18
    # Probability that a date is written in a form requiring normalisation
    # ("3rd of Jan 2024" for target "2024-01-03"). Span selection structurally
    # cannot emit the target there; this is the measured cost of the ideology.
    verbose_date_prob: float = 0.30
    seed: int = 0
    split_seed_offset: int = 10_000
    real_dataset: str = ""
    real_root: str = "data/raw"


@dataclass
class ModelConfig:
    """Layout-aware encoder plus one of the two heads."""

    head: str = "span"  # "span" | "generative"
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    dropout: float = 0.1
    vocab_size: int = 512
    # 2-D layout signal. `use_2d_pos` adds the hand-rolled sinusoidal box
    # encoding; `use_spatial_bias` adds the relative-geometry attention bias.
    use_2d_pos: bool = True
    use_spatial_bias: bool = True
    n_pos_bands: int = 8
    n_spatial_buckets: int = 9
    max_span_len: int = 6
    # Generative head only.
    dec_hidden: int = 96
    dec_max_len: int = 20


@dataclass
class VerifyConfig:
    """The bounded verification loop."""

    enabled: bool = True
    max_iters: int = 3
    type_check: bool = True
    arithmetic: bool = True
    arithmetic_tol: float = 0.011
    abstain: bool = True
    # Confidence below which a field is abstained on even if all checks pass.
    # 0.0 disables the threshold so only hard check failures cause abstention.
    min_confidence: float = 0.0


@dataclass
class OptimConfig:
    """Training schedule."""

    epochs: int = 6
    batch_size: int = 16
    lr: float = 3e-3
    weight_decay: float = 1e-4
    warmup_frac: float = 0.1
    grad_clip: float = 1.0
    label_smoothing: float = 0.0


@dataclass
class RunConfig:
    """Bookkeeping for one run."""

    name: str = "gdx_span"
    seed: int = 0
    out_dir: str = "results/runs"
    n_threads: int = 2
    save_per_item: bool = True


@dataclass
class BaselineConfig:
    """Non-neural baselines."""

    heuristic_variant: str = "auto"  # "auto" selects on validation
    llm_backend: str = "stub"  # "stub" | "openai" | "anthropic"
    llm_model: str = "offline-stub"
    llm_temperature: float = 0.0


@dataclass
class Config:
    """Top-level configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    verify: VerifyConfig = field(default_factory=VerifyConfig)
    optim: OptimConfig = field(default_factory=OptimConfig)
    run: RunConfig = field(default_factory=RunConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        text = yaml.safe_dump(self.to_dict(), sort_keys=True, default_flow_style=False)
        p.write_text(text, encoding="utf-8", newline="\n")
        return p

    @property
    def run_dir(self) -> Path:
        return Path(self.run.out_dir) / self.run.name


def _coerce(value: Any, annotation: Any, path: str) -> Any:
    """Coerce a YAML scalar to the field's annotated type."""
    origin = get_origin(annotation)
    if origin is not None:
        args = [a for a in get_args(annotation) if a is not type(None)]
        if len(args) == 1:
            return _coerce(value, args[0], path)
        return value
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "yes", "1", "on"}:
                return True
            if low in {"false", "no", "0", "off"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        raise TypeError(f"{path}: cannot read {value!r} as bool")
    if annotation is int:
        if isinstance(value, bool):
            raise TypeError(f"{path}: refusing to read bool {value!r} as int")
        return int(value)
    if annotation is float:
        return float(value)
    if annotation is str:
        return str(value)
    return value


@cache
def _hints(cls: type) -> dict[str, Any]:
    """Resolved type hints for a dataclass.

    ``from __future__ import annotations`` makes every ``field.type`` a *string*,
    so a naive ``field.type is int`` check silently never fires and no coercion
    happens. Resolving hints once per class is the fix.
    """
    return get_type_hints(cls, globalns=globals())


def from_dict(cls: type[T], data: dict[str, Any], path: str = "") -> T:
    """Build a dataclass from a nested dict, rejecting unknown keys.

    Args:
        cls: A dataclass type.
        data: Nested mapping.
        path: Dotted prefix, used in error messages.

    Returns:
        An instance of ``cls``.

    Raises:
        KeyError: On a key that is not a field of the target dataclass. The
            message lists the valid keys, which is what makes a typo cheap to
            fix rather than a silent wrong experiment.
    """
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    known = {f.name: f for f in fields(cls)}
    unknown = sorted(set(data) - set(known))
    if unknown:
        prefix = f"{path}." if path else ""
        raise KeyError(
            f"unknown config key(s) {[prefix + u for u in unknown]}; "
            f"valid keys here: {sorted(known)}"
        )
    hints = _hints(cls)
    kwargs: dict[str, Any] = {}
    for name in known:
        if name not in data:
            continue
        value = data[name]
        annotation = hints.get(name, Any)
        sub_path = f"{path}.{name}" if path else name
        if is_dataclass(annotation):
            if not isinstance(value, dict):
                raise TypeError(f"{sub_path}: expected a mapping, got {type(value).__name__}")
            kwargs[name] = from_dict(annotation, value, sub_path)
        else:
            kwargs[name] = _coerce(value, annotation, sub_path)
    return cls(**kwargs)


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into ``base``, returning a new dict."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_yaml_with_bases(path: str | Path, _seen: tuple[Path, ...] = ()) -> dict[str, Any]:
    """Load a YAML config, resolving ``_base_`` inheritance.

    ``_base_`` may be a single relative path or a list of them; later entries
    win over earlier ones, and the current file wins over all of them.

    Raises:
        FileNotFoundError: If the file or one of its bases is missing.
        ValueError: On a cyclic ``_base_`` chain.
    """
    p = Path(path).resolve()
    if not p.exists():
        raise FileNotFoundError(f"config not found: {p}")
    if p in _seen:
        chain = " -> ".join(str(s.name) for s in (*_seen, p))
        raise ValueError(f"cyclic _base_ chain: {chain}")
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top level of a config must be a mapping")
    bases = raw.pop("_base_", [])
    if isinstance(bases, str):
        bases = [bases]
    merged: dict[str, Any] = {}
    for base in bases:
        merged = deep_merge(merged, load_yaml_with_bases(p.parent / base, (*_seen, p)))
    return deep_merge(merged, raw)


def parse_set_override(item: str) -> tuple[list[str], Any]:
    """Parse one ``a.b.c=value`` override into a key path and a parsed value.

    The value goes through ``yaml.safe_load`` so ``false``, ``3``, ``1e-3`` and
    ``[1, 2]`` all arrive as the right Python type.

    Raises:
        ValueError: If there is no ``=`` or the key path is empty.
    """
    if "=" not in item:
        raise ValueError(f"--set expects key=value, got {item!r}")
    key, _, raw = item.partition("=")
    key = key.strip()
    if not key:
        raise ValueError(f"--set expects a non-empty key, got {item!r}")
    try:
        value = yaml.safe_load(raw)
    except yaml.YAMLError:
        value = raw
    if value is None and raw.strip() != "":
        value = raw
    if isinstance(value, str):
        # YAML 1.1 does not recognise `3e-4` as a float -- it needs `3.0e-4` --
        # so `--set optim.lr=3e-4` would arrive as a string. The annotated type
        # would coerce it anyway, but only for float-typed fields; doing it here
        # means the parsed value is right regardless of where it lands.
        text = value.strip()
        try:
            value = int(text)
        except ValueError:
            try:
                value = float(text)
            except ValueError:
                value = text
    return key.split("."), value


def apply_overrides(data: dict[str, Any], overrides: list[str] | None) -> dict[str, Any]:
    """Apply a list of ``--set`` strings to a nested dict, returning a new dict."""
    out = copy.deepcopy(data)
    for item in overrides or []:
        path, value = parse_set_override(item)
        node = out
        for part in path[:-1]:
            nxt = node.get(part)
            if not isinstance(nxt, dict):
                nxt = {}
                node[part] = nxt
            node = nxt
        node[path[-1]] = value
    return out


def load_config(
    path: str | Path | None = None,
    overrides: list[str] | None = None,
    base: dict[str, Any] | None = None,
) -> Config:
    """Compose a :class:`Config` from a YAML file and ``--set`` overrides.

    Args:
        path: YAML config. ``None`` starts from dataclass defaults.
        overrides: ``["a.b=1", ...]``, applied last so the command line always
            wins.
        base: An extra dict merged under the file, for programmatic callers.

    Returns:
        A validated :class:`Config`.
    """
    data: dict[str, Any] = dict(base or {})
    if path is not None:
        data = deep_merge(data, load_yaml_with_bases(path))
    data = apply_overrides(data, overrides)
    return from_dict(Config, data)
