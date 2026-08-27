"""Configuration: inheritance, override precedence, type coercion, and rejection.

The seam these tests protect is the one that silently voids experiments. A typo
in a config key that is *ignored* produces a plausible number for the wrong
setting, and nothing about the run looks wrong. So unknown keys raise, ``--set``
values are parsed into their annotated types, and the precedence order
(defaults < base < file < ``--set``) is asserted rather than assumed.
"""

from __future__ import annotations

import pytest
import yaml

from gdx.config import (
    Config,
    DataConfig,
    ModelConfig,
    apply_overrides,
    deep_merge,
    from_dict,
    load_config,
    load_yaml_with_bases,
    parse_set_override,
)

CONFIG_FILES = (
    "base", "smoke", "span", "generative", "no_2d_pos", "no_spatial_bias",
    "no_verify", "real_funsd",
)


# --- shipped configs -------------------------------------------------------

@pytest.mark.parametrize("name", CONFIG_FILES)
def test_shipped_config_loads(name):  # noqa: ANN001
    cfg = load_config(f"configs/{name}.yaml")
    assert isinstance(cfg, Config)
    assert cfg.model.head in ("span", "generative")
    assert cfg.data.n_train > 0
    assert cfg.optim.epochs > 0


def test_smoke_config_is_actually_small():
    """A smoke config that is not small defeats its purpose."""
    cfg = load_config("configs/smoke.yaml")
    assert cfg.data.n_train <= 100
    assert cfg.optim.epochs <= 3


def test_ablation_configs_differ_from_span_in_exactly_one_field():
    """An ablation that changes two things attributes nothing."""
    base = load_config("configs/span.yaml")
    for name, field_name in (
        ("no_2d_pos", "use_2d_pos"),
        ("no_spatial_bias", "use_spatial_bias"),
    ):
        cfg = load_config(f"configs/{name}.yaml")
        diffs = {
            k
            for k in vars(base.model)
            if getattr(base.model, k) != getattr(cfg.model, k)
        }
        assert diffs == {field_name}, f"{name} changed {diffs}"


def test_no_verify_config_changes_only_the_verify_switch():
    base = load_config("configs/span.yaml")
    cfg = load_config("configs/no_verify.yaml")
    diffs = {k for k in vars(base.verify) if getattr(base.verify, k) != getattr(cfg.verify, k)}
    assert diffs == {"enabled"}
    assert vars(base.model) == vars(cfg.model)


# --- unknown keys ----------------------------------------------------------

def test_unknown_top_level_key_raises():
    with pytest.raises(KeyError, match="unknown config key"):
        from_dict(Config, {"modle": {}})


def test_unknown_nested_key_raises_with_the_dotted_path():
    with pytest.raises(KeyError, match=r"model\.use_2d_position"):
        from_dict(Config, {"model": {"use_2d_position": True}})


def test_unknown_key_message_lists_the_valid_ones():
    with pytest.raises(KeyError) as excinfo:
        from_dict(ModelConfig, {"d_modell": 64})
    assert "d_model" in str(excinfo.value)


def test_a_scalar_where_a_mapping_belongs_raises():
    with pytest.raises(TypeError, match="expected a mapping"):
        from_dict(Config, {"model": 3})


# --- type coercion ---------------------------------------------------------

@pytest.mark.parametrize(
    ("raw", "expected"),
    [("true", True), ("false", False), ("yes", True), ("no", False), (1, True), (0, False)],
)
def test_bool_coercion(raw, expected):  # noqa: ANN001
    cfg = from_dict(ModelConfig, {"use_2d_pos": raw})
    assert cfg.use_2d_pos is expected


def test_bool_coercion_rejects_nonsense():
    with pytest.raises(TypeError, match="as bool"):
        from_dict(ModelConfig, {"use_2d_pos": "maybe"})


def test_int_field_rejects_a_bool():
    """``epochs: true`` is a mistake, not a request for one epoch."""
    with pytest.raises(TypeError, match="refusing to read bool"):
        from_dict(Config, {"optim": {"epochs": True}})


def test_int_and_float_coercion_from_strings():
    cfg = from_dict(Config, {"optim": {"epochs": "4", "lr": "1e-3"}})
    assert cfg.optim.epochs == 4
    assert isinstance(cfg.optim.epochs, int)
    assert cfg.optim.lr == pytest.approx(1e-3)
    assert isinstance(cfg.optim.lr, float)


def test_string_annotation_resolution_actually_fires():
    """``from __future__ import annotations`` makes field types strings.

    A naive implementation compares ``field.type is int``, which never matches,
    so nothing is coerced and a YAML string reaches the model as a string. This
    test is the regression guard for that.
    """
    cfg = from_dict(DataConfig, {"n_train": "17", "box_jitter": "0.5"})
    assert cfg.n_train == 17 and isinstance(cfg.n_train, int)
    assert cfg.box_jitter == 0.5 and isinstance(cfg.box_jitter, float)


# --- --set overrides -------------------------------------------------------

@pytest.mark.parametrize(
    ("item", "path", "value"),
    [
        ("a.b=1", ["a", "b"], 1),
        ("verify.enabled=false", ["verify", "enabled"], False),
        ("optim.lr=3e-4", ["optim", "lr"], 3e-4),
        ("run.name=hello", ["run", "name"], "hello"),
        ("x=[1, 2]", ["x"], [1, 2]),
    ],
)
def test_parse_set_override(item, path, value):  # noqa: ANN001
    got_path, got_value = parse_set_override(item)
    assert got_path == path
    assert got_value == value


def test_parse_set_override_requires_an_equals():
    with pytest.raises(ValueError, match="key=value"):
        parse_set_override("optim.lr")


def test_parse_set_override_requires_a_key():
    with pytest.raises(ValueError, match="non-empty key"):
        parse_set_override("=4")


def test_override_beats_the_file():
    cfg = load_config("configs/span.yaml", ["optim.epochs=99", "model.head=generative"])
    assert cfg.optim.epochs == 99
    assert cfg.model.head == "generative"


def test_override_creates_missing_intermediate_nodes():
    out = apply_overrides({}, ["a.b.c=5"])
    assert out == {"a": {"b": {"c": 5}}}


def test_apply_overrides_does_not_mutate_its_input():
    data = {"a": {"b": 1}}
    apply_overrides(data, ["a.b=2"])
    assert data == {"a": {"b": 1}}


# --- _base_ inheritance ----------------------------------------------------

def test_base_inheritance_chain(tmp_path):  # noqa: ANN001
    (tmp_path / "a.yaml").write_text("optim:\n  epochs: 1\n  lr: 0.1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\noptim:\n  epochs: 2\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("_base_: b.yaml\nrun:\n  name: c\n", encoding="utf-8")
    merged = load_yaml_with_bases(tmp_path / "c.yaml")
    assert merged["optim"] == {"epochs": 2, "lr": 0.1}
    assert merged["run"]["name"] == "c"


def test_multiple_bases_later_wins(tmp_path):  # noqa: ANN001
    (tmp_path / "a.yaml").write_text("optim:\n  epochs: 1\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("optim:\n  epochs: 2\n", encoding="utf-8")
    (tmp_path / "c.yaml").write_text("_base_: [a.yaml, b.yaml]\n", encoding="utf-8")
    assert load_yaml_with_bases(tmp_path / "c.yaml")["optim"]["epochs"] == 2


def test_cyclic_base_chain_raises(tmp_path):  # noqa: ANN001
    (tmp_path / "a.yaml").write_text("_base_: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("_base_: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="cyclic"):
        load_yaml_with_bases(tmp_path / "a.yaml")


def test_missing_config_raises():
    with pytest.raises(FileNotFoundError):
        load_config("configs/does-not-exist.yaml")


def test_non_mapping_config_raises(tmp_path):  # noqa: ANN001
    path = tmp_path / "bad.yaml"
    path.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_yaml_with_bases(path)


def test_empty_config_file_is_all_defaults(tmp_path):  # noqa: ANN001
    path = tmp_path / "empty.yaml"
    path.write_text("", encoding="utf-8")
    assert load_config(path).to_dict() == Config().to_dict()


# --- deep_merge ------------------------------------------------------------

def test_deep_merge_is_recursive_and_pure():
    base = {"a": {"b": 1, "c": 2}, "d": 3}
    override = {"a": {"b": 9}}
    out = deep_merge(base, override)
    assert out == {"a": {"b": 9, "c": 2}, "d": 3}
    assert base["a"]["b"] == 1


def test_deep_merge_replaces_a_scalar_with_a_mapping():
    assert deep_merge({"a": 1}, {"a": {"b": 2}}) == {"a": {"b": 2}}


# --- round trip ------------------------------------------------------------

def test_save_and_reload_round_trips(tmp_path):  # noqa: ANN001
    cfg = load_config("configs/span.yaml", ["optim.epochs=3"])
    path = cfg.save(tmp_path / "saved.yaml")
    assert path.read_bytes().count(b"\r\n") == 0, "config must be written with LF endings"
    reloaded = load_config(path)
    assert reloaded.to_dict() == cfg.to_dict()


def test_saved_config_is_fully_resolved(tmp_path):  # noqa: ANN001
    """A run's saved config must be replayable without its ``_base_`` chain."""
    cfg = load_config("configs/no_2d_pos.yaml")
    path = cfg.save(tmp_path / "resolved.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert "_base_" not in raw
    assert set(raw) == {"data", "model", "verify", "optim", "run", "baseline"}


def test_run_dir_is_out_dir_plus_name():
    cfg = load_config("configs/span.yaml", ["run.name=xyz", "run.out_dir=tmp/runs"])
    assert cfg.run_dir.as_posix() == "tmp/runs/xyz"
