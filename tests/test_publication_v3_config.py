"""Review-run configuration must preserve v2 and isolate new results."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def test_five_seed_review_config_isolated_from_v2() -> None:
    old = yaml.safe_load((ROOT / "config/config_publication.yaml").read_text(encoding="utf-8"))
    new = yaml.safe_load(
        (ROOT / "config/config_publication_v3_5seed.yaml").read_text(encoding="utf-8")
    )

    assert new["publication"]["seeds"] == [17, 42, 73, 101, 202]
    assert len(set(new["publication"]["seeds"])) == 5
    assert new["publication"]["protocol_version"] != old["publication"]["protocol_version"]
    assert new["paths"]["results"] == "results/publication_v3_5seed"
    assert all(
        value.startswith("results/publication_v3_5seed")
        for key, value in new["paths"].items()
        if key in {"results", "models", "figures", "reports"}
    )
    assert new["publication"]["split_seed"] == old["publication"]["split_seed"]
    audit_path = ROOT / new["publication"]["source_group_audit"]
    assert audit_path.is_file()

    for key in (
        "datasets",
        "task",
        "preprocessing",
        "augmentation",
        "model",
        "training",
        "evaluation",
    ):
        assert new[key] == old[key], key
    for key in (
        "architectures",
        "methods",
        "finetune_fractions",
        "adaptation",
        "finetune",
        "source",
        "locked_test",
    ):
        assert new["publication"][key] == old["publication"][key], key
