"""El inventario Git debe seguir siendo exclusivamente código y mapeos BUSI públicos."""

import subprocess

from scripts import check_public_scope as scope

forbidden_path = scope.forbidden_path


def test_public_scope_allows_only_the_selected_busi_tables() -> None:
    for path in (
        "README.md",
        ".github/workflows/ci.yml",
        "config/config_publication_v3_5seed.yaml",
        "src/data/publication_splits.py",
        "scripts/11_run_publication_experiments.py",
        "tests/test_public_scope.py",
        "resources/busi_curation/mapping_curated_BUSI.csv",
        "resources/busi_curation/dataset_comment_list.csv",
        "resources/busi_curation/busi_pawlowska_sensitivity_audit.csv",
    ):
        assert not forbidden_path(path), path


def test_public_scope_rejects_patient_data_and_editorial_artifacts() -> None:
    for path in (
        "data/raw/image.png",
        "results/publication_v3_5seed/checkpoint.pt",
        "publication_package/manuscrito_final.md",
        "Latex Doc/OJEMB/main.tex",
        "resources/busi_curation/target_test_manifest.csv",
        "docs/article.pdf",
        "src/private.json",
        "config/kaggle.json",
        "config/.env",
        "foo/../src/training/train.py",
    ):
        assert forbidden_path(path), path


def test_public_scope_checks_staged_blob_not_worktree_size(tmp_path, monkeypatch, capsys) -> None:
    """Un archivo grande en el índice no se vuelve publicable al reducir la copia local."""
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    readme = tmp_path / "README.md"
    readme.write_bytes(b"x" * (scope.MAX_FILE_BYTES + 1))
    subprocess.run(["git", "add", "README.md"], cwd=tmp_path, check=True)
    readme.write_text("Copia local pequeña, índice todavía grande.\n", encoding="utf-8")

    monkeypatch.setattr(scope, "ROOT", tmp_path)
    assert scope.main() == 1
    assert "Archivo demasiado grande: README.md" in capsys.readouterr().out
