"""Rechaza archivos privados o ajenos al paquete de código antes de publicarlo."""

from __future__ import annotations

import subprocess
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_BYTES = 1_000_000
ROOT_FILES = {
    ".gitattributes",
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    ".github/workflows/ci.yml",
}
CONFIG_FILES = {
    "config/config_publication.yaml",  # comparación histórica; nunca valor por defecto
    "config/config_publication_v3_5seed.yaml",
}
DOC_FILES = {
    "docs/PROTOCOL_PUBLICATION_V2.md",
    "docs/PROTOCOL_REVIEW_V3_5SEED_LOCAL.md",
}
RESOURCE_FILES = {
    "resources/busi_curation/README.md",
    "resources/busi_curation/mapping_curated_BUSI.csv",
    "resources/busi_curation/dataset_comment_list.csv",
    "resources/busi_curation/busi_pawlowska_sensitivity_audit.csv",
}
SCRIPT_FILES = {
    "scripts/02_preprocess.py",
    "scripts/10_curate_busi.py",
    "scripts/11_run_publication_experiments.py",
    "scripts/12_finalize_publication_inference.py",
    "scripts/13_analyze_publication_results.py",
    "scripts/14_generate_publication_artifacts.py",
    "scripts/check_public_scope.py",
}


def forbidden_path(filename: str) -> bool:
    """True si la ruta no pertenece a la lista explícita de publicación."""
    relative = filename.replace("\\", "/")
    path = PurePosixPath(relative)
    if (
        not relative
        or relative.startswith("/")
        or path.parts[0] in {".", ".."}
        or ":" in path.parts[0]
        or ".." in path.parts
    ):
        return True
    if relative in ROOT_FILES | CONFIG_FILES | DOC_FILES | RESOURCE_FILES | SCRIPT_FILES:
        return False
    return not (
        len(path.parts) >= 2
        and path.parts[0] in {"src", "tests"}
        and path.suffix == ".py"
    )


def main() -> int:
    command = ["git", "-c", f"safe.directory={ROOT.as_posix()}", "ls-files", "--stage", "-z"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, check=True)
    entries: list[tuple[str, bytes, bytes, bytes]] = []
    for item in result.stdout.split(b"\0"):
        if item:
            metadata, filename = item.split(b"\t", 1)
            mode, oid, stage = metadata.split()
            entries.append((filename.decode("utf-8"), mode, oid, stage))
    blob_command = [
        "git",
        "-c",
        f"safe.directory={ROOT.as_posix()}",
        "cat-file",
        "--batch-check=%(objecttype) %(objectsize)",
    ]
    blobs = subprocess.run(
        blob_command,
        cwd=ROOT,
        input=b"\n".join(oid for _, _, oid, _ in entries) + b"\n",
        capture_output=True,
        check=True,
    )
    blob_info = [line.split() for line in blobs.stdout.splitlines()]
    if len(blob_info) != len(entries):
        raise RuntimeError("El índice Git y la consulta de blobs no coinciden.")
    problems: list[str] = []
    for (filename, mode, _, stage), info in zip(entries, blob_info, strict=True):
        if forbidden_path(filename):
            problems.append(f"Fuera de la lista pública: {filename}")
        elif stage != b"0" or mode not in {b"100644", b"100755"} or info[0] != b"blob":
            problems.append(f"No es archivo regular: {filename}")
        elif int(info[1]) > MAX_FILE_BYTES:
            problems.append(f"Archivo demasiado grande: {filename}")
    for problem in problems:
        print(problem)
    if problems:
        return 1
    print(f"Inventario público verificado: {len(entries)} archivos Git.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
