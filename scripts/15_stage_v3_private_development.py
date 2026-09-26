"""Copia exclusivamente los insumos privados autorizados para desarrollo v3.

Secuencia: --prepare-manifest; generar splits fuente con --source-only dentro
del clon; --stage-development. Nunca abre los manifests de test/calibración.
No inicia entrenamiento, piloto ni inferencia.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path, PureWindowsPath

import cv2

SPLITS = Path("results/publication_v3_5seed/splits")
CURATED_MANIFEST = Path("data/processed/busi_curated_manifest.csv")
BUS_BRA_LICENSE = Path("data/raw/BUS-BRA/BUSBRA/BUSBRA/LICENSE.txt")
STAGING_REPORT = Path("results/publication_v3_5seed/logs/private_staging_sha256.json")
FINAL_STAGING_REPORT = Path(
    "results/publication_v3_5seed/logs/private_finalization_staging_sha256.json"
)
DEFAULT_SEEDS = (17, 42, 73, 101, 202)
CURATED_MANIFEST_SHA256 = "74ce92d576695afaf04ee7c4df87beff1fb4931b98bd0eb91c1e5a72230cd9a8"
TARGET_ADAPT_SHA256 = "9f87bed663301a8d32379e85380631a6aebd2fb56defe81762b9087d783b13a8"
BUS_BRA_LICENSE_SHA256 = "80d9a1453638c386dfcf66e52284578bc9b49401481b28de386b4a0dda55a526"
FINETUNE_SHA256 = {
    17: "0284d1271441f9752a77c9351aadc7839a4ba033d001fff35556b7a63cf4f688",
    42: "e13a05eac52a3636de8f78d2cb2bd76b5beb704c80cf2f8f895d7b27c9bfe8b5",
    73: "868020b5f6e69fa96909e904803b11c109bddc6462ae0e125f7821330875a324",
    101: "b60105c9c461ae4ee253062d7194a84259bbb0d58a8399c8228a432ece7bc46e",
    202: "957b51d34ebb43f52fdf20f85694265bd8653fb352844a8b31e804b5f49df49f",
}
SOURCE_TEST_SHA256 = "381fd8dbc940c9880c7b884b02c102ad340f3227c8b1ca93f75934d1fb7d7f92"
FINALIZATION_INPUT_SHA256 = {
    "target_assignments.csv": "1bb2f974e1a470d12908fe410c9ecd4ca854b17f3901379e88294033098b6372",
    "assignment_hashes.json": "787ef047c3eec4121fcbec1b05e1d44f43b85399e12e12dd08ca6cff218ead02",
    "target_calibration_manifest.csv": (
        "4e300b59d2d961f44f2dd04bd697e1b22a4256028f40278d0cf71fd33bb23321"
    ),
    "target_test_manifest.csv": (
        "794969a11784500cff23d2b6020e23bce1b09845cb48b71c3c0ba304b4e1f542"
    ),
}
BUDGETS = (0.05, 0.10, 0.20)


def _roots(source_root: Path, clone_root: Path) -> tuple[Path, Path]:
    source = Path(source_root).resolve(strict=True)
    clone = Path(clone_root).resolve(strict=True)
    if source == clone or source in clone.parents or clone in source.parents:
        raise ValueError("El clon y el proyecto original deben ser directorios separados.")
    return source, clone


def _relative_path(value: str) -> Path:
    raw = str(value).strip().replace("\\", "/")
    windows = PureWindowsPath(raw)
    parts = raw.split("/")
    if not raw or windows.drive or windows.root or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Ruta fuera del proyecto o inválida: {value!r}")
    return Path(*parts)


def _contained(root: Path, relative: Path, *, exists: bool) -> Path:
    candidate = root / relative
    resolved = candidate.resolve(strict=exists)
    if not resolved.is_relative_to(root):
        raise ValueError(f"Ruta fuera del proyecto: {relative}")
    if exists and not resolved.is_file():
        raise ValueError(f"No es un archivo: {relative}")
    return resolved


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_sha256(path: Path, expected: str) -> str:
    observed = _sha256(path)
    if observed != expected.lower():
        raise ValueError(f"SHA-256 no aprobado para {path.name}: {observed} != {expected}")
    return observed


def _validate_source_metadata(clone: Path, expected_curated_sha256: str) -> None:
    metadata_path = _contained(clone, SPLITS / "source_split_metadata.json", exists=True)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    hashes = metadata.get("hashes", {})
    curated_path = _contained(clone, CURATED_MANIFEST, exists=True)
    curated_hash = _require_sha256(curated_path, expected_curated_sha256)
    if hashes.get("source_manifest_input") != curated_hash:
        raise ValueError("SHA-256 del manifiesto curado difiere de source_split_metadata.json")
    for partition in ("source_train", "source_val"):
        manifest_path = _contained(clone, SPLITS / f"{partition}_manifest.csv", exists=True)
        expected = metadata.get(f"{partition}_manifest_sha256")
        if not isinstance(expected, str):
            raise ValueError(f"Falta huella SHA-256 para {partition}")
        _require_sha256(manifest_path, expected)


def _validate_finetune(path: Path, adapt: list[dict[str, str]], seed: int) -> None:
    required = {
        "sample_id",
        "patient_id",
        "label_idx",
        "budget_fraction",
        "selected",
        "role",
        "selection_seed",
    }
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"finetune sin columnas {sorted(required)}: {path.name}")
        rows = list(reader)
    adapt_by_id = {row["sample_id"]: row for row in adapt}
    seen: dict[float, set[str]] = {budget: set() for budget in BUDGETS}
    for row in rows:
        try:
            budget = float(row["budget_fraction"])
            selection_seed = int(row["selection_seed"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"finetune con presupuesto o semilla inválidos: {path.name}"
            ) from error
        sample_id = row["sample_id"].strip()
        if budget not in seen or selection_seed != seed:
            raise ValueError(f"finetune con presupuesto o selection_seed inválidos: {path.name}")
        if sample_id not in adapt_by_id or sample_id in seen[budget]:
            raise ValueError(f"finetune con sample_id ajeno o duplicado: {path.name}")
        reference = adapt_by_id[sample_id]
        if (
            row["patient_id"] != reference["patient_id"]
            or row["label_idx"] != reference["label_idx"]
        ):
            raise ValueError(f"finetune con paciente o clase inconsistente: {path.name}")
        selected = row["selected"].strip().lower()
        role = row["role"].strip()
        if (
            selected not in {"true", "false"}
            or (selected == "true" and role not in {"train", "val"})
            or (selected == "false" and role)
        ):
            raise ValueError(f"finetune con selected o role inválidos: {path.name}")
        seen[budget].add(sample_id)
    if any(ids != set(adapt_by_id) for ids in seen.values()):
        raise ValueError(f"finetune incompleto para target_adapt: {path.name}")


def _read_manifest(path: Path, *, partition: str) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"sample_id", "image_path"}
        if not required.issubset(reader.fieldnames or ()):
            raise ValueError(f"Manifest {path.name} sin columnas {sorted(required)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Manifest vacío: {path.name}")
    ids = [row["sample_id"].strip() for row in rows]
    if any(not value for value in ids) or len(ids) != len(set(ids)):
        raise ValueError(f"sample_id vacío o duplicado en {path.name}")
    if any(row.get("partition", partition) != partition for row in rows):
        raise ValueError(f"Partición inconsistente en {path.name}")
    return rows


def _mask_paths(row: dict[str, str], *, source: bool) -> list[str]:
    if source and row.get("mask_paths", ""):
        return row["mask_paths"].split("|")
    raw = row.get("mask_path", "")
    return [raw] if raw else []


def _validate_pixels(source_root: Path, rows: list[dict[str, str]], *, source: bool) -> None:
    """Comprueba que los insumos ROI permitidos sean decodificables y compatibles."""
    for row in rows:
        image_rel = _relative_path(row["image_path"])
        image_path = _contained(source_root, image_rel, exists=True)
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"No se pudo decodificar imagen development: {image_rel}")
        masks = _mask_paths(row, source=source)
        if not masks:
            raise ValueError(f"Falta máscara ROI para imagen development: {image_rel}")
        for mask_value in masks:
            mask_rel = _relative_path(mask_value)
            mask_path = _contained(source_root, mask_rel, exists=True)
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None or mask.size == 0:
                raise ValueError(f"No se pudo decodificar máscara ROI: {mask_rel}")
            if mask.shape != image.shape[:2]:
                raise ValueError(f"Máscara ROI con dimensiones distintas de imagen: {mask_rel}")
            if not cv2.countNonZero(mask):
                raise ValueError(f"Máscara ROI vacía: {mask_rel}")


def _checked_copy(
    source_root: Path, clone_root: Path, relative: Path, *, expected_hash: str | None = None
) -> bool:
    origin = _contained(source_root, relative, exists=True)
    destination = _contained(clone_root, relative, exists=False)
    source_hash = (
        _sha256(origin) if expected_hash is None else _require_sha256(origin, expected_hash)
    )
    if destination.exists():
        if not destination.is_file() or _sha256(destination) != source_hash:
            raise ValueError(f"SHA-256 inconsistente en destino existente: {relative}")
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".stage-v3-", dir=destination.parent)
    try:
        with os.fdopen(fd, "wb") as output, origin.open("rb") as input_file:
            shutil.copyfileobj(input_file, output, length=1024 * 1024)
        if _sha256(Path(temporary)) != source_hash:
            raise ValueError(f"SHA-256 cambió durante la copia: {relative}")
        if destination.exists():
            raise ValueError(f"Destino apareció durante la copia: {relative}")
        os.replace(temporary, destination)
        if _sha256(destination) != source_hash:
            raise ValueError(f"SHA-256 posterior a copia no coincide: {relative}")
    finally:
        Path(temporary).unlink(missing_ok=True)
    return True


def prepare_manifest(source_root: Path, clone_root: Path) -> dict[str, object]:
    """Copia solo el índice Curated BUSI; no sigue las rutas de imagen internas."""
    source, clone = _roots(source_root, clone_root)
    copied = _checked_copy(
        source, clone, CURATED_MANIFEST, expected_hash=CURATED_MANIFEST_SHA256
    )
    return {"copied_files": int(copied), "manifest_sha256": _sha256(clone / CURATED_MANIFEST)}


def stage_development(
    source_root: Path,
    clone_root: Path,
    *,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
) -> dict[str, object]:
    """Copia los bytes train/val/adapt y los CSV de adaptación al clon."""
    source, clone = _roots(source_root, clone_root)
    if seeds != DEFAULT_SEEDS:
        raise ValueError(f"Las semillas deben ser exactamente {DEFAULT_SEEDS}.")
    _validate_source_metadata(clone, CURATED_MANIFEST_SHA256)
    _require_sha256(_contained(source, CURATED_MANIFEST, exists=True), CURATED_MANIFEST_SHA256)
    train = _read_manifest(
        _contained(clone, SPLITS / "source_train_manifest.csv", exists=True),
        partition="source_train",
    )
    val = _read_manifest(
        _contained(clone, SPLITS / "source_val_manifest.csv", exists=True),
        partition="source_val",
    )
    adapt_rel = SPLITS / "target_adapt_manifest.csv"
    adapt_path = _contained(source, adapt_rel, exists=True)
    _require_sha256(adapt_path, TARGET_ADAPT_SHA256)
    _require_sha256(_contained(source, BUS_BRA_LICENSE, exists=True), BUS_BRA_LICENSE_SHA256)
    adapt = _read_manifest(adapt_path, partition="target_adapt")
    if any(not row.get("patient_id") or not row.get("label_idx") for row in adapt):
        raise ValueError("target_adapt sin patient_id o label_idx")

    source_ids = {row["sample_id"] for row in train}
    if source_ids & {row["sample_id"] for row in val}:
        raise ValueError("source_train y source_val comparten sample_id")
    train_images = {_relative_path(row["image_path"]) for row in train}
    val_images = {_relative_path(row["image_path"]) for row in val}
    if train_images & val_images:
        raise ValueError("source_train y source_val comparten image_path")
    adapt_images = {_relative_path(row["image_path"]) for row in adapt}
    if (train_images | val_images) & adapt_images:
        raise ValueError("source y target_adapt comparten image_path")
    assignments: list[Path] = []
    for seed in seeds:
        relative = SPLITS / f"finetune_assignments_seed{int(seed)}.csv"
        path = _contained(source, relative, exists=True)
        if seed not in FINETUNE_SHA256:
            raise ValueError(f"Falta huella SHA-256 aprobada para semilla {seed}")
        _require_sha256(path, FINETUNE_SHA256[seed])
        _validate_finetune(path, adapt, seed)
        assignments.append(relative)

    # Preflight all allowed paths and destination conflicts before writing anything.
    pixels: set[Path] = set()
    for rows, source_rows in ((train, True), (val, True), (adapt, False)):
        for row in rows:
            pixels.add(_relative_path(row["image_path"]))
            for mask in _mask_paths(row, source=source_rows):
                pixels.add(_relative_path(mask))
    files = sorted(
        pixels | {adapt_rel, BUS_BRA_LICENSE, *assignments}, key=lambda item: item.as_posix()
    )
    hashes: dict[str, str] = {}
    for relative in files:
        origin = _contained(source, relative, exists=True)
        destination = _contained(clone, relative, exists=False)
        digest = _sha256(origin)
        if destination.exists() and (not destination.is_file() or _sha256(destination) != digest):
            raise ValueError(f"SHA-256 inconsistente en destino existente: {relative}")
        hashes[relative.as_posix()] = digest
    for rows, source_rows in ((train, True), (val, True), (adapt, False)):
        _validate_pixels(source, rows, source=source_rows)
    copied = sum(
        _checked_copy(source, clone, relative, expected_hash=hashes[relative.as_posix()])
        for relative in files
    )
    detailed_report = {"copied_files": copied, "verified_files": len(files), "sha256": hashes}
    report_path = _contained(clone, STAGING_REPORT, exists=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(detailed_report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".private-staging-", dir=report_path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(encoded)
        os.replace(temporary, report_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {
        "copied_files": copied,
        "verified_files": len(files),
        "report_path": str(STAGING_REPORT.as_posix()),
        "report_sha256": _sha256(report_path),
    }


def stage_finalization(source_root: Path, clone_root: Path) -> dict[str, object]:
    """Copia insumos finales congelados sin iniciar inferencia ni acceso lógico al test."""
    source, clone = _roots(source_root, clone_root)
    split_root = SPLITS
    source_test_path = _contained(clone, split_root / "source_test_manifest.csv", exists=True)
    _require_sha256(source_test_path, SOURCE_TEST_SHA256)
    source_test = _read_manifest(source_test_path, partition="source_test")
    adapt = _read_manifest(
        _contained(clone, split_root / "target_adapt_manifest.csv", exists=True),
        partition="target_adapt",
    )

    final_relatives = {
        name: split_root / name for name in FINALIZATION_INPUT_SHA256
    }
    for name, relative in final_relatives.items():
        _require_sha256(
            _contained(source, relative, exists=True), FINALIZATION_INPUT_SHA256[name]
        )
    calibration = _read_manifest(
        _contained(source, final_relatives["target_calibration_manifest.csv"], exists=True),
        partition="target_calibration",
    )
    target_test = _read_manifest(
        _contained(source, final_relatives["target_test_manifest.csv"], exists=True),
        partition="target_test",
    )
    for rows, name, require_patient in (
        (source_test, "source_test", False),
        (calibration, "target_calibration", True),
        (target_test, "target_test", True),
    ):
        required = {"label", "label_idx", "image_path"}
        if require_patient:
            required.add("patient_id")
        if any(not all(str(row.get(key, "")).strip() for key in required) for row in rows):
            raise ValueError(f"{name} carece de campos obligatorios: {sorted(required)}")

    target_sets = {
        name: {row["patient_id"] for row in rows}
        for name, rows in (
            ("adapt", adapt),
            ("calibration", calibration),
            ("test", target_test),
        )
    }
    if any(
        target_sets[left] & target_sets[right]
        for left, right in (("adapt", "calibration"), ("adapt", "test"), ("calibration", "test"))
    ):
        raise ValueError("Las particiones objetivo comparten patient_id")

    pixels: set[Path] = set()
    for rows, source_rows in ((source_test, True), (calibration, False), (target_test, False)):
        for row in rows:
            pixels.add(_relative_path(row["image_path"]))
            for mask in _mask_paths(row, source=source_rows):
                pixels.add(_relative_path(mask))
    files = sorted(
        pixels | set(final_relatives.values()), key=lambda item: item.as_posix()
    )
    hashes: dict[str, str] = {}
    for relative in files:
        origin = _contained(source, relative, exists=True)
        destination = _contained(clone, relative, exists=False)
        digest = _sha256(origin)
        if destination.exists() and (not destination.is_file() or _sha256(destination) != digest):
            raise ValueError(f"SHA-256 inconsistente en destino existente: {relative}")
        hashes[relative.as_posix()] = digest
    for rows, source_rows in ((source_test, True), (calibration, False), (target_test, False)):
        _validate_pixels(source, rows, source=source_rows)
    copied = sum(
        _checked_copy(source, clone, relative, expected_hash=hashes[relative.as_posix()])
        for relative in files
    )
    detailed_report = {
        "copied_files": copied,
        "verified_files": len(files),
        "source_test_images": len(source_test),
        "target_calibration_images": len(calibration),
        "target_calibration_patients": len(target_sets["calibration"]),
        "target_test_images": len(target_test),
        "target_test_patients": len(target_sets["test"]),
        "sha256": hashes,
        "test_access_started": False,
    }
    report_path = _contained(clone, FINAL_STAGING_REPORT, exists=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(detailed_report, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=".private-finalization-", dir=report_path.parent)
    try:
        with os.fdopen(fd, "wb") as output:
            output.write(encoded)
        os.replace(temporary, report_path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return {
        "copied_files": copied,
        "verified_files": len(files),
        "report_path": str(FINAL_STAGING_REPORT.as_posix()),
        "report_sha256": _sha256(report_path),
        "test_access_started": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--prepare-manifest", action="store_true")
    modes.add_argument("--stage-development", action="store_true")
    modes.add_argument("--stage-finalization", action="store_true")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--clone-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    args = parser.parse_args()
    if args.prepare_manifest:
        result = prepare_manifest(args.source_root, args.clone_root)
    elif args.stage_development:
        result = stage_development(args.source_root, args.clone_root, seeds=tuple(args.seeds))
    else:
        result = stage_finalization(args.source_root, args.clone_root)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
