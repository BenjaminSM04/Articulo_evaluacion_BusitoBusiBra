"""Standalone BUSI curation for the publication cohort and sensitivity audit.

The primary cohort is Curated BUSI v1 (2026), distributed by its authors through
Zenodo with a two-column mapping and a curated copy of each image/mask.  The
secondary sensitivity cohort follows the supplementary audit published by
Pawłowska, Karwat, and Żołek (2023), whose grouped fields use braces and ``&``.

Both sources are validated against frozen counts and cryptographic provenance.
No image is copied or modified.  Output paths are POSIX-style paths relative to
the project root, so the manifests are portable and do not disclose a local
workstation path.
"""

from __future__ import annotations

import csv
import hashlib
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Mapping, Sequence

DATASET_SOURCE_DOI = "10.1016/j.dib.2019.104863"
PAWLOWSKA_SOURCE_DOI = "10.1016/j.dib.2023.109247"
CURATED_BUSI_SOURCE_DOI = "10.5281/zenodo.19047974"
CURATED_BUSI_MAPPING_COMMIT = "54687ba5f2cf4378d36fc5d762bd15c6068d2bd4"
CURATED_BUSI_MAPPING_SHA256 = "4aa2096b4b8ef34d3dc03ddcd513f059020601ef89f2dc06741214f1e197f848"

EXPECTED_SOURCE_TOTAL = 780
EXPECTED_SOURCE_CLASS_COUNTS: Mapping[str, int] = {
    "benign": 437,
    "malignant": 210,
    "normal": 133,
}
EXPECTED_CURATED_CLASS_COUNTS: Mapping[str, int] = {
    "benign": 296,
    "malignant": 161,
}
EXPECTED_OFFICIAL_TOTAL = 450
EXPECTED_OFFICIAL_CLASS_COUNTS: Mapping[str, int] = {
    "benign": 222,
    "malignant": 164,
    "normal": 64,
}
EXPECTED_OFFICIAL_BINARY_CLASS_COUNTS: Mapping[str, int] = {
    "benign": 222,
    "malignant": 164,
}

LABEL_INDEX: Mapping[str, int] = {"benign": 0, "malignant": 1}
LABEL_GLOBAL_OFFSETS: Mapping[str, int] = {
    "benign": 0,
    "malignant": 437,
    "normal": 647,
}
EXCLUDED_OBJECTIONS = frozenset({"axilla", "needle", "multiclass"})
ALLOWED_OBJECTIONS = EXCLUDED_OBJECTIONS | {""}
IMAGE_EXTENSIONS = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})

COMMENT_COLUMNS = ("ID", "Filename", "Objection", "Annotation")
CURATED_MAPPING_COLUMNS = ("class", "id")
FILENAME_RE = re.compile(r"^(benign|malignant|normal) \((\d+)\)$", re.IGNORECASE)

MANIFEST_COLUMNS = (
    "manifest_id",
    "dataset",
    "curation_variant",
    "group_id",
    "image_id",
    "class_id",
    "ids",
    "filenames",
    "filename",
    "label",
    "label_idx",
    "patient_id",
    "image_path",
    "mask_path",
    "mask_paths",
    "n_masks",
    "annotation",
    "annotation_tags",
    "objection",
    "image_sha256",
    "mask_sha256s",
    "classifier_image_policy",
    "classifier_mask_usage",
    "verification_image_path",
    "verification_mask_path",
    "verification_image_sha256",
    "verification_mask_sha256",
    "source_doi",
    "dataset_source_doi",
    "source_commit",
    "source_metadata_path",
    "source_metadata_sha256",
)

AUDIT_COLUMNS = (
    "curation_variant",
    "group_id",
    "source_row",
    "image_id",
    "class_id",
    "member_position",
    "ids",
    "filenames",
    "filename",
    "label",
    "label_idx",
    "objection",
    "annotation",
    "annotation_tags",
    "group_eligible",
    "selected_image_id",
    "decision",
    "decision_reason",
    "image_path",
    "mask_path",
    "mask_paths",
    "n_masks",
    "image_sha256",
    "mask_sha256s",
    "classifier_image_policy",
    "classifier_mask_usage",
    "verification_image_path",
    "verification_mask_path",
    "verification_image_sha256",
    "verification_mask_sha256",
    "source_doi",
    "dataset_source_doi",
    "source_commit",
    "source_metadata_path",
    "source_metadata_sha256",
)


class BusiCurationError(ValueError):
    """Raised when the source audit or the local BUSI files are inconsistent."""


@dataclass(frozen=True)
class BusiMember:
    """One original BUSI image represented in the curation source."""

    image_id: int
    filename: str
    label: str
    class_number: int


@dataclass(frozen=True)
class BusiGroup:
    """One source row containing one or more related BUSI images."""

    group_id: str
    source_row: int
    members: tuple[BusiMember, ...]
    objection: str
    annotation_tags: tuple[str, ...]

    @property
    def ids(self) -> tuple[int, ...]:
        return tuple(member.image_id for member in self.members)

    @property
    def filenames(self) -> tuple[str, ...]:
        return tuple(member.filename for member in self.members)


@dataclass(frozen=True)
class BusiDecision:
    """Image-level decision derived from a curation group."""

    group: BusiGroup
    member: BusiMember
    member_position: int
    group_eligible: bool
    selected_image_id: int | None
    decision: str
    decision_reasons: tuple[str, ...]


@dataclass(frozen=True)
class CuratedMappingEntry:
    """One image listed in the official Curated BUSI v1 mapping."""

    source_row: int
    label: str
    class_number: int

    @property
    def image_key(self) -> str:
        return f"{self.label}_id_{self.class_number}"

    @property
    def group_id(self) -> str:
        return f"CBUSI-{self.label.upper()}-{self.class_number:04d}"

    @property
    def original_filename(self) -> str:
        return f"{self.label} ({self.class_number})"

    @property
    def original_global_id(self) -> int:
        return LABEL_GLOBAL_OFFSETS[self.label] + self.class_number


@dataclass(frozen=True)
class FileMetadata:
    """Portable paths and hashes for one raw image and all of its masks."""

    image_path: str
    mask_path: str
    mask_paths: tuple[str, ...]
    image_sha256: str
    mask_sha256s: tuple[str, ...]


def parse_braced_ampersand_list(value: str, *, field: str, source_row: int) -> tuple[str, ...]:
    """Parse ``{item&item}`` fields while preserving their source order."""

    raw = value.strip()
    if len(raw) < 2 or not raw.startswith("{") or not raw.endswith("}"):
        raise BusiCurationError(f"Fila {source_row}: {field} debe usar llaves, recibido {value!r}.")
    inner = raw[1:-1].strip()
    if not inner:
        raise BusiCurationError(f"Fila {source_row}: {field} no puede estar vacío.")
    items = tuple(item.strip() for item in inner.split("&"))
    if any(not item for item in items):
        raise BusiCurationError(
            f"Fila {source_row}: {field} contiene un elemento vacío: {value!r}."
        )
    return items


def parse_annotation_tags(value: str) -> tuple[str, ...]:
    """Parse the optional ampersand-separated annotation field."""

    raw = value.strip()
    if not raw:
        return ()
    if raw.startswith("{") or raw.endswith("}"):
        if not (raw.startswith("{") and raw.endswith("}")):
            raise BusiCurationError(f"Annotation tiene llaves desbalanceadas: {value!r}.")
        raw = raw[1:-1].strip()
    tags = tuple(tag.strip().lower() for tag in raw.split("&"))
    if any(not tag for tag in tags):
        raise BusiCurationError(f"Annotation contiene una etiqueta vacía: {value!r}.")
    return tags


def _parse_member(image_id: str, filename: str, *, source_row: int) -> BusiMember:
    try:
        numeric_id = int(image_id)
    except ValueError as exc:
        raise BusiCurationError(f"Fila {source_row}: ID no entero {image_id!r}.") from exc

    match = FILENAME_RE.fullmatch(filename)
    if match is None:
        raise BusiCurationError(f"Fila {source_row}: Filename BUSI inválido {filename!r}.")
    label = match.group(1).lower()
    class_number = int(match.group(2))
    return BusiMember(
        image_id=numeric_id,
        filename=filename,
        label=label,
        class_number=class_number,
    )


def load_comment_groups(path: str | Path) -> list[BusiGroup]:
    """Read the semicolon-delimited source and expand its grouped fields."""

    source_path = Path(path)
    if not source_path.is_file():
        raise FileNotFoundError(f"No existe la lista de curación BUSI: {source_path}")

    groups: list[BusiGroup] = []
    with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None or tuple(reader.fieldnames) != COMMENT_COLUMNS:
            raise BusiCurationError(
                "Cabecera inesperada. Se requiere exactamente "
                f"{';'.join(COMMENT_COLUMNS)}, recibido {reader.fieldnames!r}."
            )

        for source_row, row in enumerate(reader, start=2):
            ids = parse_braced_ampersand_list(row["ID"], field="ID", source_row=source_row)
            filenames = parse_braced_ampersand_list(
                row["Filename"], field="Filename", source_row=source_row
            )
            if len(ids) != len(filenames):
                raise BusiCurationError(
                    f"Fila {source_row}: {len(ids)} IDs pero " f"{len(filenames)} filenames."
                )

            objection = row["Objection"].strip().lower()
            if objection not in ALLOWED_OBJECTIONS:
                raise BusiCurationError(
                    f"Fila {source_row}: Objection desconocida {objection!r}; "
                    f"permitidas: {sorted(ALLOWED_OBJECTIONS)!r}."
                )

            members = tuple(
                _parse_member(image_id, filename, source_row=source_row)
                for image_id, filename in zip(ids, filenames, strict=True)
            )
            group_id = f"BUSI-G{min(member.image_id for member in members):04d}"
            groups.append(
                BusiGroup(
                    group_id=group_id,
                    source_row=source_row,
                    members=members,
                    objection=objection,
                    annotation_tags=parse_annotation_tags(row["Annotation"]),
                )
            )

    if not groups:
        raise BusiCurationError(f"La lista de curación está vacía: {source_path}")
    return groups


def load_curated_mapping(path: str | Path) -> list[CuratedMappingEntry]:
    """Read the official semicolon-delimited Curated BUSI v1 mapping."""

    mapping_path = Path(path)
    if not mapping_path.is_file():
        raise FileNotFoundError(f"No existe el mapping oficial Curated BUSI: {mapping_path}")

    entries: list[CuratedMappingEntry] = []
    with mapping_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None or tuple(reader.fieldnames) != CURATED_MAPPING_COLUMNS:
            raise BusiCurationError(
                "Cabecera inesperada del mapping Curated BUSI. Se requiere exactamente "
                f"{';'.join(CURATED_MAPPING_COLUMNS)}, recibido {reader.fieldnames!r}."
            )
        for source_row, row in enumerate(reader, start=2):
            label = row["class"].strip().lower()
            if label not in EXPECTED_SOURCE_CLASS_COUNTS:
                raise BusiCurationError(
                    f"Fila {source_row}: clase desconocida en mapping oficial {label!r}."
                )
            try:
                class_number = int(row["id"].strip())
            except ValueError as exc:
                raise BusiCurationError(f"Fila {source_row}: id no entero {row['id']!r}.") from exc
            entries.append(
                CuratedMappingEntry(
                    source_row=source_row,
                    label=label,
                    class_number=class_number,
                )
            )

    if not entries:
        raise BusiCurationError(f"El mapping oficial está vacío: {mapping_path}")
    return entries


def validate_curated_mapping(
    entries: Sequence[CuratedMappingEntry],
    *,
    mapping_path: str | Path,
) -> dict[str, int]:
    """Verify the frozen Curated BUSI v1 mapping counts, IDs, and SHA-256."""

    class_counts = Counter(entry.label for entry in entries)
    keys = [(entry.label, entry.class_number) for entry in entries]
    duplicate_keys = sorted(key for key, count in Counter(keys).items() if count > 1)
    mapping_sha256 = sha256_file(mapping_path)
    problems: list[str] = []

    if len(entries) != EXPECTED_OFFICIAL_TOTAL:
        problems.append(f"total={len(entries)}, esperado={EXPECTED_OFFICIAL_TOTAL}")
    if dict(class_counts) != dict(EXPECTED_OFFICIAL_CLASS_COUNTS):
        problems.append(
            f"conteos por clase={dict(class_counts)}, "
            f"esperados={dict(EXPECTED_OFFICIAL_CLASS_COUNTS)}"
        )
    if duplicate_keys:
        problems.append(f"pares class/id duplicados={duplicate_keys[:10]}")
    if mapping_sha256 != CURATED_BUSI_MAPPING_SHA256:
        problems.append(f"SHA-256={mapping_sha256}, esperado={CURATED_BUSI_MAPPING_SHA256}")

    for entry in entries:
        maximum = EXPECTED_SOURCE_CLASS_COUNTS[entry.label]
        if not 1 <= entry.class_number <= maximum:
            problems.append(
                f"id fuera de rango: {entry.label}/{entry.class_number}, "
                f"rango permitido=1..{maximum}"
            )
            break

    binary_counts = {label: class_counts[label] for label in EXPECTED_OFFICIAL_BINARY_CLASS_COUNTS}
    if binary_counts != dict(EXPECTED_OFFICIAL_BINARY_CLASS_COUNTS):
        problems.append(
            f"composición binaria={binary_counts}, "
            f"esperada={dict(EXPECTED_OFFICIAL_BINARY_CLASS_COUNTS)}"
        )

    if problems:
        raise BusiCurationError(
            "El mapping oficial Curated BUSI no coincide con la versión congelada: "
            + "; ".join(problems)
        )
    return dict(class_counts)


def validate_source_coverage(
    groups: Sequence[BusiGroup],
    *,
    expected_total: int = EXPECTED_SOURCE_TOTAL,
    expected_class_counts: Mapping[str, int] = EXPECTED_SOURCE_CLASS_COUNTS,
) -> dict[str, int]:
    """Verify IDs, filenames, class totals, and within-class numbering exactly."""

    members = [member for group in groups for member in group.members]
    ids = [member.image_id for member in members]
    filenames = [member.filename.casefold() for member in members]
    class_counts = Counter(member.label for member in members)

    duplicate_ids = sorted(image_id for image_id, count in Counter(ids).items() if count > 1)
    duplicate_filenames = sorted(
        filename for filename, count in Counter(filenames).items() if count > 1
    )
    expected_ids = set(range(1, expected_total + 1))
    observed_ids = set(ids)
    missing_ids = sorted(expected_ids - observed_ids)
    unexpected_ids = sorted(observed_ids - expected_ids)

    problems: list[str] = []
    if len(members) != expected_total:
        problems.append(f"total={len(members)}, esperado={expected_total}")
    if duplicate_ids:
        problems.append(f"IDs duplicados={duplicate_ids[:10]}")
    if missing_ids:
        problems.append(f"IDs faltantes={missing_ids[:10]}")
    if unexpected_ids:
        problems.append(f"IDs fuera de rango={unexpected_ids[:10]}")
    if duplicate_filenames:
        problems.append(f"filenames duplicados={duplicate_filenames[:10]}")
    if dict(class_counts) != dict(expected_class_counts):
        problems.append(
            f"conteos por clase={dict(class_counts)}, " f"esperados={dict(expected_class_counts)}"
        )

    offset = 0
    for label, expected_count in expected_class_counts.items():
        observed_numbers = {member.class_number for member in members if member.label == label}
        expected_numbers = set(range(1, expected_count + 1))
        if observed_numbers != expected_numbers:
            missing = sorted(expected_numbers - observed_numbers)
            unexpected = sorted(observed_numbers - expected_numbers)
            problems.append(
                f"numeración {label}: faltantes={missing[:10]}, "
                f"fuera de rango={unexpected[:10]}"
            )

        for member in members:
            if member.label != label:
                continue
            expected_global_id = offset + member.class_number
            if member.image_id != expected_global_id:
                problems.append(
                    f"mapeo ID/filename inconsistente: ID {member.image_id}, "
                    f"{member.filename!r}, esperado ID {expected_global_id}"
                )
                break
        offset += expected_count

    group_ids = [group.group_id for group in groups]
    duplicate_group_ids = sorted(
        group_id for group_id, count in Counter(group_ids).items() if count > 1
    )
    if duplicate_group_ids:
        problems.append(f"group_id duplicados={duplicate_group_ids[:10]}")

    if problems:
        raise BusiCurationError("La lista no cubre BUSI exactamente: " + "; ".join(problems))
    return dict(class_counts)


def decide_binary_curation(groups: Sequence[BusiGroup]) -> list[BusiDecision]:
    """Apply the published group-level exclusions and first-file rule."""

    decisions: list[BusiDecision] = []
    for group in groups:
        exclusion_reasons: list[str] = []
        if any(member.label == "normal" for member in group.members):
            exclusion_reasons.append("normal_class")
        if group.objection in EXCLUDED_OBJECTIONS:
            exclusion_reasons.append(f"objection_{group.objection}")

        eligible = not exclusion_reasons
        selected_image_id = group.members[0].image_id if eligible else None
        for member_position, member in enumerate(group.members, start=1):
            if not eligible:
                decision = "exclude"
                reasons = tuple(exclusion_reasons)
            elif member_position == 1:
                decision = "keep"
                reasons = ("first_in_eligible_group",)
            else:
                decision = "exclude"
                reasons = ("duplicate_group_member",)

            decisions.append(
                BusiDecision(
                    group=group,
                    member=member,
                    member_position=member_position,
                    group_eligible=eligible,
                    selected_image_id=selected_image_id,
                    decision=decision,
                    decision_reasons=reasons,
                )
            )
    return decisions


def validate_curated_selection(
    decisions: Sequence[BusiDecision],
    *,
    expected_class_counts: Mapping[str, int] = EXPECTED_CURATED_CLASS_COUNTS,
) -> dict[str, int]:
    """Verify that the kept binary subset has exactly the expected composition."""

    kept = [decision for decision in decisions if decision.decision == "keep"]
    class_counts = Counter(decision.member.label for decision in kept)
    expected_total = sum(expected_class_counts.values())

    problems: list[str] = []
    if len(kept) != expected_total:
        problems.append(f"total curado={len(kept)}, esperado={expected_total}")
    if dict(class_counts) != dict(expected_class_counts):
        problems.append(
            f"conteos curados={dict(class_counts)}, " f"esperados={dict(expected_class_counts)}"
        )
    if any(decision.member.label == "normal" for decision in kept):
        problems.append("se conservaron imágenes normal")
    if any(decision.group.objection in EXCLUDED_OBJECTIONS for decision in kept):
        problems.append("se conservaron grupos con objeción excluyente")

    eligible_groups = {decision.group.group_id for decision in decisions if decision.group_eligible}
    kept_by_group = Counter(decision.group.group_id for decision in kept)
    invalid_kept_groups = sorted(
        group_id for group_id in eligible_groups if kept_by_group.get(group_id, 0) != 1
    )
    if invalid_kept_groups:
        problems.append(
            "grupos elegibles sin exactamente un representante=" f"{invalid_kept_groups[:10]}"
        )

    if problems:
        raise BusiCurationError(
            "La selección binaria no coincide con la curación esperada: " + "; ".join(problems)
        )
    return dict(class_counts)


@lru_cache(maxsize=None)
def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a lowercase SHA-256 digest without loading the whole file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=None)
def _directory_files(directory: Path) -> tuple[Path, ...]:
    return tuple(path for path in directory.iterdir() if path.is_file())


def _portable_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    root = project_root.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise BusiCurationError(
            f"{resolved} está fuera del project_root {root}; "
            "no se puede escribir una ruta publicable."
        ) from exc


def _mask_sort_key(path: Path, image_stem: str) -> tuple[int, int, str]:
    suffix = path.stem[len(image_stem) :]
    if suffix == "_mask":
        return (0, 0, path.name.casefold())
    match = re.fullmatch(r"_mask_(\d+)", suffix, re.IGNORECASE)
    if match is None:
        return (2, 0, path.name.casefold())
    return (1, int(match.group(1)), path.name.casefold())


@lru_cache(maxsize=None)
def resolve_file_metadata(
    member: BusiMember,
    *,
    raw_root: str | Path,
    project_root: str | Path,
) -> FileMetadata:
    """Resolve one raw image and every existing BUSI mask, then hash them."""

    root = Path(raw_root)
    class_dir = root / member.label
    if not class_dir.is_dir():
        raise FileNotFoundError(f"No existe la carpeta de clase BUSI: {class_dir}")

    image_candidates = sorted(
        path
        for path in _directory_files(class_dir)
        if path.suffix.lower() in IMAGE_EXTENSIONS
        and path.stem.casefold() == member.filename.casefold()
    )
    if len(image_candidates) != 1:
        raise BusiCurationError(
            f"{member.filename!r}: se esperaba una imagen cruda, "
            f"se encontraron {len(image_candidates)} en {class_dir}."
        )
    image_path = image_candidates[0]

    mask_name_re = re.compile(
        rf"^{re.escape(image_path.stem)}_mask(?:_\d+)?"
        rf"(?:{'|'.join(re.escape(ext) for ext in sorted(IMAGE_EXTENSIONS))})$",
        re.IGNORECASE,
    )
    mask_paths = sorted(
        (path for path in _directory_files(class_dir) if mask_name_re.fullmatch(path.name)),
        key=lambda path: _mask_sort_key(path, image_path.stem),
    )
    if not mask_paths:
        raise BusiCurationError(f"No se encontró máscara para {image_path}.")

    project = Path(project_root)
    portable_masks = tuple(_portable_path(path, project) for path in mask_paths)
    return FileMetadata(
        image_path=_portable_path(image_path, project),
        mask_path=portable_masks[0],
        mask_paths=portable_masks,
        image_sha256=sha256_file(image_path),
        mask_sha256s=tuple(sha256_file(path) for path in mask_paths),
    )


def validate_curated_file_inventory(
    entries: Sequence[CuratedMappingEntry],
    *,
    curated_root: str | Path,
) -> dict[str, int]:
    """Require an exact one-image/one-mask copy for every official mapping row."""

    root = Path(curated_root)
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.is_dir() or not masks_dir.is_dir():
        raise FileNotFoundError(f"Curated BUSI debe contener images/ y masks/ bajo {root}.")

    expected_images = {f"{entry.image_key}.png" for entry in entries}
    expected_masks = {f"{entry.image_key}_mask.png" for entry in entries}
    actual_images = {path.name for path in _directory_files(images_dir)}
    actual_masks = {path.name for path in _directory_files(masks_dir)}

    missing_images = sorted(expected_images - actual_images)
    extra_images = sorted(actual_images - expected_images)
    missing_masks = sorted(expected_masks - actual_masks)
    extra_masks = sorted(actual_masks - expected_masks)
    problems: list[str] = []
    if missing_images:
        problems.append(f"imágenes faltantes={missing_images[:10]}")
    if extra_images:
        problems.append(f"imágenes no mapeadas={extra_images[:10]}")
    if missing_masks:
        problems.append(f"máscaras faltantes={missing_masks[:10]}")
    if extra_masks:
        problems.append(f"máscaras no mapeadas={extra_masks[:10]}")
    if problems:
        raise BusiCurationError(
            "La copia local Curated BUSI no coincide con el mapping oficial: " + "; ".join(problems)
        )
    return {"images": len(actual_images), "masks": len(actual_masks)}


@lru_cache(maxsize=None)
def resolve_curated_file_metadata(
    entry: CuratedMappingEntry,
    *,
    curated_root: str | Path,
    project_root: str | Path,
) -> FileMetadata:
    """Resolve and hash one image/mask pair from the official curated copy."""

    root = Path(curated_root)
    image_path = root / "images" / f"{entry.image_key}.png"
    mask_path = root / "masks" / f"{entry.image_key}_mask.png"
    if not image_path.is_file():
        raise FileNotFoundError(f"No existe imagen Curated BUSI: {image_path}")
    if not mask_path.is_file():
        raise FileNotFoundError(f"No existe máscara Curated BUSI: {mask_path}")

    project = Path(project_root)
    portable_mask = _portable_path(mask_path, project)
    return FileMetadata(
        image_path=_portable_path(image_path, project),
        mask_path=portable_mask,
        mask_paths=(portable_mask,),
        image_sha256=sha256_file(image_path),
        mask_sha256s=(sha256_file(mask_path),),
    )


def _join(values: Iterable[object], separator: str = "|") -> str:
    return separator.join(str(value) for value in values)


def _file_columns(metadata: FileMetadata) -> dict[str, object]:
    return {
        "image_path": metadata.image_path,
        "mask_path": metadata.mask_path,
        "mask_paths": _join(metadata.mask_paths),
        "n_masks": len(metadata.mask_paths),
        "image_sha256": metadata.image_sha256,
        "mask_sha256s": _join(metadata.mask_sha256s),
    }


def _pawlowska_common_row(
    decision: BusiDecision,
    metadata: FileMetadata,
    *,
    source_metadata_path: str,
    source_metadata_sha256: str,
) -> dict[str, object]:
    group = decision.group
    return {
        "curation_variant": "pawlowska_2023_sensitivity",
        "group_id": group.group_id,
        "image_id": decision.member.image_id,
        "class_id": decision.member.class_number,
        "ids": _join(group.ids),
        "filenames": _join(group.filenames),
        "filename": decision.member.filename,
        "label": decision.member.label,
        "label_idx": LABEL_INDEX.get(decision.member.label, ""),
        "objection": group.objection,
        "annotation": _join(group.annotation_tags, separator="&"),
        "annotation_tags": _join(group.annotation_tags),
        **_file_columns(metadata),
        "classifier_image_policy": "original_busi_pixels_resize_224_at_runtime",
        "classifier_mask_usage": "not_classifier_input",
        "verification_image_path": "",
        "verification_mask_path": "",
        "verification_image_sha256": "",
        "verification_mask_sha256": "",
        "source_doi": PAWLOWSKA_SOURCE_DOI,
        "dataset_source_doi": DATASET_SOURCE_DOI,
        "source_commit": "",
        "source_metadata_path": source_metadata_path,
        "source_metadata_sha256": source_metadata_sha256,
    }


def build_pawlowska_curation_rows(
    *,
    comment_list: str | Path,
    raw_root: str | Path,
    project_root: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build the 457-image Pawłowska sensitivity manifest and 780-row audit."""

    comment_path = Path(comment_list)
    project = Path(project_root)
    groups = load_comment_groups(comment_path)
    source_counts = validate_source_coverage(groups)
    decisions = decide_binary_curation(groups)
    curated_counts = validate_curated_selection(decisions)

    portable_comment_path = _portable_path(comment_path, project)
    comment_sha256 = sha256_file(comment_path)
    metadata_by_image_id: dict[int, FileMetadata] = {}
    for decision in decisions:
        metadata_by_image_id[decision.member.image_id] = resolve_file_metadata(
            decision.member,
            raw_root=raw_root,
            project_root=project,
        )

    audit_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for decision in decisions:
        common = _pawlowska_common_row(
            decision,
            metadata_by_image_id[decision.member.image_id],
            source_metadata_path=portable_comment_path,
            source_metadata_sha256=comment_sha256,
        )
        audit_rows.append(
            {
                **common,
                "source_row": decision.group.source_row,
                "member_position": decision.member_position,
                "group_eligible": str(decision.group_eligible).lower(),
                "selected_image_id": decision.selected_image_id or "",
                "decision": decision.decision,
                "decision_reason": _join(decision.decision_reasons),
            }
        )

        if decision.decision == "keep":
            manifest_rows.append(
                {
                    **common,
                    "manifest_id": f"BUSI-PAWLOWSKA-{decision.member.image_id:04d}",
                    "dataset": "busi",
                    "patient_id": "",
                }
            )

    decision_counts = Counter(row["decision_reason"] for row in audit_rows)
    summary: dict[str, object] = {
        "source_groups": len(groups),
        "source_images": len(decisions),
        "source_class_counts": source_counts,
        "curated_images": len(manifest_rows),
        "curated_class_counts": curated_counts,
        "decision_reason_counts": dict(sorted(decision_counts.items())),
        "source_metadata_path": portable_comment_path,
        "source_metadata_sha256": comment_sha256,
        "curation_source_doi": PAWLOWSKA_SOURCE_DOI,
        "dataset_source_doi": DATASET_SOURCE_DOI,
    }
    return manifest_rows, audit_rows, summary


def build_curation_rows(
    *,
    comment_list: str | Path,
    raw_root: str | Path,
    project_root: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Backward-compatible name for the Pawłowska sensitivity builder."""

    return build_pawlowska_curation_rows(
        comment_list=comment_list,
        raw_root=raw_root,
        project_root=project_root,
    )


def _official_common_row(
    entry: CuratedMappingEntry,
    original_metadata: FileMetadata,
    verification_metadata: FileMetadata,
    *,
    source_metadata_path: str,
    source_metadata_sha256: str,
) -> dict[str, object]:
    return {
        "curation_variant": "curated_busi_v1_official",
        "group_id": entry.group_id,
        "image_id": entry.original_global_id,
        "class_id": entry.class_number,
        "ids": entry.class_number,
        "filenames": entry.image_key,
        "filename": entry.original_filename,
        "label": entry.label,
        "label_idx": LABEL_INDEX.get(entry.label, ""),
        "objection": "",
        "annotation": "",
        "annotation_tags": "",
        **_file_columns(original_metadata),
        "classifier_image_policy": "official_ids_original_busi_pixels_resize_224_at_runtime",
        "classifier_mask_usage": "not_classifier_input",
        "verification_image_path": verification_metadata.image_path,
        "verification_mask_path": verification_metadata.mask_path,
        "verification_image_sha256": verification_metadata.image_sha256,
        "verification_mask_sha256": verification_metadata.mask_sha256s[0],
        "source_doi": CURATED_BUSI_SOURCE_DOI,
        "dataset_source_doi": DATASET_SOURCE_DOI,
        "source_commit": CURATED_BUSI_MAPPING_COMMIT,
        "source_metadata_path": source_metadata_path,
        "source_metadata_sha256": source_metadata_sha256,
    }


def build_official_curation_rows(
    *,
    mapping: str | Path,
    curated_root: str | Path,
    original_raw_root: str | Path,
    project_root: str | Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, object]]:
    """Build the primary 386-image binary manifest and 450-row official audit."""

    mapping_path = Path(mapping)
    project = Path(project_root)
    entries = load_curated_mapping(mapping_path)
    source_counts = validate_curated_mapping(entries, mapping_path=mapping_path)
    inventory_counts = validate_curated_file_inventory(
        entries,
        curated_root=curated_root,
    )
    portable_mapping_path = _portable_path(mapping_path, project)
    mapping_sha256 = sha256_file(mapping_path)

    original_metadata_by_key: dict[tuple[str, int], FileMetadata] = {}
    verification_metadata_by_key: dict[tuple[str, int], FileMetadata] = {}
    for entry in entries:
        original_metadata_by_key[(entry.label, entry.class_number)] = resolve_file_metadata(
            BusiMember(
                image_id=entry.original_global_id,
                filename=entry.original_filename,
                label=entry.label,
                class_number=entry.class_number,
            ),
            raw_root=original_raw_root,
            project_root=project,
        )
        verification_metadata_by_key[(entry.label, entry.class_number)] = (
            resolve_curated_file_metadata(
                entry,
                curated_root=curated_root,
                project_root=project,
            )
        )

    audit_rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    for entry in entries:
        common = _official_common_row(
            entry,
            original_metadata_by_key[(entry.label, entry.class_number)],
            verification_metadata_by_key[(entry.label, entry.class_number)],
            source_metadata_path=portable_mapping_path,
            source_metadata_sha256=mapping_sha256,
        )
        keep = entry.label in LABEL_INDEX
        audit_rows.append(
            {
                **common,
                "source_row": entry.source_row,
                "member_position": 1,
                "group_eligible": str(keep).lower(),
                "selected_image_id": entry.original_global_id if keep else "",
                "decision": "keep" if keep else "exclude",
                "decision_reason": ("official_curated_binary" if keep else "normal_class"),
            }
        )
        if keep:
            manifest_rows.append(
                {
                    **common,
                    "manifest_id": (f"CURATED-BUSI-{entry.label.upper()}-{entry.class_number:04d}"),
                    "dataset": "busi",
                    "patient_id": "",
                }
            )

    curated_counts = Counter(row["label"] for row in manifest_rows)
    if len(manifest_rows) != sum(EXPECTED_OFFICIAL_BINARY_CLASS_COUNTS.values()):
        raise BusiCurationError(
            f"Manifest oficial produjo {len(manifest_rows)} filas; se esperaban 386."
        )
    if dict(curated_counts) != dict(EXPECTED_OFFICIAL_BINARY_CLASS_COUNTS):
        raise BusiCurationError(
            f"Composición binaria oficial {dict(curated_counts)}; "
            f"esperada {dict(EXPECTED_OFFICIAL_BINARY_CLASS_COUNTS)}."
        )
    if len(audit_rows) != EXPECTED_OFFICIAL_TOTAL:
        raise BusiCurationError(
            f"Auditoría oficial produjo {len(audit_rows)} filas; se esperaban 450."
        )

    summary: dict[str, object] = {
        "source_images": len(entries),
        "source_class_counts": source_counts,
        "file_inventory": inventory_counts,
        "curated_images": len(manifest_rows),
        "curated_class_counts": dict(curated_counts),
        "decision_reason_counts": dict(
            sorted(Counter(row["decision_reason"] for row in audit_rows).items())
        ),
        "source_metadata_path": portable_mapping_path,
        "source_metadata_sha256": mapping_sha256,
        "source_commit": CURATED_BUSI_MAPPING_COMMIT,
        "curation_source_doi": CURATED_BUSI_SOURCE_DOI,
        "dataset_source_doi": DATASET_SOURCE_DOI,
        "classifier_image_policy": ("official_ids_original_busi_pixels_resize_224_at_runtime"),
        "classifier_mask_usage": "not_classifier_input",
    }
    return manifest_rows, audit_rows, summary


def write_csv(
    path: str | Path,
    rows: Sequence[Mapping[str, object]],
    *,
    fieldnames: Sequence[str],
) -> Path:
    """Write a deterministic UTF-8 CSV and reject undeclared output columns."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fieldnames,
                extrasaction="raise",
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return output_path


def curate_busi(
    *,
    mapping: str | Path,
    curated_root: str | Path,
    comment_list: str | Path,
    original_raw_root: str | Path,
    project_root: str | Path,
    primary_manifest_out: str | Path,
    primary_audit_out: str | Path,
    sensitivity_manifest_out: str | Path,
    sensitivity_audit_out: str | Path,
    primary_publication_manifest_out: str | Path | None = None,
) -> dict[str, object]:
    """Create the official primary cohort and Pawłowska sensitivity artifacts."""

    primary_manifest, primary_audit, primary_summary = build_official_curation_rows(
        mapping=mapping,
        curated_root=curated_root,
        original_raw_root=original_raw_root,
        project_root=project_root,
    )
    sensitivity_manifest, sensitivity_audit, sensitivity_summary = build_pawlowska_curation_rows(
        comment_list=comment_list,
        raw_root=original_raw_root,
        project_root=project_root,
    )

    primary_manifest_path = write_csv(
        primary_manifest_out,
        primary_manifest,
        fieldnames=MANIFEST_COLUMNS,
    )
    primary_audit_path = write_csv(
        primary_audit_out,
        primary_audit,
        fieldnames=AUDIT_COLUMNS,
    )
    sensitivity_manifest_path = write_csv(
        sensitivity_manifest_out,
        sensitivity_manifest,
        fieldnames=MANIFEST_COLUMNS,
    )
    sensitivity_audit_path = write_csv(
        sensitivity_audit_out,
        sensitivity_audit,
        fieldnames=AUDIT_COLUMNS,
    )
    publication_manifest_path = None
    if primary_publication_manifest_out is not None:
        publication_manifest_path = write_csv(
            primary_publication_manifest_out,
            primary_manifest,
            fieldnames=MANIFEST_COLUMNS,
        )

    project = Path(project_root)
    return {
        "primary": {
            **primary_summary,
            "manifest_path": _portable_path(primary_manifest_path, project),
            "audit_path": _portable_path(primary_audit_path, project),
            "publication_manifest_path": (
                _portable_path(publication_manifest_path, project)
                if publication_manifest_path is not None
                else ""
            ),
        },
        "sensitivity": {
            **sensitivity_summary,
            "manifest_path": _portable_path(sensitivity_manifest_path, project),
            "audit_path": _portable_path(sensitivity_audit_path, project),
        },
    }
