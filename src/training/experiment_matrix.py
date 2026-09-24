"""Canonical, dependency-ordered registry of v3 training and control jobs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    family: str
    arch: str
    seed: int
    method: str
    preprocessing: str = "none"
    hyperparameter: str = ""
    value: float | None = None
    parent_id: str = ""
    trainable: bool = True

    @property
    def stage(self) -> str:
        if self.family == "main":
            return "main"
        if self.family == "sensitivity":
            return "sensitivity"
        return "controls"


MAIN_METHODS = (
    "source_direct",
    "source_only_matched",
    "dann",
    "coral",
    "mmd",
    "finetune_5pct",
    "finetune_10pct",
    "finetune_20pct",
)
CONTROL_PREPROCESSING = ("intensity", "roi")
SENSITIVITY_VALUES = (
    ("dann", "dann_lambda", 0.5),
    ("dann", "dann_lambda", 2.0),
    ("coral", "align_weight", 0.5),
    ("coral", "align_weight", 2.0),
    ("mmd", "mmd_sigma_scale", 0.5),
    ("mmd", "mmd_sigma_scale", 2.0),
)


def _id(family: str, arch: str, seed: int, method: str = "") -> str:
    stem = f"{family}_{arch}_seed{seed}"
    return f"{stem}_{method}" if method else stem


def build_experiment_matrix(
    *, architectures: Sequence[str], seeds: Sequence[int]
) -> tuple[Experiment, ...]:
    """Construct every prespecified v3 job, with parent jobs placed first."""

    if set(architectures) != {"resnet18", "efficientnet_b0"} or len(architectures) != 2:
        raise ValueError("v3 requiere exactamente ResNet-18 y EfficientNet-B0")
    if len(seeds) != 5 and len(seeds) != 1:
        raise ValueError("v3 requiere cinco semillas, o una para pruebas de la matriz")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Las semillas de v3 deben ser unicas")

    jobs: list[Experiment] = []
    for arch in architectures:
        for seed_value in seeds:
            seed = int(seed_value)
            main_source = _id("main", arch, seed, "source_direct")
            main_matched = _id("main", arch, seed, "source_only_matched")
            for method in MAIN_METHODS:
                jobs.append(
                    Experiment(
                        experiment_id=_id("main", arch, seed, method),
                        family="main",
                        arch=arch,
                        seed=seed,
                        method=method,
                        hyperparameter=(
                            "budget_fraction" if method.startswith("finetune_") else ""
                        ),
                        value=(
                            float(method.removeprefix("finetune_").removesuffix("pct")) / 100
                            if method.startswith("finetune_")
                            else None
                        ),
                        parent_id="" if method == "source_direct" else main_source,
                    )
                )
            for family in CONTROL_PREPROCESSING:
                control_source = _id(family, arch, seed, "source_direct")
                jobs.extend(
                    (
                        Experiment(
                            experiment_id=control_source,
                            family=family,
                            arch=arch,
                            seed=seed,
                            method="source_direct",
                            preprocessing=family,
                        ),
                        Experiment(
                            experiment_id=_id(family, arch, seed, "source_only_matched"),
                            family=family,
                            arch=arch,
                            seed=seed,
                            method="source_only_matched",
                            preprocessing=family,
                            parent_id=control_source,
                        ),
                    )
                )
            jobs.append(
                Experiment(
                    experiment_id=_id("adabn", arch, seed),
                    family="adabn",
                    arch=arch,
                    seed=seed,
                    method="adabn",
                    parent_id=main_matched,
                    trainable=False,
                )
            )
            if arch == "resnet18":
                for method, hyperparameter, value in SENSITIVITY_VALUES:
                    jobs.append(
                        Experiment(
                            experiment_id=_id(
                                "sensitivity", arch, seed, f"{method}_{value:g}"
                            ),
                            family="sensitivity",
                            arch=arch,
                            seed=seed,
                            method=method,
                            hyperparameter=hyperparameter,
                            value=value,
                            parent_id=main_source,
                        )
                    )
    ids = [job.experiment_id for job in jobs]
    if len(ids) != len(set(ids)):
        raise RuntimeError("La matriz v3 contiene identificadores duplicados")
    return tuple(jobs)


def pilot_experiments(jobs: Sequence[Experiment]) -> tuple[Experiment, ...]:
    """The development pilot plus the ResNet source parent it requires."""

    selected = tuple(
        job
        for job in jobs
        if job.seed == 17
        and (
            job.arch == "efficientnet_b0"
            or job.family == "sensitivity"
            or (job.arch == "resnet18" and job.family == "main" and job.method == "source_direct")
        )
    )
    selected_ids = {job.experiment_id for job in selected}
    missing_parent = any(
        job.parent_id and job.parent_id not in selected_ids for job in selected
    )
    if len(selected) != 20 or missing_parent:
        raise RuntimeError("El piloto v3 no contiene su matriz exacta y padres")
    return selected
