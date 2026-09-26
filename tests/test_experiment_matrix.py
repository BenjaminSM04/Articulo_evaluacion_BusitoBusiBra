"""The v3 job matrix and pilot must be a single, reproducible registry."""

from __future__ import annotations

from collections import Counter

from src.training.experiment_matrix import build_experiment_matrix, pilot_experiments


def test_v3_registry_has_exact_family_counts_and_unique_ids() -> None:
    jobs = build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"),
        seeds=(17, 42, 73, 101, 202),
    )
    assert len(jobs) == 160
    assert len({job.experiment_id for job in jobs}) == len(jobs)
    assert Counter(job.family for job in jobs) == {
        "main": 80,
        "intensity": 20,
        "roi": 20,
        "adabn": 10,
        "sensitivity": 30,
    }
    assert sum(job.trainable for job in jobs) == 150
    assert {job.seed for job in jobs} == {17, 42, 73, 101, 202}
    assert all(job.arch == "resnet18" for job in jobs if job.family == "sensitivity")


def test_v3_registry_lineage_and_hyperparameters() -> None:
    jobs = build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"), seeds=(17,)
    )
    by_id = {job.experiment_id: job for job in jobs}
    for job in jobs:
        if job.parent_id:
            assert job.parent_id in by_id
            parent = by_id[job.parent_id]
            assert (parent.arch, parent.seed) == (job.arch, job.seed)
    assert Counter((job.method, job.hyperparameter, job.value) for job in jobs
                   if job.family == "sensitivity") == {
        ("dann", "dann_lambda", 0.5): 1,
        ("dann", "dann_lambda", 2.0): 1,
        ("coral", "align_weight", 0.5): 1,
        ("coral", "align_weight", 2.0): 1,
        ("mmd", "mmd_sigma_scale", 0.5): 1,
        ("mmd", "mmd_sigma_scale", 2.0): 1,
    }
    assert all(job.parent_id and job.preprocessing == "none" for job in jobs
               if job.family == "adabn")


def test_pilot_is_reusable_dependency_closed_subset() -> None:
    all_jobs = build_experiment_matrix(
        architectures=("resnet18", "efficientnet_b0"),
        seeds=(17, 42, 73, 101, 202),
    )
    pilot = pilot_experiments(all_jobs)
    assert len(pilot) == 20  # 19 requested jobs plus the ResNet-18 source parent.
    assert Counter(job.arch for job in pilot) == {"efficientnet_b0": 13, "resnet18": 7}
    assert {job.seed for job in pilot} == {17}
    pilot_ids = {job.experiment_id for job in pilot}
    assert all(not job.parent_id or job.parent_id in pilot_ids for job in pilot)
    assert all(job in all_jobs for job in pilot)
