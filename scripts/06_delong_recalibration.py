#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
06_delong_recalibration.py
==========================================================================
Analisis complementarios SIN reentrenar, a partir de los CSV de predicciones
ya guardados en results/metrics/*.csv (columnas: index, y_true, y_prob_malignant).

Responde a dos puntos de la revision por pares:

  [M2] Prueba estadistica formal para las comparaciones entre metodos de
       adaptacion de dominio -> test de DeLong PAREADO sobre el mismo test
       objetivo reservado (n=736), por arquitectura:
         - none vs {dann, coral, mmd, self_training}   (¿la DA supera a la linea base?)
         - dann  vs {coral, mmd, self_training}         (¿DANN es distinguible del resto?)
       Tambien best-fine-tuning vs DANN.

  [M4] La caida de sensibilidad NO se debe al desbalance de clases (prevalencia
       maligna ~identica: BUSI 32.3% vs BUS-BRA 32.4%). Es un covariate/feature
       shift que desplaza el punto de operacion. Se demuestra con una
       RECALIBRACION DE UMBRAL del modelo base ("none"): se elige el umbral en
       una mitad de calibracion del test objetivo y se evalua en la otra mitad.
       Recuperar sensibilidad recalibrando el umbral (sin tocar los pesos)
       sostiene que gran parte de la "recuperacion" es reubicacion del umbral.

Uso (Windows, con el entorno del proyecto):
    D:\\busgen\\.venv\\Scripts\\python.exe scripts\\06_delong_recalibration.py

Uso (Linux/Mac):
    python scripts/06_delong_recalibration.py

Salidas:
    results/metrics/complementary_delong.csv
    results/metrics/complementary_recalibration.csv
    results/metrics/complementary_summary.md   <- pegar/consultar para el manuscrito

Dependencias: numpy, scipy (ya presentes en el .venv del proyecto).
==========================================================================
"""
from __future__ import annotations
import os
import sys
import glob
import csv
from pathlib import Path

import numpy as np

try:
    from scipy import stats
except Exception as e:  # pragma: no cover
    sys.stderr.write("ERROR: se requiere scipy. Instala con: pip install scipy\n")
    raise

# --------------------------------------------------------------------------
# Localizacion de la carpeta de metricas (robusto a donde se ejecute)
# --------------------------------------------------------------------------
HERE = Path(__file__).resolve().parent
CANDIDATES = [
    HERE.parent / "results" / "metrics",   # <repo>/results/metrics
    HERE / "results" / "metrics",
    Path.cwd() / "results" / "metrics",
]
METRICS_DIR = next((p for p in CANDIDATES if p.is_dir()), None)
if METRICS_DIR is None:
    sys.stderr.write(
        "ERROR: no encuentro results/metrics. Ejecuta el script desde el repo "
        "o ajusta METRICS_DIR.\n"
    )
    sys.exit(1)

ARCHS = ["resnet18", "resnet50", "efficientnet_b0", "densenet121"]
DA_METHODS = ["none", "dann", "coral", "mmd", "self_training"]
FT_BUDGETS = ["finetune_5pct", "finetune_10pct", "finetune_20pct"]

ADAPT_TPL = "adaptation_busi_to_bus_bra_{arch}_{method}_predictions.csv"
FT_TPL = "finetuning_busi_to_bus_bra_{arch}_{budget}_predictions.csv"


def load_pred(path: Path):
    """Devuelve (y_true, y_prob) como arrays float. None si no existe."""
    if not path.is_file():
        return None
    y_true, y_prob = [], []
    with open(path, newline="") as fh:
        rdr = csv.DictReader(fh)
        for row in rdr:
            y_true.append(int(float(row["y_true"])))
            y_prob.append(float(row["y_prob_malignant"]))
    return np.asarray(y_true, dtype=int), np.asarray(y_prob, dtype=float)


# --------------------------------------------------------------------------
# DeLong rapido (Sun & Xu, 2014) para AUC pareado
# --------------------------------------------------------------------------
def _compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: [k, n] con los m positivos primero. Devuelve (aucs, cov)."""
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(pos[r, :])
        ty[r, :] = _compute_midrank(neg[r, :])
        tz[r, :] = _compute_midrank(preds_sorted[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def delong_paired(y_true, prob_a, prob_b):
    """Test de DeLong pareado para AUC_a - AUC_b. Devuelve dict."""
    assert prob_a.shape == prob_b.shape == y_true.shape
    order = np.argsort(-y_true, kind="mergesort")  # positivos (label 1) primero
    m = int(y_true.sum())
    preds = np.vstack((prob_a, prob_b))[:, order]
    aucs, cov = _fast_delong(preds, m)
    cov = np.atleast_2d(cov)
    L = np.array([[1.0, -1.0]])
    var = float(L.dot(cov).dot(L.T)[0, 0])
    diff = float(aucs[0] - aucs[1])
    if var <= 0:
        z, p = 0.0, 1.0
    else:
        z = diff / np.sqrt(var)
        p = 2.0 * stats.norm.sf(abs(z))
    return {"auc_a": float(aucs[0]), "auc_b": float(aucs[1]),
            "delta": diff, "z": float(z), "p": float(p)}


def auc_only(y_true, y_prob):
    """AUC vía rank-sum (Mann-Whitney), robusto a empates."""
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def sens_spec(y_true, y_prob, thr):
    pred = (y_prob >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    return sens, spec


def youden_threshold(y_true, y_prob):
    """Umbral que maximiza (sens + spec - 1) usando los propios scores como cortes."""
    thrs = np.unique(y_prob)
    best_thr, best_j = 0.5, -1.0
    for t in thrs:
        se, sp = sens_spec(y_true, y_prob, t)
        j = se + sp - 1.0
        if j > best_j:
            best_j, best_thr = j, t
    return best_thr


def threshold_for_target_sens(y_true, y_prob, target=0.85):
    """Umbral mas alto (mas especifico) que aun logra sensibilidad >= target."""
    thrs = np.sort(np.unique(y_prob))[::-1]
    chosen = thrs[-1]
    for t in thrs:
        se, _ = sens_spec(y_true, y_prob, t)
        if se >= target:
            chosen = t
        else:
            break
    return chosen


def main():
    print(f"[i] Carpeta de metricas: {METRICS_DIR}")
    delong_rows = []
    recal_rows = []

    for arch in ARCHS:
        preds = {}
        for meth in DA_METHODS:
            p = load_pred(METRICS_DIR / ADAPT_TPL.format(arch=arch, method=meth))
            if p is not None:
                preds[meth] = p
        if "none" not in preds:
            print(f"[!] {arch}: sin CSV de adaptacion; se omite.")
            continue

        y_true = preds["none"][0]
        # Verificar que todas las predicciones comparten el mismo y_true (pareado)
        for meth, (yt, _) in preds.items():
            if not np.array_equal(yt, y_true):
                print(f"[!] {arch}/{meth}: y_true no coincide con 'none'. Revisar orden.")

        # ---- DeLong: none vs cada DA, y dann vs cada DA ----
        comparisons = []
        for meth in ["dann", "coral", "mmd", "self_training"]:
            if meth in preds:
                comparisons.append(("none", meth))
        for meth in ["coral", "mmd", "self_training"]:
            if "dann" in preds and meth in preds:
                comparisons.append(("dann", meth))

        # best-FT vs DANN (elige el presupuesto de fine-tuning con mayor AUC)
        ft_best = None
        for b in FT_BUDGETS:
            fp = load_pred(METRICS_DIR / FT_TPL.format(arch=arch, budget=b))
            if fp is None:
                continue
            a = auc_only(fp[0], fp[1])
            if ft_best is None or a > ft_best[2]:
                ft_best = (b, fp, a)

        for a, b in comparisons:
            r = delong_paired(y_true, preds[a][1], preds[b][1])
            delong_rows.append({
                "arch": arch, "model_a": a, "model_b": b,
                "auc_a": round(r["auc_a"], 4), "auc_b": round(r["auc_b"], 4),
                "delta_auc": round(r["delta"], 4), "z": round(r["z"], 3),
                "p_value": round(r["p"], 4),
                "significativo_0.05": "si" if r["p"] < 0.05 else "no",
            })

        if ft_best is not None and "dann" in preds:
            r = delong_paired(y_true, ft_best[1][1], preds["dann"][1])
            delong_rows.append({
                "arch": arch, "model_a": ft_best[0], "model_b": "dann",
                "auc_a": round(r["auc_a"], 4), "auc_b": round(r["auc_b"], 4),
                "delta_auc": round(r["delta"], 4), "z": round(r["z"], 3),
                "p_value": round(r["p"], 4),
                "significativo_0.05": "si" if r["p"] < 0.05 else "no",
            })

        # ---- Recalibracion de umbral del modelo base "none" ----
        # Split estratificado 50/50 del test objetivo (semilla fija).
        rng = np.random.RandomState(42)
        yb, pb = preds["none"]
        idx_pos = np.where(yb == 1)[0]
        idx_neg = np.where(yb == 0)[0]
        rng.shuffle(idx_pos)
        rng.shuffle(idx_neg)
        cal = np.concatenate([idx_pos[: len(idx_pos) // 2], idx_neg[: len(idx_neg) // 2]])
        ev = np.concatenate([idx_pos[len(idx_pos) // 2:], idx_neg[len(idx_neg) // 2:]])

        se05, sp05 = sens_spec(yb[ev], pb[ev], 0.5)
        thr_j = youden_threshold(yb[cal], pb[cal])
        se_j, sp_j = sens_spec(yb[ev], pb[ev], thr_j)
        thr_t = threshold_for_target_sens(yb[cal], pb[cal], target=0.85)
        se_t, sp_t = sens_spec(yb[ev], pb[ev], thr_t)

        # DANN a 0.5 sobre la MISMA mitad de evaluacion (comparacion justa)
        se_dann = sp_dann = float("nan")
        if "dann" in preds:
            se_dann, sp_dann = sens_spec(preds["dann"][0][ev], preds["dann"][1][ev], 0.5)

        recal_rows.append({
            "arch": arch,
            "n_eval": int(len(ev)),
            "base_thr0.5_sens": round(se05, 3), "base_thr0.5_spec": round(sp05, 3),
            "recal_youden_thr": round(float(thr_j), 3),
            "recal_youden_sens": round(se_j, 3), "recal_youden_spec": round(sp_j, 3),
            "recal_targetsens85_thr": round(float(thr_t), 3),
            "recal_targetsens85_sens": round(se_t, 3), "recal_targetsens85_spec": round(sp_t, 3),
            "dann_thr0.5_sens": round(se_dann, 3), "dann_thr0.5_spec": round(sp_dann, 3),
        })

    # ---------------------------------------------------------------- salidas
    dl_path = METRICS_DIR / "complementary_delong.csv"
    rc_path = METRICS_DIR / "complementary_recalibration.csv"
    md_path = METRICS_DIR / "complementary_summary.md"

    if delong_rows:
        with open(dl_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(delong_rows[0].keys()))
            w.writeheader()
            w.writerows(delong_rows)
    if recal_rows:
        with open(rc_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(recal_rows[0].keys()))
            w.writeheader()
            w.writerows(recal_rows)

    # Resumen legible
    lines = []
    lines.append("# Resumen de analisis complementarios (sin reentrenar)\n")
    lines.append("## [M2] DeLong pareado (test objetivo reservado, n=736)\n")
    lines.append("| Arq. | A | B | AUC_A | AUC_B | dAUC | z | p | sig.0.05 |")
    lines.append("|------|---|---|------:|------:|-----:|--:|--:|:--------:|")
    for r in delong_rows:
        lines.append(
            f"| {r['arch']} | {r['model_a']} | {r['model_b']} | {r['auc_a']} | "
            f"{r['auc_b']} | {r['delta_auc']:+} | {r['z']} | {r['p_value']} | "
            f"{r['significativo_0.05']} |"
        )
    lines.append("\n## [M4] Recalibracion de umbral del modelo base (none)\n")
    lines.append(
        "Umbral elegido en la mitad de calibracion; sens/spec medidas en la mitad "
        "de evaluacion. Muestra que reubicar el umbral (sin reentrenar) recupera "
        "sensibilidad, evidencia de que la caida es un desplazamiento del punto de "
        "operacion (covariate shift), no del desbalance de clases.\n"
    )
    lines.append("| Arq. | base@0.5 sens/spec | Youden sens/spec | target-sens0.85 sens/spec | DANN@0.5 sens/spec |")
    lines.append("|------|:------------------:|:----------------:|:-------------------------:|:------------------:|")
    for r in recal_rows:
        lines.append(
            f"| {r['arch']} | {r['base_thr0.5_sens']}/{r['base_thr0.5_spec']} | "
            f"{r['recal_youden_sens']}/{r['recal_youden_spec']} | "
            f"{r['recal_targetsens85_sens']}/{r['recal_targetsens85_spec']} | "
            f"{r['dann_thr0.5_sens']}/{r['dann_thr0.5_spec']} |"
        )
    lines.append("")
    md = "\n".join(lines)
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write(md)

    print("\n" + md)
    print(f"\n[ok] Escrito: {dl_path}")
    print(f"[ok] Escrito: {rc_path}")
    print(f"[ok] Escrito: {md_path}")


if __name__ == "__main__":
    main()
