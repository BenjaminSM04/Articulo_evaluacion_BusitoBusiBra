# QUICKSTART — verificación y primera corrida

Guía de comandos exactos. Ejecuta todo **desde la raíz del proyecto** (la carpeta que contiene
`src/`, `scripts/`, `config/`).

## 0. Instalar el entorno

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

Instala PyTorch con la build de CUDA de tu GPU (https://pytorch.org/get-started/locally/). Ejemplo
CUDA 12.1:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 1. Pruebas unitarias (rápidas, sin datos ni GPU)

```bash
pytest -q
```

Verifican: carga de config, *forward* de ResNet-50 y del modelo DANN, cálculo de métricas y que
las particiones de validación cruzada **no mezclan pacientes**. Deberían pasar en segundos
(usan `pretrained=false`, no descargan pesos).

## 2. Smoke test end-to-end con datos SINTÉTICOS (sin Kaggle, sin GPU)

Valida TODA la maquinaria (preprocesamiento → deduplicación → entrenamiento → matriz cross-domain
→ reporte) sobre imágenes falsas. Un solo comando:

```bash
# Windows:
scripts\run_smoke.bat
# Linux/Mac:
bash scripts/run_smoke.sh
```

Esto usa `config/config_smoke.yaml` (CPU, imágenes 96×96, 2 épocas). Al terminar, revisa:

- `results/reports/SUMMARY.md` — tablas de baselines y la matriz de generalización.
- `results/figures/` — curvas y el heatmap de la matriz.
- `data/processed/dedup_report.csv` — confirma que detectó los duplicados inyectados a propósito.

> Si el smoke test corre sin errores, el pipeline funciona en tu equipo. Las métricas serán bajas
> (datos sintéticos, sin preentrenamiento): es esperado.

## 3. Datos REALES: configurar Kaggle y descargar

> **Importante:** este paso necesita TUS credenciales de Kaggle. No puede hacerse de forma remota.

1. Crea cuenta en kaggle.com → *Settings* → *API* → **Create New API Token**. Se descarga
   `kaggle.json`.
2. Colócalo en:
   - Windows: `%USERPROFILE%\.kaggle\kaggle.json`
   - Linux/Mac: `~/.kaggle/kaggle.json` y luego `chmod 600 ~/.kaggle/kaggle.json`
3. Acepta los términos de cada dataset en su página de Kaggle (una vez, desde el navegador).
4. Descarga y preprocesa:

```bash
python scripts/01_download_data.py --config config/config.yaml
python scripts/02_preprocess.py    --config config/config.yaml
```

Alternativa sin Kaggle: descarga manual (URLs en [`../data/README.md`](../data/README.md)) y
descomprime en `data/raw/BUSI/` y `data/raw/BUS-BRA/`.

## 4. Experimentos reales (GPU)

```bash
# Baselines intra-dominio (CV 5-fold)
python scripts/03_train_baseline.py --config config/config.yaml --dataset busi    --arch resnet50
python scripts/03_train_baseline.py --config config/config.yaml --dataset bus_bra --arch resnet50

# Matriz de generalización cruzada (sin adaptación, luego con cada técnica)
python scripts/04_cross_domain_matrix.py --config config/config.yaml --arch resnet50 --adaptation none
python scripts/04_cross_domain_matrix.py --config config/config.yaml --arch resnet50 --adaptation finetune
python scripts/04_cross_domain_matrix.py --config config/config.yaml --arch resnet50 --adaptation dann

# Explicabilidad sobre un checkpoint (p.ej. modelo de BUSI evaluado en BUS-BRA)
python scripts/05_explainability.py --config config/config.yaml \
  --checkpoint results/models/baseline_busi_resnet50_fold0.pt --dataset bus_bra --method both

# Reporte consolidado
python scripts/06_generate_report.py --config config/config.yaml
```

Para usar fine-tuning real, fija en `config.yaml` → `domain_adaptation.finetune.n_target_labeled`
un valor > 0 (nº de imágenes etiquetadas del target por clase).

## Problemas frecuentes

- **`kaggle: command not found` / 401**: revisa `kaggle.json` y que aceptaste los términos del dataset.
- **CUDA no disponible**: pon `device: cpu` en el config (lento) o instala el PyTorch con CUDA correcto.
- **`to_markdown` falla**: falta `tabulate` (`pip install tabulate`); ya está en `requirements.txt`.
- **Columnas de BUS-BRA no reconocidas**: ajusta `build_bus_bra_manifest` a los nombres de tu CSV
  (la función ya intenta emparejar de forma flexible).
