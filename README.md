# Generalización entre poblaciones de modelos de deep learning para clasificación de cáncer de mama en ecografía

Adaptación de dominio y explicabilidad (XAI).

Este repositorio contiene el código, la documentación y el protocolo experimental de un estudio
de **generalización cruzada (cross-domain)**: se entrenan modelos de clasificación de lesiones
mamarias en ecografía sobre una población (**BUSI**, Egipto) y se evalúan sobre otra población
(**BUS-BRA**, Brasil), cuantificando la caída de desempeño por *domain shift* y midiendo en qué
medida la **adaptación de dominio** la mitiga. La interpretabilidad (**Grad-CAM** y **SHAP**) se
usa para comprobar si los modelos atienden a regiones clínicamente plausibles.

> **Estado del repositorio:** código, configuración, documentación técnica y pruebas de humo para
> reproducir el pipeline experimental. El manuscrito/artículo científico no se versiona aquí; se
> enlazará como recurso externo cuando esté disponible.

---

## 1. Pregunta de investigación

1. **¿Cuánto cae el desempeño** (AUC, sensibilidad, especificidad) de un clasificador
   benigno/maligno cuando se entrena en BUSI y se evalúa en BUS-BRA (y viceversa), frente al
   desempeño intra-dominio?
2. **¿Qué técnicas de adaptación de dominio** (fine-tuning, DANN/adversarial, self-training)
   recuperan parte de esa caída, y cuánto?
3. **¿Las explicaciones (Grad-CAM/SHAP) son consistentes** entre poblaciones, o el modelo se
   apoya en artefactos de adquisición (texto, marcas, sombras) en lugar de la lesión?

## 2. Datasets

| Dataset | Población | Pacientes / Imágenes | Clases | Acceso |
|---|---|---|---|---|
| **BUSI** | Egipto (El Cairo, Baheya Hospital) | 600 / 780 | benigno (437), maligno (210), normal (133) | Público (Kaggle / Data in Brief) |
| **BUS-BRA** | Brasil (Río de Janeiro, INCA) | 1064 / 1875 | benigno (722), maligno (342) + BI-RADS 2–5 | Público (Zenodo / Kaggle) |
| **UDIAT** *(opcional, validación externa)* | España (Sabadell) | — / 163 | benigno (110), maligno (53) | Bajo solicitud a los autores |

> ⚠️ **BUSI tiene problemas documentados de calidad** (≈235 imágenes duplicadas, ~70 imágenes de
> axila y algunas con aguja de biopsia). Si no se deduplica, se produce **fuga de datos** que
> infla las métricas. El pipeline incluye un paso de deduplicación obligatorio. Ver
> [`docs/02_metodologia.md`](docs/02_metodologia.md) §Datos.

Instrucciones de descarga: [`data/README.md`](data/README.md).

## 3. Estructura del proyecto

```
.
├── README.md                  <- Este archivo
├── requirements.txt           <- Dependencias pip
├── environment.yml            <- Entorno conda (alternativa)
├── pyproject.toml             <- Instalación del paquete `src` y configuración de herramientas
├── config/
│   └── config.yaml            <- ÚNICA fuente de verdad para hiperparámetros y rutas
├── data/
│   ├── README.md              <- Cómo obtener BUSI / BUS-BRA
│   ├── raw/                   <- Datos originales (NO se versionan)
│   └── processed/             <- Datos preprocesados + manifests CSV (NO se versionan)
├── docs/
│   ├── 01_revision_literatura.md
│   ├── 02_metodologia.md
│   └── referencias.bib
├── src/                       <- Paquete Python instalable
│   ├── config.py              <- Carga/validación del config.yaml
│   ├── data/                  <- Descarga, datasets, preprocesamiento
│   ├── models/                <- Arquitecturas (ResNet-50, EfficientNet-B0)
│   ├── training/              <- Augmentation y bucle de entrenamiento
│   ├── evaluation/            <- Métricas y matriz de generalización cruzada
│   ├── explainability/        <- Grad-CAM y SHAP
│   └── utils/                 <- Semillas, logging, reporting
├── scripts/                   <- Puntos de entrada numerados (CLI)
│   ├── 01_download_data.py
│   ├── 02_preprocess.py
│   ├── 03_train_baseline.py
│   ├── 04_cross_domain_matrix.py
│   ├── 05_explainability.py
│   └── 06_generate_report.py
├── results/                   <- Modelos, figuras y reportes (salida)
└── tests/                     <- Pruebas de humo
```

## 4. Instalación

Requisitos: **Python 3.10+**, GPU NVIDIA con CUDA (recomendado), Git.

### Opción A — venv + pip

```bash
git clone <URL-de-tu-repo>.git
cd busgen

python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .          # instala el paquete `src` en modo editable
```

> **PyTorch + CUDA:** instala la build adecuada para tu GPU siguiendo
> https://pytorch.org/get-started/locally/ . Por ejemplo, para CUDA 12.1:
> `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`

### Opción B — conda

```bash
conda env create -f environment.yml
conda activate busgen
pip install -e .
```

## 5. Reproducir los resultados (flujo completo)

Todos los scripts leen `config/config.yaml`. Cada paso es independiente y deja artefactos en disco.

```bash
# 1) Descargar BUSI y BUS-BRA a data/raw/
python scripts/01_download_data.py --config config/config.yaml

# 2) Preprocesar: deduplicar, redimensionar, unificar etiquetas, generar manifests CSV
python scripts/02_preprocess.py --config config/config.yaml

# 3) Entrenar baselines intra-dominio (validación cruzada por paciente)
python scripts/03_train_baseline.py --config config/config.yaml --dataset busi    --arch resnet50
python scripts/03_train_baseline.py --config config/config.yaml --dataset bus_bra --arch resnet50

# 4) Matriz de generalización cruzada (entrenar en X, evaluar en Y) + adaptación de dominio
python scripts/04_cross_domain_matrix.py --config config/config.yaml --arch resnet50

# 5) Explicabilidad (Grad-CAM y SHAP sobre un conjunto de casos)
python scripts/05_explainability.py --config config/config.yaml --checkpoint results/models/<...>.pt

# 6) Reporte final (tablas + figuras agregadas)
python scripts/06_generate_report.py --config config/config.yaml
```

**Reproducibilidad:** todas las semillas se fijan desde `config.yaml` (`seed: 42`). Las
particiones de validación cruzada se agrupan **por paciente** para evitar fuga de datos. Cada
ejecución guarda el `config.yaml` efectivo y el hash de git en `results/reports/`.

## 6. Métricas

Clasificación binaria benigno/maligno. Se reportan: **accuracy, sensibilidad (recall maligno),
especificidad, AUC-ROC, F1, balanced accuracy**, con **intervalos de confianza por bootstrap**.
La métrica principal para comparar dominios es el **AUC** y la **caída relativa de AUC**
(intra-dominio − cross-domain).

## 7. Artículo y citación

El artículo científico asociado se enlazará externamente para evitar versionar borradores del
manuscrito dentro del repositorio.

Si usas este código, cita los datasets originales (BUSI, BUS-BRA) y este repositorio. Plantilla
en [`docs/referencias.bib`](docs/referencias.bib).

## 8. Licencia y ética

Datos de uso público con fines de investigación; respeta las licencias de cada dataset
(BUSI: CC BY 4.0; BUS-BRA: licencia de PEB/COPPE-UFRJ). Este trabajo es metodológico y **no es un
dispositivo clínico**.
