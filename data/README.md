# Datos

Este directorio aloja los datasets. **Su contenido no se versiona** (ver `.gitignore`); solo se
versiona este README. Tras la descarga la estructura esperada es:

```
data/
├── raw/
│   ├── BUSI/
│   │   ├── benign/      (benign (1).png, benign (1)_mask.png, ...)
│   │   ├── malignant/
│   │   └── normal/
│   └── BUS-BRA/
│       ├── Images/      (bus_0001-l.png, ...)
│       ├── Masks/
│       └── bus_data.csv (ID, Pathology, BIRADS, Device, ...)
└── processed/
    ├── busi/            (imágenes redimensionadas)
    ├── bus_bra/
    ├── busi_manifest.csv
    ├── bus_bra_manifest.csv
    └── dedup_report.csv (duplicados eliminados de BUSI)
```

## BUSI (Egipto)

- **Origen:** Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A. *Dataset of breast ultrasound images.*
  Data in Brief, 2020. Recolectado en Baheya Hospital (El Cairo).
- **Contenido:** 780 imágenes PNG (~500×500), 600 mujeres (25–75 años): 437 benignas, 210 malignas,
  133 normales, con máscaras de segmentación.
- **Licencia:** CC BY 4.0.
- **Descarga automática** (requiere la API de Kaggle configurada, ver más abajo):

  ```bash
  python scripts/01_download_data.py --config config/config.yaml --only busi
  ```

- **Descarga manual:** Kaggle → *"Breast Ultrasound Images Dataset"*
  (`aryashah2k/breast-ultrasound-images-dataset`). Descomprime en `data/raw/BUSI/`.

> ⚠️ **Calidad conocida:** BUSI contiene ~235 imágenes duplicadas, ~70 imágenes de axila (no mama)
> y algunas con aguja de biopsia. El preprocesamiento (`scripts/02_preprocess.py`) deduplica con
> *perceptual hashing* y registra lo eliminado en `data/processed/dedup_report.csv`. **No omitas
> este paso:** sin deduplicar, los duplicados se reparten entre train/test e inflan las métricas.

## BUS-BRA (Brasil)

- **Origen:** Gómez-Flores W., et al. *BUS-BRA: A breast ultrasound dataset for assessing
  computer-aided diagnosis systems.* Medical Physics, 2024. Recolectado en el Instituto Nacional
  de Câncer (INCA), Río de Janeiro, con 4 ecógrafos.
- **Contenido:** 1875 imágenes de 1064 pacientes: 722 benignas, 342 malignas; anotaciones BI-RADS
  (2–5) y delineaciones de tumor. Particiones de validación cruzada 5- y 10-fold incluidas.
- **Distribución/copyright:** PEB/COPPE-UFRJ. Fuentes: Zenodo (`zenodo.org/records/8231412`) y
  Kaggle (`orvile/bus-bra-a-breast-ultrasound-dataset`).
- **Descarga automática:**

  ```bash
  python scripts/01_download_data.py --config config/config.yaml --only bus_bra
  ```

## UDIAT (España) — validación externa opcional

163 imágenes (110 benignas, 53 malignas), UDIAT Diagnostic Centre, Sabadell (Yap et al., 2018).
**No es de descarga pública**; debe solicitarse a los autores. Si lo obtienes, colócalo en
`data/raw/UDIAT/` y añádelo a `config.yaml` como un tercer dataset.

## Configurar la API de Kaggle

1. Crea una cuenta en kaggle.com → *Account* → *Create New API Token* → descarga `kaggle.json`.
2. Colócalo en `~/.kaggle/kaggle.json` (Linux/Mac) o `%USERPROFILE%\.kaggle\kaggle.json` (Windows).
3. `chmod 600 ~/.kaggle/kaggle.json` (Linux/Mac).

## Ética y uso

Datos anonimizados de uso investigativo. Respeta las licencias de cada fuente. Este proyecto es
metodológico y no constituye un dispositivo de diagnóstico clínico.
