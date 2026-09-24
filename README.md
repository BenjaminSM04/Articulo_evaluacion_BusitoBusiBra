# Código del reanálisis BUSI → BUS-BRA para OJEMB

Este repositorio contiene el **código y el protocolo de una revisión experimental pendiente**. No contiene el artículo, imágenes médicas, manifiestos de pacientes, predicciones, checkpoints ni resultados del reanálisis de cinco semillas. Por ahora **no se ejecutará entrenamiento ni inferencia**.

`publication_v2` es un análisis histórico conservado fuera de este repositorio. La configuración `config/config_publication_v3_5seed.yaml` prepara una corrida separada con cinco semillas, pero **no representa resultados obtenidos**. Los checkpoints v2 no se combinarán con los futuros v3.

## Alcance científico

El protocolo parte de pesos ImageNet preentrenados, no de inicialización aleatoria. Compara ResNet-18 y EfficientNet-B0 entrenadas con Curated BUSI como fuente y BUS-BRA como destino, con controles de tamaño equiparado, DANN, CORAL, MMD y tres presupuestos de ajuste fino. La partición fuente v3 agrupa imágenes BUSI relacionadas; no equivale a una separación por paciente. La partición BUS-BRA histórica se conserva.

El test BUS-BRA tuvo exposición exploratoria previa: los resultados futuros seguirán siendo descriptivos, no una validación externa independiente. La revisión visual de duplicados y artefactos y los controles AdaBN, normalización y ROI deben quedar resueltos y preespecificados antes de una corrida o de abrir de nuevo el test. Una cohorte externa nueva continúa siendo necesaria para confirmar los hallazgos.

## Contenido público y datos privados

- `config/`: configuración histórica v2 y propuesta aislada v3 de cinco semillas.
- `src/` y `scripts/`: preparación, particiones, entrenamiento, evaluación y análisis. Son herramientas; incluirlas no autoriza ejecutarlas sobre datos clínicos.
- `resources/busi_curation/`: mapeos y auditoría de imágenes BUSI de origen público; [alcance y procedencia](resources/busi_curation/README.md).
- `tests/`: pruebas sintéticas sin entrenamiento experimental ni acceso al test clínico.
- `docs/PROTOCOL_REVIEW_V3_5SEED_LOCAL.md`: condiciones para una ejecución **local futura**, actualmente aplazada. `docs/PROTOCOL_PUBLICATION_V2.md` documenta el protocolo histórico, no una instrucción para reproducir cifras v2 como v3.

Los datos originales deben obtenerse y conservarse fuera de Git. Para preparar una corrida futura se necesitarán, **en almacenamiento local privado**, las imágenes BUSI originales, la copia oficial Curated BUSI usada para verificar la selección, las imágenes BUS-BRA, los manifiestos fuente/destino y el archivo de partición histórica `results/metrics/bus_bra_adaptation_test_split.csv`. Este último es indispensable para conservar las cohortes BUS-BRA: un clon público, por sí solo, **no permite reconstruir la partición histórica ni reproducir las cifras**. Nunca se debe generar una partición nueva y presentarla como la histórica.

No añadas a Git `data/raw/`, `data/processed/`, `results/`, `kaggle.json`, `.env`, material editorial ni otros archivos con identificadores privados. Revisa los archivos preparados para cada commit; `.gitignore` no es una revisión de confidencialidad.

## Comprobación del código sin entrenamiento

Se requiere Python 3.11. Para probar el código en Windows con CPU, sin credenciales ni datasets:

```powershell
python --version  # debe indicar Python 3.11.x
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pytest -q tests
```

Para la ejecucion local con GPU de v3 se verificaron Python 3.11.9,
PyTorch 2.5.1+cu121 y torchvision 0.20.1+cu121 en Windows. Tras crear una
`.venv` nueva con ese Python, instala PyTorch desde el indice CUDA 12.1 y
luego las dependencias fijadas:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Si `python --version` indica otra versión, instala o selecciona explícitamente un ejecutable Python 3.11 para crear el entorno. En particular, el Python 3.13 del sistema no sirve para esta configuración fijada. El entorno embebido del respaldo local no forma parte del repositorio público.

La CI usa Python 3.11, PyTorch CPU, Ruff sobre el código de revisión y estas pruebas. No descarga datos ni llama a los scripts de entrenamiento o inferencia. Las pruebas crean únicamente fixtures temporales sintéticos; no producen checkpoints del estudio.

## Antes de cualquier experimento posterior

La revisión visual de las imágenes relacionadas y las decisiones sobre controles, particiones y análisis deben cerrarse primero. Después se comprobarán licencias, rutas y huellas SHA-256 de código y datos privados; se medirá un piloto local y se respetará un máximo de 18 horas acumuladas, con pausa si la proyección no cabe. Nada de esto forma parte de la entrega de código actual. Las instrucciones de seguridad y ejecución aplazada están en el [protocolo local](docs/PROTOCOL_REVIEW_V3_5SEED_LOCAL.md).

Este software es de investigación y **no es un dispositivo ni una recomendación clínica**. La revisión editorial, ética y final del artículo se gestiona fuera de este repositorio.
