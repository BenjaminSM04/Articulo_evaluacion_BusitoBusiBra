# Código del reanálisis BUSI → BUS-BRA para OJEMB

Este repositorio contiene el **código y el protocolo del reanálisis v3**. No contiene el artículo, imágenes médicas, manifiestos de pacientes, predicciones, checkpoints ni resultados de cinco semillas. El autor aprobó el protocolo el 23 de septiembre de 2026; esa aprobación no sustituye una determinación ética escrita. El código y las pruebas no representan resultados experimentales.

`publication_v2` es un análisis histórico conservado fuera de este repositorio. La configuración `config/config_publication_v3_5seed.yaml` prepara una corrida separada con cinco semillas, pero **no representa resultados obtenidos**. Los checkpoints v2 no se combinarán con los futuros v3.

## Alcance científico

El protocolo parte de pesos ImageNet preentrenados, no de inicialización aleatoria. Compara ResNet-18 y EfficientNet-B0 entrenadas con 358 imágenes Curated BUSI como fuente y BUS-BRA como destino. La matriz preespecificada contiene 160 variantes: 80 principales, 20 de intensidad, 20 ROI, 10 AdaBN y 30 de sensibilidad. La partición fuente v3 agrupa imágenes relacionadas; no equivale a una separación por paciente. La partición BUS-BRA histórica se conserva.

El test BUS-BRA tuvo exposición exploratoria previa: los resultados futuros seguirán siendo descriptivos, no una validación externa independiente. La revisión BUSI y las reglas de controles están documentadas en [preparación v3](docs/PREPARACION_V3.md). `target_calibration` y `target_test` siguen bloqueados hasta recibir constancia ética institucional escrita. Una cohorte externa nueva continúa siendo necesaria para confirmar los hallazgos.

## Contenido público y datos privados

- `config/`: configuración histórica v2 y propuesta aislada v3 de cinco semillas.
- `src/` y `scripts/`: preparación, particiones, entrenamiento, evaluación y análisis. Son herramientas; incluirlas no autoriza ejecutarlas sobre datos clínicos.
- `resources/busi_curation/`: mapeos y auditoría de imágenes BUSI de origen público; [alcance y procedencia](resources/busi_curation/README.md).
- `tests/`: pruebas sintéticas sin entrenamiento experimental ni acceso al test clínico.
- `docs/PROTOCOL_REVIEW_V3_5SEED_LOCAL.md`: condiciones para la ejecución local v3, piloto y pausa previa a inferencia final. `docs/PROTOCOL_PUBLICATION_V2.md` documenta el protocolo histórico, no una instrucción para reproducir cifras v2 como v3.
- `docs/FINALIZATION_READINESS_V3.md`: estado del entrenamiento, base ética para datos públicos desidentificados, supuesto de cobertura APC y preparación de la evaluación final.

Los datos originales deben obtenerse y conservarse fuera de Git. El respaldo
privado conserva los datasets y la partición histórica BUS-BRA; un clon público
por sí solo **no permite reconstruir esa partición ni reproducir las cifras**.
Para preparar el entrenamiento v3, el utilitario del [protocolo local](docs/PROTOCOL_REVIEW_V3_5SEED_LOCAL.md)
copia al clon privado únicamente imágenes y máscaras de desarrollo,
`source_train`, `source_val` y `target_adapt`, después de comprobar huellas y
pertenencia. No copia ni abre imágenes de test o calibración. Nunca se debe
generar una partición BUS-BRA nueva y presentarla como la histórica.

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

## Antes del piloto y la corrida

El plan inicial preveía un piloto de 20 trabajos y un límite acumulado de 18 horas. El autor autorizó posteriormente ejecutar la matriz completa con `--full-no-pilot`, sin esos dos gates. Se mantienen las huellas, los controles de recursos y `--resume`. La inferencia final no forma parte de esa autorización: exige constancia ética escrita y una decisión separada. Las instrucciones y el alcance exacto están en el [protocolo local](docs/PROTOCOL_REVIEW_V3_5SEED_LOCAL.md).

Este software es de investigación y **no es un dispositivo ni una recomendación clínica**. La revisión editorial, ética y final del artículo se gestiona fuera de este repositorio.
