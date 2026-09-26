# Revisión OJEMB v3: cinco semillas en equipo local

**Estado al 24 de septiembre de 2026:** protocolo aprobado por el autor; código,
particiones y controles preparados para validación previa al piloto. No se debe interpretar la existencia de
scripts o pruebas sintéticas como entrenamiento ejecutado. El manuscrito y los
resultados v2 permanecen fuera de este repositorio. La aprobación del autor no
equivale a una constancia ética institucional.

## Diseño científico congelado

- Configuración aislada: `config/config_publication_v3_5seed.yaml`, semillas
  `17, 42, 73, 101, 202` y salidas solo en `results/publication_v3_5seed/`.
  Los pesos iniciales son preentrenados en ImageNet, no aleatorios.
- Dos arquitecturas: ResNet-18 y EfficientNet-B0. La matriz tiene 160 variantes:
  80 principales (ocho brazos por arquitectura y semilla), 20 de normalización
  de intensidad, 20 ROI, 10 AdaBN y 30 de sensibilidad de hiperparámetros.
  Las 10 AdaBN son transformaciones de checkpoints, no entrenamiento con gradientes.
- La cohorte fuente primaria es la intersección auditada de 358 imágenes Curated
  BUSI binarias: 212 benignas y 146 malignas. Se excluyen exactamente 28
  (9 miembros duplicados, 14 axila y 5 aguja). Tres pares relacionados se
  agrupan siempre en un mismo split. La separación por grupo no demuestra
  separación por paciente. Las decisiones figuran en [PREPARACION_V3.md](PREPARACION_V3.md).
- La fuente se reparte 229/57/72 en train/val/test, semilla 20260723,
  proporciones 64/16/20 y disyunción estricta por grupo. Solo se regeneran
  asignaciones fuente. BUS-BRA conserva las cohortes y el split histórico por
  paciente; no se debe crear una nueva partición que pretenda sustituirlo.
- Los 48 checkpoints v2 son históricos y no se mezclarán con v3. Una diferencia
  v2–v3 no puede atribuirse solo al número de semillas porque cambió la fuente.
  El test BUS-BRA tuvo exposición exploratoria previa; el análisis v3 será
  descriptivo y exigirá una nueva cohorte externa para confirmación.

La familia primaria son las seis comparaciones DANN/CORAL/MMD contra
`source_only_matched`, con corrección Holm. AdaBN, intensidad y ROI constituyen
una familia secundaria separada; la sensibilidad de hiperparámetros es
descriptiva. Se reportarán NLL, Brier, ECE y umbrales 0,5, Youden de fuente y
Youden de calibración cuando esta última quede autorizada. Los intervalos
bootstrap por paciente y la dispersión entre semillas se comunicarán por
separado.

## Entradas privadas y procedencia

El repositorio público no contiene imágenes, máscaras, manifiestos de pacientes,
particiones históricas, predicciones, checkpoints ni resultados. El respaldo
`D:\busgen` no se modifica. La preparación local se hace mediante una lista
cerrada, en dos fases:

1. Copiar solo `data/processed/busi_curated_manifest.csv` para generar en el
   clon las asignaciones fuente v3. Su huella se comprueba contra el original
   histórico. Esta fase no sigue las rutas de imágenes dentro del CSV.
2. Copiar únicamente imágenes y máscaras mencionadas en `source_train`,
   `source_val` y `target_adapt`, más el manifiesto y las cinco asignaciones
   históricas de `target_adapt`. Las huellas fuente/objetivo y las dimensiones
   de cada imagen/máscara se comprueban antes de la copia; se conserva el aviso
   `LICENSE.txt` de BUS-BRA. No se copia el dataset completo, el manifiesto
   BUS-BRA general ni `bus_bra_adaptation_test_split.csv`.
3. Registrar SHA-256 de configuración, commit, código, CSV de auditoría,
   manifiestos, imágenes y máscaras de desarrollo, asignaciones y entorno;
   Python, PyTorch, CUDA, controlador y GPU. Cambios de bytes impiden
   `--resume` sin auditoría.

No se abren imágenes de `source_test`, `target_calibration` ni `target_test`.
La generación `--source-only` escribe una asignación fuente de test, pero no
lee sus píxeles. Las condiciones de uso y citas de BUSI/Curated BUSI y BUS-BRA
deben verificarse antes de publicar artefactos derivados; ningún archivo
clínico se publica en Git.

Ni los datos privados ni sus índices por paciente se subirán a Git o a nube.
`cross_validation.n_folds: 5` es heredado; las réplicas las controla
`publication.seeds`.

## Preparación local sin ejecutar modelos

Desde la raíz del clon, con Python 3.11 y las dependencias instaladas:

```powershell
$origenV3 = 'D:\busgen'
$clonV3 = 'D:\busgen-ojemb-code'
.\.venv\Scripts\python.exe scripts/15_stage_v3_private_development.py --prepare-manifest --source-root $origenV3 --clone-root $clonV3
.\.venv\Scripts\python.exe -m src.data.publication_splits --config config/config_publication_v3_5seed.yaml --source-only
.\.venv\Scripts\python.exe scripts/15_stage_v3_private_development.py --stage-development --source-root $origenV3 --clone-root $clonV3
.\.venv\Scripts\python.exe scripts/11_run_publication_experiments.py --config config/config_publication_v3_5seed.yaml --plan-only --pilot
```

Estas órdenes copian solo desarrollo y generan la asignación fuente; la última
solo imprime el plan. Ninguna entrena o abre `source_test`, `target_calibration`
ni `target_test`. Antes de autorizar el piloto se deben comprobar las huellas
generadas, la copia de la licencia BUS-BRA, Git limpio, CUDA y espacio libre.
El informe detallado de copia permanece ignorado por Git bajo `results/`.
Los pesos ImageNet iniciales son los que carga `BaselineClassifier` mediante
`timm` 1.0.27, no los de torchvision. Se precargaron sin entrenar desde
`timm/resnet18.a1_in1k` (revisión HF `491b427b45c94c7fb0e78b5474cc919aff584bbf`,
`model.safetensors` SHA-256 `80c49dee3da4822c009c5a7fe591e9223c5a2cfcf95a4067ca4dfb5a7b89c612`)
y `timm/efficientnet_b0.ra_in1k` (revisión HF
`1b5383e5f79cc0f7fc067e372f8f26a5fa73f26a`, SHA-256
`d569899762ea9b1384ee07f4af64805cf8caa1c55f9253ebb1080dc40e87a2cd`).
Las huellas deben coincidir al iniciar y reanudar el piloto.

## Piloto y límite de tiempo

La RTX 2060 SUPER de 8 GB es la ruta principal, sin contratar GPU. La duración
histórica de v2 no garantiza la de v3. El piloto reutilizable consta de 20
trabajos: 13 EfficientNet-B0/semilla 17 (principal y controles), seis variantes
de sensibilidad ResNet-18/semilla 17 y el checkpoint fuente ResNet-18/semilla
17 que ellas requieren. No abre `source_test`, `target_calibration` ni
`target_test`; puede cronometrarse inferencia de desarrollo sobre
`target_adapt` sin usar etiquetas para seleccionar modelos.

La proyección debe incluir las 160 variantes, inferencia de desarrollo,
controles y tiempo ya consumido. Solo se continúa si el total previsto es
**≤16 horas**, la VRAM pico cabe en 8 GB y quedan **≥20 GB libres**. Se reservan
2 horas para incertidumbre; el runner se detiene entre trabajos y no debe
superar deliberadamente **18 horas acumuladas**. Si el piloto falla, no se
mezclarán semillas locales/remotas ni se presentará una matriz parcial como
estudio de cinco semillas. Se replanteará el alcance con el autor.

`--resume` exige las mismas huellas. Nunca se debe borrar
`TRAINING_ACTIVE.lock` sin verificar si el proceso sigue activo y documentar
la interrupción. Los checkpoints v3 permanecerán fuera de Git.

El comando de arranque, **reservado para una autorización posterior**, será:

```powershell
.\.venv\Scripts\python.exe scripts/11_run_publication_experiments.py --config config/config_publication_v3_5seed.yaml --pilot --max-wall-hours 18
```

Si el gate del piloto queda aprobado, el resto de la matriz se retoma con
`--resume --stage all --max-wall-hours 18`. No ejecutar estas órdenes como
parte de la preparación actual.

## Compuerta ética y etapa posterior

El entrenamiento de desarrollo está aprobado por el autor, pero
`target_calibration` y `target_test` siguen bloqueados hasta obtener una
determinación ética escrita o equivalente institucional. Aunque se completen
los 150 checkpoints entrenados y 10 AdaBN, el finalizador no se ejecutará
hasta recibirla. Abrir el test requerirá una decisión separada y un registro
de acceso único. El artículo no se actualizará hasta contrastar artefactos
v3 completos. La cobertura total del APC de OJEMB también sigue por confirmar;
la aprobación académica del artículo no acredita esa cobertura.
