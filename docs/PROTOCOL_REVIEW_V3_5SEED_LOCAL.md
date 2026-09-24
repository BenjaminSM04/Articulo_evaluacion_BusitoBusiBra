# Revisión OJEMB v3: cinco semillas en equipo local

**Estado al 23 de septiembre de 2026:** protocolo aprobado por el autor; código,
particiones y controles en preparación. No se debe interpretar la existencia de
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

El clon público no contiene imágenes, máscaras, manifiestos de pacientes,
particiones históricas, predicciones, checkpoints ni resultados. El respaldo
`D:\busgen` no se modifica. Antes del piloto, copiar al clon local privado:

1. BUSI original, Curated BUSI v1 y BUS-BRA, con licencias y avisos cotejados.
2. `data/processed/busi_curated_manifest.csv` y
   `data/processed/bus_bra_manifest.csv`, cotejados con los mapeos públicos.
3. `results/metrics/bus_bra_adaptation_test_split.csv` y solo las asignaciones
   históricas necesarias de BUS-BRA. Preservar sus bytes y huellas. La
   generación `--source-only` no debe leer ni sobrescribir las asignaciones
   históricas del destino ni los archivos de test.
4. SHA-256 de configuración, commit, código, CSV de auditoría, manifiestos,
   imágenes y máscaras de desarrollo, particiones y entorno; Python, PyTorch,
   CUDA, controlador y GPU. Cambios de bytes impiden `--resume` sin auditoría.

Ni los datos privados ni sus índices por paciente se subirán a Git o a nube.
`cross_validation.n_folds: 5` es heredado; las réplicas las controla
`publication.seeds`.

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

## Compuerta ética y etapa posterior

El entrenamiento de desarrollo está aprobado por el autor, pero
`target_calibration` y `target_test` siguen bloqueados hasta obtener una
determinación ética escrita o equivalente institucional. Aunque se completen
los 150 checkpoints entrenados y 10 AdaBN, el finalizador no se ejecutará
hasta recibirla. Abrir el test requerirá una decisión separada y un registro
de acceso único. El artículo no se actualizará hasta contrastar artefactos
v3 completos. La cobertura total del APC de OJEMB también sigue por confirmar;
la aprobación académica del artículo no acredita esa cobertura.
