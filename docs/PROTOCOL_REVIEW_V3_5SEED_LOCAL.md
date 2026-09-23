# Revisión OJEMB v3: cinco semillas en equipo local (ejecución aplazada)

**Estado: código preparado; no se ha entrenado, inferido ni abierto el test v3.** Esta etapa solo publica herramientas y protocolo. Ningún comando de entrenamiento o evaluación final debe ejecutarse hasta que el autor cierre la auditoría visual, los controles y el plan de análisis, y autorice expresamente una nueva corrida. El artículo y los resultados v2 permanecen fuera de este repositorio.

## Diseño científico que se conservará

- Configuración separada: `config/config_publication_v3_5seed.yaml`, cinco semillas `17, 42, 73, 101, 202` y salidas exclusivamente en `results/publication_v3_5seed/`. Los pesos iniciales son preentrenados en ImageNet; repetir el entrenamiento **no** significa inicializar redes al azar.
- Dos arquitecturas: ResNet-18 y EfficientNet-B0. Cada semilla incluye fuente directa, control de fuente equiparada, DANN, CORAL, MMD y ajuste fino al 5 %, 10 % y 20 % de los pacientes de adaptación etiquetados: 80 checkpoints previstos. La selección de pacientes para ajuste fino también varía por semilla; su dispersión no estima solo variación de inicialización.
- Curated BUSI conserva 386 imágenes para la tarea binaria. La partición fuente v3 se hace disjunta por grupos de imágenes relacionadas documentados en la auditoría pública: cuatro grupos cruzaban splits v2 y seis imágenes cambiaron de partición, preservando tamaños y clases. Esto **no** demuestra separación por paciente ni reemplaza una revisión visual.
- BUS-BRA conserva las cohortes y la partición histórica por paciente. El archivo privado `results/metrics/bus_bra_adaptation_test_split.csv` es imprescindible; no debe sustituirse por un split nuevo. `cross_validation.n_folds: 5` es una clave heredada y no controla las cinco réplicas: lo hace `publication.seeds`.
- Los checkpoints v2 son históricos y no se mezclarán con v3. Como la fuente BUSI también cambió, una diferencia v2–v3 no puede atribuirse exclusivamente a aumentar el número de semillas.
- El test BUS-BRA tuvo exposición exploratoria previa. La inferencia futura será **descriptiva** y no una validación externa nueva. Una cohorte independiente sigue siendo necesaria. Los intervalos bootstrap por paciente (condicionados en modelos entrenados) y la media/desviación entre semillas deben comunicarse por separado.

## Entradas privadas y verificación previa

Este clon no contiene imágenes, máscaras, manifiestos de pacientes, particiones históricas, checkpoints ni resultados. El trabajo previo permanece en un respaldo local privado, que **no debe modificarse ni limpiarse** para preparar este repositorio. Antes de una futura ejecución se necesitarán, en almacenamiento privado local:

1. BUSI original, la copia Curated BUSI v1 para cotejar selección y BUS-BRA, con licencias verificadas.
2. `data/processed/busi_curated_manifest.csv` y `data/processed/bus_bra_manifest.csv`, derivados de los datos autorizados y validados contra los mapeos públicos de `resources/busi_curation/`.
3. `results/metrics/bus_bra_adaptation_test_split.csv` y las asignaciones v3 locales preparadas anteriormente. Cotejar tamaños, IDs y SHA-256 con la copia privada bloqueada; si falta algún artefacto, detenerse. No regenerar la cohorte histórica con otra semilla.
4. Un registro inmutable de SHA-256 de configuración, código, CSV de auditoría, manifiestos y asignaciones, además de versiones de Python, PyTorch, CUDA, controlador y GPU. Cualquier cambio de bytes exige revisar nuevamente las huellas antes de reanudar.

Las rutas relativas de la configuración son internas al equipo local. Ni el ZIP de datos ni los manifiestos privados se subirán a Git o a servicios de nube como parte de esta etapa.

## Bloqueos científicos antes del piloto

- Completar y firmar la revisión visual de grupos BUSI relacionados, los 28 registros cuya elegibilidad difiere entre curaciones y los candidatos perceptuales. Registrar también la decisión sobre texto, overlays, agujas y otras marcas visibles.
- Definir y probar, usando solo datos de desarrollo, AdaBN, normalización de intensidad, sensibilidad con ROI de máscara y selección/sensibilidad de hiperparámetros. La ROI con anotación se etiquetará como tal, nunca como localizador autónomo utilizable en clínica.
- Congelar particiones, reglas de calibración, comparaciones, métricas y controles **antes** de acceder al test. Confirmar que el programa impide abrir `target_test` durante preparación y entrenamiento.
- Verificar que la preparación de particiones y las pruebas con fixtures sintéticos no hayan escrito en `results/publication_v2` ni abierto el test. La publicación de código no satisface ninguno de estos bloqueos.

## Presupuesto de tiempo para la futura corrida

Se utilizará el equipo local sin contratar GPU. La duración histórica de v2 **no garantiza** la duración v3 ni incluye los nuevos controles; el presupuesto se calculará con una medición nueva y autorizada.

Cuando se autorice la ejecución, se hará un piloto de una arquitectura y una semilla en el mismo entorno fijado. Se proyectará desde su tiempo real el costo de los 80 checkpoints, los controles, la inferencia y un margen para fallos. Solo se continuará si la proyección completa cabe en **16 horas**, reservando 2 horas; el límite máximo será **18 horas acumuladas**. Se registrará el tiempo tras cada lote. Si se rebasa la proyección, se detendrá y conservará el estado para reanudarlo sin presentar una matriz parcial como estudio de cinco semillas.

El runner dispone de `--resume`, sujeto a las mismas huellas de configuración, código, datos, asignaciones y entorno. Nunca se debe borrar manualmente `TRAINING_ACTIVE.lock` sin comprobar si el proceso sigue vivo y documentar la interrupción. Los archivos de salida se guardarán solo en `results/publication_v3_5seed/` y permanecerán fuera de Git.

## Test y artículo: etapa posterior y separada

Solo después de completar y bloquear modelos y controles se decidirá expresamente si ejecutar `scripts/12_finalize_publication_inference.py` con `--unlock-test`, seguido del análisis y los artefactos. No se hará en esta entrega de código. Si el test se abre, registrar el evento y verificar que se consulta una sola vez conforme al protocolo.

El manuscrito OJEMB tampoco se modificará hasta contrastar todas las predicciones, estadísticos, tablas y figuras v3 con sus artefactos. Debe seguir diciendo que el test histórico ya se había explorado y que la inferencia es descriptiva. La documentación ética y la aprobación final del autor son asuntos editoriales independientes de este repositorio.
