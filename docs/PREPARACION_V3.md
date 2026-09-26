# Preparacion de la cohorte y protocolo v3

Decisiones aprobadas por el autor en esta conversacion el 23 de septiembre de
2026. Esta aprobacion no equivale a una determinacion etica institucional. La
inferencia sobre `target_calibration` y `target_test` sigue bloqueada hasta que
exista constancia escrita. Los IDs de BUSI siguientes son de imagenes publicas;
no identifican pacientes.

## Cohorte fuente congelada

La fuente primaria es la interseccion de las 386 imagenes binarias de Curated
BUSI v1 con la auditoria publica de Pawlowska. El cruce usa clase e ID dentro
de clase y exige correspondencia uno a uno. Se conservan exactamente 358
imagenes: 212 benignas y 146 malignas. Se excluyen exactamente 28:

| Motivo registrado | Imagenes |
|---|---:|
| Miembro duplicado de grupo | 9 |
| Objecion de axila | 14 |
| Objecion de aguja | 5 |

Se conservan, siempre dentro de un mismo split, los pares de IDs globales
benign 121/102, malignant 621/639 y malignant 644/576. La inspeccion visual
respalda su relacion regional, pero no permite afirmar identidad de paciente.
Los candidatos benign 376/malignant 499 y benign 380/malignant 630 son falsos
positivos del cribado perceptual y quedan en grupos separados.

La auditoria publica de las 358 imagenes retenidas registra 4 con la etiqueta
explicita `overlay` (benign 407, 408, 410 y malignant 490). Esta cuenta no
incluye necesariamente todo texto o medidor visible: hay otras etiquetas de
anotacion. La sensibilidad ROI evaluara el efecto de la region anotada sin
seleccionar variantes por el test.

La particion fuente es disjunta por grupos, usa semilla 20260723 y proporciones
64/16/20. Los tamanos esperados son 229/57/72 para train/val/test. Se
regeneran solo asignaciones fuente; las asignaciones BUS-BRA historicas se
conservan. Ninguna funcion de entrenamiento recibe `source_test`,
`target_calibration` o `target_test`.

## Controles y matriz

La matriz contiene 160 variantes: 80 principales, 20 de intensidad, 20 ROI,
10 AdaBN y 30 de sensibilidad de hiperparametros. Intensidad aplica
percentiles 1/99 agrupados de todos los pixeles RGB originales, antes del
redimensionado; un rango nulo produce ceros. ROI usa la union de mascaras con
margen `ceil(10 %)` por dimension de la caja y falla ante ausencia, vacio o
desalineacion. AdaBN reinicia estadisticas de BatchNorm y hace una pasada
ordenada por `sample_id` sobre `target_adapt`, batch 16, momentum acumulativo,
sin gradientes ni dropout.

Las seis comparaciones principales DANN/CORAL/MMD frente a fuente equiparada
comparten Holm. Los seis controles AdaBN/intensidad/ROI forman una familia
secundaria separada. La sensibilidad de hiperparametros es descriptiva.
Se reportaran NLL, Brier, ECE y sensibilidad a umbrales 0.5, Youden fuente y
Youden de calibracion cuando esta ultima quede autorizada.

## Procedencia y huellas

| Artefacto | SHA-256 o estado |
|---|---|
| Mapeo Curated BUSI ejecutado | `4aa2096b4b8ef34d3dc03ddcd513f059020601ef89f2dc06741214f1e197f848` |
| Auditoria BUSI ejecutada | `ff09d0bdc2f672380ee1d7b48f76f311f9f0faaa1fc1298645ca4f518cdb11c6` |
| Configuracion v3 y commit ejecutado | Pendiente de congelar tras CI |
| Asignaciones fuente v3 | Pendiente de regenerar en almacenamiento local privado |
| Datos y entorno | Pendiente de copiar y cotejar SHA-256 antes del piloto |

El piloto local incluye 13 trabajos de EfficientNet-B0/semilla 17, las seis
sensibilidades ResNet-18/semilla 17 y el checkpoint fuente ResNet-18/semilla 17
que estas necesitan: 20 checkpoints reales. Se continuara solo si tiempo
consumido mas proyeccion total es como maximo 16 horas, la VRAM cabe en 8 GB
y quedan al menos 20 GB libres. El limite acumulado absoluto es 18 horas.

El dataset BUS-BRA local incluye una licencia que permite su uso y exige citar
el trabajo original y conservar el aviso en copias sustanciales. Antes de la
corrida se verificaran ademas las condiciones de las imagenes BUSI y la
disponibilidad/licencia de las mascaras usadas por ROI.
