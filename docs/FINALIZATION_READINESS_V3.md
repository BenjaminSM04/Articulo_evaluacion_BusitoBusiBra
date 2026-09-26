# Preparación de la evaluación final v3

Estado documentado el 26 de septiembre de 2026.

## Entrenamiento congelado

- Matriz: 160 variantes únicas y exitosas.
- Arquitecturas: ResNet-18 y EfficientNet-B0.
- Semillas: 17, 42, 73, 101 y 202.
- Artefactos: 150 checkpoints entrenados y 10 transformaciones AdaBN.
- Duración acumulada registrada: 2,931176 horas.
- El índice conserva la huella SHA-256 de cada checkpoint.

ResNet-50 y DenseNet-121 pertenecen al estudio histórico de cuatro
arquitecturas. No forman parte de la matriz v3 de cinco semillas. No se utilizó
una arquitectura denominada ResNet-10.

La corrida histórica de aproximadamente 18 horas incluyó ResNet-18,
ResNet-50, EfficientNet-B0 y DenseNet-121, validación cruzada de cinco folds,
métodos de adaptación y tres presupuestos de fine-tuning. La v3 utiliza dos
arquitecturas pequeñas, reutiliza checkpoints fuente para las continuaciones,
entrena la adaptación durante 15 épocas, usa precisión mixta y ejecuta AdaBN
sin gradientes. Por ello, 2,93 horas en la RTX 2060 SUPER es coherente con los
registros observados.

## Base ética

La justificación aplicable es el análisis secundario de datos públicos y
desidentificados, documentada en
`docs/ETHICS_BASIS_PUBLIC_DEIDENTIFIED_DATA.md`. El finalizador exige la huella
de este documento mediante `--ethics-basis`; no atribuye una exención o
aprobación inexistente a la universidad.

## APC de OJEMB

Para preparar el paquete editorial se adopta la hipótesis de trabajo de que la
Universidad del Valle cubrirá íntegramente el APC. Esta hipótesis debe
confirmarse antes del envío o de aceptar cualquier factura. La confirmación
pendiente no cambia el análisis científico ni bloquea la preparación local.

## Preparación privada previa

El siguiente comando copia y verifica por SHA-256 los manifiestos, imágenes y
máscaras congelados necesarios para la evaluación. No ejecuta inferencia, no
produce métricas y no crea `TEST_ACCESS_STARTED.json`.

```powershell
.\.venv\Scripts\python.exe scripts/15_stage_v3_private_development.py `
  --stage-finalization `
  --source-root D:\busgen `
  --clone-root D:\busgen-ojemb-code
```

## Comando reservado para la evaluación final

No ejecutar hasta decidir formalmente abrir el test:

```powershell
.\.venv\Scripts\python.exe scripts/12_finalize_publication_inference.py `
  --config config/config_publication_v3_5seed.yaml `
  --unlock-test `
  --ethics-basis docs/ETHICS_BASIS_PUBLIC_DEIDENTIFIED_DATA.md
```

Después de la evaluación corresponderá ejecutar el análisis estadístico,
generar tablas y figuras, actualizar el manuscrito OJEMB y responder las
observaciones del revisor con los resultados v3.
