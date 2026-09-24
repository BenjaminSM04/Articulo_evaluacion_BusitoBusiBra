# Informe de controles locales de preprocesamiento v3

## Entrega

- `src/training/control_preprocessing.py`: normalización RGB por percentiles 1 y 99 y recorte por unión de máscaras con margen por dimensión.
- `tests/test_control_preprocessing.py`: fixtures numéricos para clipping, canales de rango nulo, caja inclusiva, redondeo del margen y bordes; valida máscaras ausentes, vacías y desalineadas.

Ambas funciones reciben arreglos en memoria, no abren archivos ni consultan datos del conjunto privado. La normalización retorna un arreglo nuevo uint8; el recorte también retorna una copia.

## Verificación

- TDD RED: `pytest tests/test_control_preprocessing.py -q` falló al importar `src.training.control_preprocessing`, como se esperaba antes de crear el módulo.
- TDD GREEN: `pytest tests/test_control_preprocessing.py -q` — 9 pruebas pasaron.
- Ruff sobre los dos archivos nuevos — todos los checks pasaron.
- Suite completa `pytest` — colección detenida por `ImportError` en `tests/test_publication_splits.py`: el test importa `create_publication_source_only_files`, que todavía no está disponible en `src.data.publication_splits`. Ese archivo y sus cambios son ajenos a esta entrega.

## Auto-revisión

- Percentiles calculados sobre los píxeles de entrada, independientemente para cada canal RGB; canales cuyo p99 coincide con p1 permanecen en cero.
- ROI une las máscaras usando foreground `> 0`, mide la caja con extremos inclusivos, redondea hacia arriba el 10% de cada dimensión y limita el recorte al marco.
- Las entradas de intensidad con forma o dtype no admitidos fallan con `ValueError`; ROI reporta máscara ausente, máscara sin foreground, forma incorrecta o desalineación con `ValueError`.
- Sólo se incluyeron estos dos módulos nuevos y este informe en el cambio local; los demás cambios del worktree quedaron fuera del commit.
