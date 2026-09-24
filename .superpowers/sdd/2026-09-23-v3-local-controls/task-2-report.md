# Task 2: cohorte y particiones fuente v3

## Cambios

- El cruce auditable acepta `(label, image_id)` global o `(label, class_id)`;
  exige claves únicas y cobertura completa de las 386 filas del manifiesto.
- La auditoría pública retiene 358 imágenes: benign 212 y malignant 146. Las
  28 exclusiones coinciden exactamente con 9 `duplicate_group_member`, 14
  `objection_axilla` y 5 `objection_needle`.
- Se mantienen juntos benign 121/102, malignant 621/639 y malignant 644/576.
  Los pares 376/499 y 380/630 no comparten grupo.
- La semilla 20260723 produce train 229, val 57 y test 72, sin grupos que
  crucen particiones.
- `--source-only` lee solo el manifiesto BUSI y la auditoría pública y escribe
  los tres manifiestos fuente, asignaciones, hashes y metadata fuente. La ruta
  de ejecución retorna antes de resolver manifiestos o particiones target.
  Sin la bandera sigue activo el generador v2.
- La configuración v3 declara la nueva cohorte fuente y versión de protocolo.

## TDD y verificación

- RED: `.venv\Scripts\python.exe -m pytest tests/test_publication_splits.py -k 'v3_source_cohort or source_only_writes or v3_source_crosswalk' -q`
  falló en colección porque `create_publication_source_only_files` aún no
  existía.
- GREEN/regresión: `.venv\Scripts\python.exe -m pytest tests/test_publication_splits.py tests/test_publication_v3_config.py -q`
  terminó con `18 passed`.
- Lint: `.venv\Scripts\ruff.exe check src/data/publication_splits.py tests/test_publication_splits.py`
  terminó con `All checks passed!`.
- Espacios: `git diff --check` terminó sin errores.

## Decisiones

- El crosswalk prioriza la coincidencia global cuando ambas claves están
  disponibles y rechaza si `image_id` y `class_id` apuntan a filas distintas.
- Las tres parejas aprobadas se aplican como grupos fuente explícitos después
  del filtro `keep`; no se agrupan las parejas de falsos positivos.
- Los hashes incluyen asignaciones fuente y los bytes de entrada del manifiesto
  y auditoría. El modo fuente no crea ni reemplaza nombres de artefactos target.

## Commit

`c06ee4b Ajusta cohorte y particiones fuente v3`
