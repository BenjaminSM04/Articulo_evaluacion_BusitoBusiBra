# Ejecucion v3 local con controles

Plan aprobado por el autor el 23-09-2026. El trabajo se realiza en la rama
`ojemb/v3-local-controls`; no autoriza abrir `target_calibration` ni `target_test`.

## Task 1: Entorno reproducible

- Fijar Python 3.11.9 y PyTorch 2.5.1+cu121/torchvision 0.20.1+cu121.
- Resolver todas las dependencias con wheels compatibles con Windows.
- Exigir CUDA funcional, pruebas base y auditoria de alcance publico.

## Task 2: Cohorte y particiones fuente v3

- Construir la cohorte fuente de 358 imagenes a partir del mapeo publico y la
  auditoria: exactamente 28 exclusiones (9 duplicados, 14 axila, 5 aguja).
- Agrupar los pares 121/102, 621/639 y 644/576; registrar 376/499 y 380/630
  como falsos positivos del cribado.
- Regenerar solo fuente en 64/16/20, semilla 20260723, sin grupos compartidos.
- Implementar `--source-only`, huellas y pruebas que demuestren que los
  manifiestos target reservados no se leen ni se escriben.

## Task 3: Registro experimental y controles

- Ampliar configuracion y registro a 160 variantes: 80 principales, 20 de
  intensidad, 20 ROI, 10 AdaBN y 30 de sensibilidad.
- Implementar percentiles RGB agrupados 1/99 antes del redimensionado y salida
  cero para un rango agrupado nulo.
- Implementar ROI por union de mascaras con margen `ceil(10 %)`, recorte al
  marco y fallo ante mascara ausente, vacia o desalineada.
- Implementar AdaBN desde `source_only_matched`, ordenado por `sample_id`,
  batch 16, sin gradiente, dropout desactivado y BN acumulativa.
- Extender el indice de checkpoints con familia, variante, hiperparametro,
  padre y huellas.

## Task 4: Runner y limites operativos

- Incorporar `--plan-only`, `--pilot`, `--stage`, `--resume` y
  `--max-wall-hours 18`.
- Hacer el piloto reutilizable de 20 trabajos: los 19 previstos y el
  checkpoint fuente ResNet-18 que requieren las seis sensibilidades.
- Bloquear cualquier carga de `source_test`, `target_calibration` y
  `target_test` durante plan, piloto y entrenamiento.
- Exigir proyeccion total <=16 h, VRAM <=8 GB y al menos 20 GB libres antes de
  continuar tras el piloto.

## Task 5: Metricas y familias estadisticas

- Incluir NLL, Brier, ECE y sensibilidad con umbrales 0.5, Youden fuente y
  Youden de calibracion.
- Mantener la familia primaria de seis comparaciones con Holm, una familia
  secundaria separada para controles y sensibilidad descriptiva.

## Task 6: Documentacion, CI y publicacion

- Actualizar protocolo, README, PREPARACION_V3 y huellas ejecutadas.
- Resolver Ruff y ejecutar toda la suite sintetica sin datos privados.
- Verificar que Git no contiene datos, predicciones ni checkpoints.
- Publicar la rama mediante PR y fusionar solo con CI verde.

## Task 7: Datos privados, piloto y corrida

- Copiar solo los datos autorizados al clon ignorado por Git y verificar
  licencias, hashes, commit y entorno.
- Ejecutar el piloto; continuar solo si supera todos los limites.
- Completar 150 checkpoints entrenados y 10 AdaBN con `--resume`.
- Pausar antes del finalizador hasta recibir la determinacion etica escrita.
