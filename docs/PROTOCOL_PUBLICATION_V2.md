# Protocolo congelado de la corrida publicable (v2.0)

Este documento define el análisis que reemplaza los resultados exploratorios del manuscrito.
Debe leerse junto con `config/config_publication.yaml`. La inferencia final sobre BUS-BRA
`target_test` se ejecuta una sola vez, después de completar entrenamientos y verificaciones.

## Cohortes

- Fuente: Curated BUSI v1.0, DOI `10.5281/zenodo.19047974`; solo benigno/maligno
  (222/164). BUSI no publica identificadores de paciente. La partición fuente es estratificada
  por imagen: 64 % entrenamiento, 16 % validación y 20 % prueba.
- Objetivo: BUS-BRA, con particiones por paciente. Se conserva el holdout histórico
  (426 pacientes; 736 imágenes), que ya había sido inspeccionado durante los experimentos
  exploratorios anteriores. Por ello, v2 se presenta como un reanálisis preespecificado y no
  como una validación externa nunca observada. El antiguo pool de adaptación se divide en
  `target_adapt` y `target_calibration`, sin pacientes compartidos.
- Los dos dominios parten de sus píxeles originales y reciben el mismo redimensionado a 224×224
  durante la carga. Los PNG BUS-BRA preprocesados de la fase exploratoria no se usan en v2.

## Comparación no supervisada

ResNet-18 es la arquitectura primaria y EfficientNet-B0 el análisis de robustez. Cada una se
entrena con tres semillas (17, 42 y 73). Para cada arquitectura/semilla se crea un único
checkpoint fuente; todos los métodos parten de una copia exacta:

1. control fuente con continuación emparejada;
2. DANN;
3. CORAL;
4. MMD.

Todos reciben las mismas filas fuente, el mismo número de actualizaciones fuente, optimizador,
tasa de aprendizaje y calendario. Los modelos se seleccionan únicamente con AUC de validación
fuente. El comparador primario de cada adaptación es el control fuente emparejado. El modelo
fuente sin continuación se conserva como descripción del transporte directo.
Durante las continuaciones de adaptación se congelan las estadísticas móviles de BatchNorm
para evitar que una exposición implícita a BUS-BRA (AdaBN) se confunda con el efecto de la
pérdida DANN, CORAL o MMD.
El entrenamiento fuente usa AdamW, tres épocas de calentamiento lineal y decaimiento coseno.
Las continuaciones de adaptación y ajuste fino usan decaimiento coseno sin calentamiento de
tasa; las cinco épocas de calentamiento de CORAL/MMD escalan solo la pérdida de alineación.
DANN utiliza la rampa logística preespecificada de inversión de gradiente.

Self-training se excluye de la corrida confirmatoria: el protocolo previo lo reiniciaba desde
ImageNet y su costo no permitía una comparación emparejada mínima.

## Ajuste fino

Los presupuestos de 5 %, 10 % y 20 % se definen sobre pacientes de `target_adapt`, estratificados
por clase y anidados dentro de cada réplica. Todas las imágenes de un paciente permanecen juntas.
Entrenamiento y validación cuentan dentro del presupuesto etiquetado. Las mismas cohortes se usan
para ambas arquitecturas. Debido al tamaño reducido de la validación al 5 %, las 20 épocas son
fijas y preespecificadas: la validación por paciente se monitoriza, pero no elige la época.

## Calibración y análisis

- Resultado primario: ROC-AUC sin calibrar por paciente.
- Agregación primaria: media del logit maligno dentro de paciente.
- Resultado confirmatorio para fuente/control/UDA: ensamble preespecificado que promedia, en
  escala logit, las predicciones de las tres semillas. Las métricas individuales se conservan
  para reportar media y desviación estándar entre semillas.
- El ajuste fino no se ensambla: cada réplica usa una cohorte etiquetada distinta y un ensamble
  excedería el presupuesto nominal. Se reporta la distribución y media ± desviación estándar de
  las tres réplicas; su variabilidad combina selección de pacientes, inicialización y orden.
- Calibración secundaria: escalado de temperatura y umbral de Youden ajustados solo en
  `target_calibration`, y congelados antes de `target_test`.
- Se reportan AUC, PR-AUC, Brier, sensibilidad, especificidad, exactitud balanceada y F1.
- Intervalos y diferencias pareadas: 5000 remuestreos estratificados por paciente.
- Comparaciones UDA: DANN/CORAL/MMD contra control emparejado, con corrección de Holm.
- Se reporta también media y desviación estándar entre semillas.
- La prueba fuente se analiza por imagen porque BUSI no contiene identificadores de paciente.

## Bloqueo de test

El comando `train` no recibe filas, etiquetas ni imágenes de `target_test`. El comando
`finalize --unlock-test` verifica que todos los checkpoints, manifests, particiones y hashes
esperados estén presentes antes de ejecutar la única inferencia final.
