# Metodología

Documento de metodología para el artículo. Describe el diseño experimental, el pipeline técnico y
los criterios de evaluación con suficiente detalle para garantizar la reproducibilidad. Todo
parámetro citado aquí vive en [`config/config.yaml`](../config/config.yaml).

---

## 1. Diseño del estudio

El estudio es un **experimento controlado de generalización cruzada** entre dos poblaciones. Se
entrena un clasificador binario (benigno vs. maligno) sobre un dominio *source* y se evalúa tanto
en su propio dominio (intra-dominio) como en el dominio *target* (cross-domain). La hipótesis
central es que existirá una caída de desempeño cross-domain atribuible al *domain shift* entre la
población egipcia (BUSI) y la brasileña (BUS-BRA), y que las técnicas de adaptación de dominio la
reducirán parcialmente.

Las preguntas de investigación y la justificación bibliográfica están en
[`01_revision_literatura.md`](01_revision_literatura.md). La tarea se restringe a **benigno/maligno**
porque es la etiqueta comparable entre ambos datasets; la clase *normal* (presente solo en BUSI) se
excluye del experimento principal para evitar una definición de tarea asimétrica entre dominios.

## 2. Datos y preprocesamiento

**Fuentes.** BUSI (Egipto, 780 imágenes) y BUS-BRA (Brasil, 1875 imágenes). Detalle y descarga en
[`../data/README.md`](../data/README.md).

**Deduplicación (paso crítico).** BUSI contiene ~235 imágenes duplicadas e imágenes que no son de
mama (axila). Antes de cualquier partición se aplica deduplicación por *perceptual hashing* (pHash,
distancia de Hamming ≤ 5 configurable); los descartes se registran en `data/processed/dedup_report.csv`.
Esto evita la fuga de datos que, de otro modo, inflaría las métricas intra-dominio de BUSI.

**Normalización y formato.** Las imágenes se redimensionan a 224×224, se convierten a 3 canales
(la ecografía es monocroma, se replica el canal) y se normalizan con los estadísticos de ImageNet
(media `[0.485, 0.456, 0.406]`, desv. `[0.229, 0.224, 0.225]`) para ser compatibles con los
*backbones* preentrenados.

**Unificación de etiquetas.** Se mapea `benign→0`, `malignant→1` en ambos datasets, y se genera un
*manifest* CSV por dataset con columnas `image_path, dataset, label, label_idx, patient_id, birads,
original_path`. Los manifests son la única interfaz entre el preprocesamiento y el resto del
pipeline.

**Augmentation (solo entrenamiento).** Volteo horizontal, rotaciones ≤ 15°, jitter de brillo/
contraste y ruido gaussiano leve. Se evita el volteo vertical y los recortes agresivos porque
alteran la semántica clínica de la imagen. La evaluación no usa augmentation.

## 3. Arquitecturas y transfer learning

Se comparan dos *backbones* preentrenados en ImageNet, obtenidos vía `timm`:

- **ResNet-50** — referencia robusta y ampliamente reportada en el área.
- **EfficientNet-B0** — alternativa eficiente con buena relación desempeño/coste.

Sobre el extractor de características se coloca una cabeza de clasificación (dropout + capa lineal a
2 clases). La estrategia de *transfer learning* por defecto es **fine-tuning completo** (todo el
backbone se actualiza) partiendo de los pesos de ImageNet, con *warmup* y *learning rate* bajo
(`1e-4`). Se contempla como ablación el congelamiento del backbone.

## 4. Protocolo de validación y entrenamiento

**Validación cruzada intra-dominio.** Para cada dataset se usa validación cruzada estratificada de
`k = 5` folds. En BUS-BRA, que dispone de identificador de paciente, los folds se **agrupan por
paciente** (`StratifiedGroupKFold`) para impedir que imágenes del mismo paciente caigan a la vez en
entrenamiento y prueba. BUSI **no provee identificador de paciente**, por lo que su partición es a
nivel de imagen estratificada; esta limitación se reporta explícitamente y es una razón adicional
para deduplicar.

**Entrenamiento.** Optimizador AdamW, *scheduler* coseno con *warmup*, precisión mixta (AMP),
*early stopping* sobre el AUC de validación (paciencia 10), pérdida de entropía cruzada con
**pesos por clase** (inverso de frecuencia) para manejar el desbalance benigno/maligno. Todas las
semillas se fijan (`seed = 42`) y cuDNN se configura en modo determinista.

## 5. Matriz de generalización cruzada

El experimento central produce una **matriz N×N** (con N = nº de dominios). Para cada dominio
*source* se entrena un modelo sobre su partición de entrenamiento/validación y se reserva un *test*
propio. Luego:

- la **diagonal** (entrenar y evaluar en el mismo dominio) reporta el desempeño intra-dominio sobre
  el test reservado del source;
- las **celdas fuera de la diagonal** (entrenar en X, evaluar en Y) reportan el desempeño
  cross-domain del modelo del source sobre **todo** el dataset target.

La métrica principal de cada celda es el **AUC** con intervalo de confianza por *bootstrap*. Para
cada source se reporta la **brecha de generalización** = AUC intra − AUC cross media.

## 6. Adaptación de dominio

Sobre las celdas cross-domain se evalúan tres estrategias, comparables entre sí porque comparten
arquitectura, particiones y métricas:

1. **Fine-tuning (few-shot supervisado).** Se reentrena el modelo del source con una pequeña muestra
   etiquetada del target (`n_target_labeled` por clase); el resto del target se usa para evaluar.
2. **DANN (adversarial, no supervisado).** Extractor compartido + cabeza de tarea + cabeza de
   dominio con *gradient reversal*; usa etiquetas del source y datos del target sin etiqueta, con
   programación creciente de λ (Ganin et al., 2016).
3. **Self-training (pseudo-etiquetado).** El modelo del source etiqueta el target; las predicciones
   de confianza ≥ 0.9 se añaden como pseudo-etiquetas y se reentrena, iterando varias rondas.

La comparación se hace contra la línea base sin adaptación (`method: none`) para cuantificar cuánta
caída recupera cada técnica.

## 7. Métricas y análisis estadístico

Se reportan **accuracy, sensibilidad, especificidad, AUC-ROC, F1 y balanced accuracy**, con la clase
*maligno* como positiva (la prioridad clínica es no perder cánceres). El umbral de decisión por
defecto es 0.5. Los **intervalos de confianza al 95 %** se calculan por *bootstrap* percentílico
(1000 remuestreos). La comparación entre métodos de adaptación se basa en la diferencia de AUC y sus
intervalos; si se requiere contraste de hipótesis, se recomienda el test de DeLong para AUCs
pareados (a incorporar en análisis).

## 8. Explicabilidad y su evaluación

Se generan **Grad-CAM** y **SHAP** sobre una muestra balanceada de casos (verdaderos/falsos
positivos y negativos). El análisis distintivo del proyecto es comparar las explicaciones
**intra-dominio vs. cross-domain**: se inspecciona si el modelo sigue atendiendo a la lesión al
cambiar de población o si se desplaza hacia artefactos de adquisición (texto, *calipers*, sombras).
Cuando exista máscara de lesión (BUSI y BUS-BRA la incluyen), se recomienda **cuantificar** el
solapamiento entre el mapa de saliencia y la lesión (p. ej. fracción de energía del mapa dentro de
la máscara) para no quedarse en lo cualitativo.

## 9. Reproducibilidad

Cada ejecución guarda el `config.yaml` efectivo y el *hash* de git en `results/reports/`. El flujo
completo (descarga → preprocesamiento → baselines → matriz → XAI → reporte) está descrito en el
[`README.md`](../README.md) y encapsulado en los seis scripts numerados de `scripts/`. Las pruebas
de humo (`pytest`) verifican el grafo de imports, el *forward* de los modelos, las métricas y la
ausencia de fuga de paciente en las particiones.

## 10. Limitaciones

El tamaño de BUSI es modesto y carece de identificador de paciente, lo que impide agrupar sus
particiones por sujeto. Los dos dominios mezclan *acquisition shift* y *cohort shift*, por lo que el
diseño cuantifica el efecto **conjunto** y no permite, por sí solo, separar ambas causas. Los mapas
de saliencia son indicativos, no explicaciones causales. Estas limitaciones se discutirán
explícitamente en el artículo.
