# Revisión de literatura

**Proyecto:** Generalización entre poblaciones de modelos de deep learning para clasificación de
cáncer de mama en ecografía, usando adaptación de dominio y explicabilidad (XAI).

Esta revisión organiza el estado del arte en cuatro ejes —(1) *domain shift* en imagen médica,
(2) datasets públicos de ecografía mamaria, (3) métodos de adaptación de dominio y
(4) explicabilidad— y cierra identificando la brecha que el proyecto pretende cubrir. Las
referencias completas, con enlaces, están al final y en [`referencias.bib`](referencias.bib).

---

## 1. Domain shift y generalización en imagen médica

Un modelo de deep learning entrenado en una institución suele perder desempeño de forma notable al
desplegarse en otra, fenómeno conocido como *domain shift*. La causa es una diferencia en la
distribución estadística de los datos entre el dominio de entrenamiento (*source*) y el de
despliegue (*target*). Es útil distinguir dos componentes: el **acquisition shift** (distinto
fabricante de ecógrafo, frecuencia del transductor, protocolo de adquisición, ajustes de ganancia)
y el **cohort shift** o desplazamiento poblacional (diferencias demográficas, geográficas,
socioeconómicas y de comorbilidad entre los pacientes). En este proyecto ambos componentes están
presentes a la vez, porque BUSI (Egipto) y BUS-BRA (Brasil) difieren tanto en equipamiento como en
población.

La magnitud del problema está bien documentada. Una revisión sistemática reciente sobre cambios
entre poblaciones en imagen médica reporta caídas de desempeño del orden de **10–25 %** al evaluar
modelos sobre poblaciones no vistas, lo que confirma que el efecto es de tamaño clínicamente
relevante y no un detalle marginal (Musa et al., *Big Data and Cognitive Computing*,
2026). El survey de Yoon et al. (2024) sobre *domain generalization* para análisis de imagen médica
sistematiza las familias de soluciones (alineación de características, aumento de datos, meta-
aprendizaje, normalización específica de dominio) y sirve como marco general. En mamografía, el
estudio multicéntrico a gran escala de García-Aguilar et al. (arXiv 2201.11620) sobre detección de
masas muestra que la generalización inter-centro sigue siendo el cuello de botella incluso con
grandes volúmenes de datos. En radiografía de tórax, Wang et al. (*Scientific Reports*, 2025)
abordan específicamente el *cross-population domain shift* con adaptación adversarial supervisada,
un planteamiento muy cercano al de este proyecto pero en otra modalidad.

Una idea metodológica transferible proviene de la segmentación: Zhang et al. (BigAug, *IEEE TMI* /
PMC7393676) muestran que un aumento de datos agresivo y diverso ("deep stacked transformations")
mejora la generalización a dominios no vistos sin necesidad de datos del target. Es una línea base
barata que conviene incluir antes de métodos más complejos.

## 2. Datasets públicos de ecografía mamaria

El proyecto se apoya en dos datasets públicos como dominios principales y contempla un tercero para
validación externa.

| Dataset | País / Centro | Pacientes | Imágenes | Clases | Notas |
|---|---|---|---|---|---|
| **BUSI** | Egipto — Baheya Hospital (El Cairo) | 600 | 780 | benigno (437), maligno (210), normal (133) | Máscaras de segmentación; ~500×500 px; CC BY 4.0 |
| **BUS-BRA** | Brasil — INCA (Río de Janeiro) | 1064 | 1875 | benigno (722), maligno (342) | BI-RADS 2–5; 4 ecógrafos; particiones CV 5/10-fold |
| **UDIAT** | España — UDIAT (Sabadell) | — | 163 | benigno (110), maligno (53) | No descargable libremente (bajo solicitud) |

**BUSI** (Al-Dhabyani et al., *Data in Brief*, 2020) es el dataset más usado del área, pero tiene
limitaciones importantes de tamaño y, sobre todo, **problemas de calidad documentados**: análisis
posteriores identificaron alrededor de **235 imágenes duplicadas**, cerca de **70 imágenes que no
son de mama sino de axila**, y algunas con aguja de biopsia (ver la *Letter to the Editor* en
PMC10293973 y el recurso de openmedlab). Las duplicaciones son críticas porque, si las imágenes
repetidas se reparten entre entrenamiento y prueba, se produce **fuga de datos** que infla
artificialmente las métricas. Cualquier estudio riguroso sobre BUSI debe deduplicar antes de
particionar; en este proyecto la deduplicación es un paso obligatorio del preprocesamiento.

**BUS-BRA** (Gómez-Flores et al., *Medical Physics*, 2024) es un dataset más grande y mejor
documentado, con tumores confirmados por biopsia, anotaciones BI-RADS, cuatro ecógrafos distintos y
particiones de validación cruzada provistas para favorecer la reproducibilidad. Su mayor tamaño y su
metadato de paciente permiten —a diferencia de BUSI— hacer particiones agrupadas por paciente, lo
que evita fuga de datos a nivel sujeto.

**UDIAT** (Yap et al., *IEEE J. Biomedical and Health Informatics*, 2018) es pequeño (163 imágenes,
Siemens, Sabadell) y no es de descarga pública, pero es un tercer dominio valioso para validación
externa si se obtiene de los autores. Existen además datasets más recientes —**BUS-UCLM** (*Nature
Scientific Data*, 2025) y **QAMEBI**— que han habilitado estudios multi-dataset y que pueden
incorporarse como dominios adicionales en fases posteriores.

## 3. Métodos de adaptación de dominio

La adaptación de dominio (DA) busca recuperar el desempeño perdido por el *domain shift*. El
proyecto contempla tres familias, de menor a mayor complejidad.

**Fine-tuning / transfer learning.** La línea base obligatoria. Se parte de un modelo preentrenado
(ImageNet) y, opcionalmente, se reentrena sobre una pequeña muestra etiquetada del dominio target
(*few-shot supervised DA*). Es simple, fuerte y a menudo difícil de superar; sirve como referencia
para juzgar métodos más sofisticados.

**Adaptación adversarial (DANN).** Ganin et al. (2016) introdujeron las *Domain-Adversarial Neural
Networks*: un extractor de características compartido, un clasificador de tarea (entrenado con las
etiquetas del source) y un clasificador de dominio conectado mediante una **capa de inversión de
gradiente** (*gradient reversal layer*). Durante el retropropagado, el gradiente del clasificador de
dominio se multiplica por una constante negativa, lo que empuja al extractor a producir
características **indistinguibles entre dominios** y, por tanto, más transferibles. Es el método de
DA *no supervisado* de referencia (no requiere etiquetas del target) y ya se ha aplicado con éxito a
imagen médica (p. ej. el citado trabajo en radiografía de tórax).

**Self-training / pseudo-etiquetado.** Se usa el modelo entrenado en el source para etiquetar las
imágenes del target; las predicciones de alta confianza se añaden como pseudo-etiquetas y el modelo
se reentrena, iterando varias rondas. Es atractivo por su simplicidad y porque no necesita
etiquetas reales del target, aunque es sensible al sesgo de confirmación si el umbral de confianza
es bajo.

Otras aproximaciones recientes específicas de ecografía mamaria incluyen la **traducción de estilo
con modelos de difusión** para acercar las imágenes del target al estilo del source (ADAptation,
Chen et al., arXiv 2507.00474) y el uso de **PCA como preprocesamiento** para mejorar la validez
externa en segmentación (arXiv 2505.23587). Conviene tenerlas como extensiones, no como núcleo.

## 4. Explicabilidad (XAI) en modelos médicos

La interpretabilidad es esencial para la confianza clínica y, en este proyecto, cumple además una
función metodológica: comprobar **dónde mira** el modelo y si ese foco se mantiene al cambiar de
población. Dos técnicas dominan la literatura.

**Grad-CAM** (Selvaraju et al., 2017) genera mapas de calor que resaltan las regiones de la imagen
más influyentes en la predicción, usando los gradientes que llegan a la última capa convolucional.
Es la herramienta cualitativa estándar en imagen médica por su bajo costo y su lectura intuitiva
para radiólogos.

**SHAP** (Lundberg & Lee, 2017) asigna a cada característica (píxel o región) una contribución con
fundamento en la teoría de juegos (valores de Shapley), ofreciendo atribuciones con signo. Es más
costoso computacionalmente pero complementa a Grad-CAM con una medida de importancia más principiada.

Las revisiones recientes específicas del dominio —"Explainable AI in breast cancer ultrasound
imaging" (*Frontiers in Digital Health*, 2026) y el survey de técnicas XAI para diagnóstico de
cáncer de mama (arXiv 2406.00532)— coinciden en que Grad-CAM, LIME y SHAP son las más empleadas, y
en que la XAI mejora la confianza y la adopción clínica. Una advertencia recurrente: los mapas de
saliencia pueden ser inestables y no deben interpretarse como explicación causal; por eso conviene
**triangular** (Grad-CAM + SHAP) y, cuando exista máscara de lesión, **cuantificar** el solapamiento
entre el foco del modelo y la lesión real, en lugar de quedarse en la inspección visual.

## 5. Trabajos cross-dataset en ecografía mamaria

La línea más cercana al proyecto es la de estudios que entrenan y evalúan **entre datasets** de
ecografía mamaria. Trabajos multi-dataset recientes combinan BUSI, BUS-BRA, BUS-UCLM y QAMEBI con
validación cruzada y reportan un hallazgo consistente y muy relevante para nosotros: la
**segmentación** se mantiene relativamente robusta entre datasets, mientras que la **clasificación**
es mucho más sensible a las diferencias de dominio, lo que motiva explícitamente el uso de
adaptación de dominio (ResearchGate 401618842; arXiv 2509.05004, "Interpretable Deep Transfer
Learning for Breast Ultrasound Cancer Detection: A Multi-Dataset Study"). Este último, además,
combina transfer learning con interpretabilidad, muy en línea con el enfoque propuesto.

## 6. Brecha de investigación y aporte del proyecto

De la literatura se desprende que: (a) el *domain shift* entre poblaciones es grande y está
cuantificado en otras modalidades, pero **falta una caracterización sistemática y reproducible del
par BUSI↔BUS-BRA** en clasificación benigno/maligno; (b) existen métodos de DA prometedores, pero
**rara vez se comparan de forma controlada** (misma arquitectura, mismas particiones, mismas
métricas con intervalos de confianza) sobre este par concreto; y (c) la explicabilidad se usa casi
siempre de forma cualitativa intra-dominio, sin analizar **si las explicaciones se mantienen al
cruzar de población**.

El aporte de este proyecto es, por tanto, un **estudio controlado y reproducible** que: cuantifica
la caída por *domain shift* en ambas direcciones con intervalos de confianza por bootstrap; compara
en igualdad de condiciones fine-tuning, DANN y self-training; y evalúa la **consistencia de las
explicaciones (Grad-CAM/SHAP) entre dominios**, prestando atención explícita a la deduplicación de
BUSI para no contaminar las conclusiones.

---

## Referencias

1. Al-Dhabyani W., Gomaa M., Khaled H., Fahmy A. *Dataset of breast ultrasound images.* Data in
   Brief, 28:104863, 2020. https://www.sciencedirect.com/science/article/pii/S2352340919312181
2. Pawłowska A. et al. *Letter to the Editor re: Dataset of breast ultrasound images* (irregularidades
   y duplicados en BUSI). PMC10293973. https://pmc.ncbi.nlm.nih.gov/articles/PMC10293973/
3. Gómez-Flores W. et al. *BUS-BRA: A breast ultrasound dataset for assessing computer-aided diagnosis
   systems.* Medical Physics, 2024. https://aapm.onlinelibrary.wiley.com/doi/abs/10.1002/mp.16812 ·
   Datos: https://zenodo.org/records/8231412
4. Yap M.H. et al. *Automated breast ultrasound lesions detection using CNNs.* IEEE J. Biomedical and
   Health Informatics, 2018. (Dataset UDIAT) https://pmc.ncbi.nlm.nih.gov/articles/PMC6177528/
5. Yoon J.S., Oh K., Shin Y., Mazurowski M.A., Suk H.-I. *Domain Generalization for Medical Image
   Analysis: A Review.* Proceedings of the IEEE, 2024. arXiv:2310.08598. https://arxiv.org/abs/2310.08598
6. Musa A., Prasad R., Onwualu P., Hernandez M. *A Systematic Review of Cross-Population Shifts in
   Medical Imaging Analysis with Deep Learning.* Big Data Cogn. Comput., 10(3):76, 2026.
   https://www.mdpi.com/2504-2289/10/3/76
7. *Addressing cross-population domain shift in chest X-ray classification through supervised
   adversarial domain adaptation.* Scientific Reports, 2025.
   https://www.nature.com/articles/s41598-025-95390-3
8. *Domain generalization in deep learning-based mass detection in mammography: a multi-center study.*
   arXiv:2201.11620. https://arxiv.org/pdf/2201.11620
9. Zhang L. et al. *Generalizing Deep Learning for Medical Image Segmentation to Unseen Domains via
   Deep Stacked Transformation (BigAug).* IEEE TMI. https://pmc.ncbi.nlm.nih.gov/articles/PMC7393676/
10. Ganin Y. et al. *Domain-Adversarial Training of Neural Networks (DANN).* JMLR, 2016.
    https://arxiv.org/abs/1505.07818
11. Selvaraju R.R. et al. *Grad-CAM: Visual Explanations from Deep Networks via Gradient-based
    Localization.* ICCV, 2017. https://arxiv.org/abs/1610.02391
12. Lundberg S.M., Lee S.-I. *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS,
    2017. https://arxiv.org/abs/1705.07874
13. *Explainable AI in breast cancer ultrasound imaging: current developments and challenges.*
    Frontiers in Digital Health, 2026.
    https://www.frontiersin.org/journals/digital-health/articles/10.3389/fdgth.2026.1846763/full
14. *Breast Cancer Diagnosis: A Comprehensive Exploration of Explainable AI (XAI) Techniques.*
    arXiv:2406.00532. https://arxiv.org/html/2406.00532v1
15. *Interpretable Deep Transfer Learning for Breast Ultrasound Cancer Detection: A Multi-Dataset
    Study.* arXiv:2509.05004. https://arxiv.org/html/2509.05004v1
16. *Deep Learning for Breast Lesion Classification on Ultrasound Images: A Novel Multi-Dataset
    Approach.* ResearchGate 401618842, 2025.
17. Chen et al. *ADAptation: Reconstruction-based Unsupervised Active Learning for Breast Ultrasound
    Diagnosis.* arXiv:2507.00474, 2025. https://arxiv.org/abs/2507.00474v1
18. *PCA for Enhanced Cross-Dataset Generalizability in Breast Ultrasound Tumor Segmentation.*
    arXiv:2505.23587. https://arxiv.org/html/2505.23587v1
19. *BUS-UCLM: Breast ultrasound lesion segmentation dataset.* Nature Scientific Data, 2025.
    https://www.nature.com/articles/s41597-025-04562-3
20. He K. et al. *Deep Residual Learning for Image Recognition (ResNet).* CVPR, 2016.
    https://arxiv.org/abs/1512.03385
21. Tan M., Le Q. *EfficientNet: Rethinking Model Scaling for CNNs.* ICML, 2019.
    https://arxiv.org/abs/1905.11946

> Nota sobre fechas: algunas referencias (6, 13, 15, 17, 19) son de 2025–2026 y se recuperaron por
> búsqueda web; verifica los datos de publicación definitivos al citarlas en el artículo.
