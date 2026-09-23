# Mapeos públicos de BUSI

Estos CSV documentan la selección de imágenes de un conjunto público. Contienen clases, identificadores de imagen, nombres de archivo, decisiones de curación, rutas relativas y hashes; **no contienen píxeles, máscaras ni manifiestos de pacientes BUS-BRA**. Los identificadores de imagen y los grupos de imágenes relacionadas **no son identificadores de paciente**.

| Archivo | Contenido |
|---|---|
| `mapping_curated_BUSI.csv` | Mapeo oficial de Curated BUSI v1 (450 filas; 386 imágenes benignas/malignas usadas en la tarea binaria). |
| `dataset_comment_list.csv` | Lista de 587 observaciones y grupos publicada para auditar el BUSI original. |
| `busi_pawlowska_sensitivity_audit.csv` | Auditoría de 780 imágenes fuente, con decisiones de elegibilidad y hashes de archivos; permite verificar grupos relacionados y una cohorte de sensibilidad, no sustituye los archivos originales. |

Procedencia y atribución: Carlos Aumente-Maestro, Jorge Díez y Beatriz Remeseiro, [Curated BUSI v1 (Zenodo)](https://doi.org/10.5281/zenodo.19047974); Anna Pawłowska, Piotr Karwat y Norbert Żołek, [archivos suplementarios de la auditoría BUSI (Mendeley Data)](https://doi.org/10.17632/k8t3gnx9h6.1). Ambos registros indican [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). `busi_pawlowska_sensitivity_audit.csv` es una **tabla derivada y modificada** para este proyecto: cruza los mapeos públicos, registra decisiones de elegibilidad y añade hashes de archivos; no es un archivo original de esos autores. Los CSV se incluyen para verificar decisiones reproducibles; la licencia y condiciones de las imágenes originales deben comprobarse por separado antes de descargarlas o usarlas.

`scripts/10_curate_busi.py` requiere imágenes BUSI locales y puede generar otros manifiestos en este directorio. Esos archivos generados **no están aprobados automáticamente para publicación**: revisar columnas, rutas, metadatos y licencia antes de incorporarlos a Git. Los manifiestos que relacionan imágenes con pacientes o particiones permanecen siempre fuera del repositorio.
