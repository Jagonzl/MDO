# MDO: clasificación del marcado diferencial de objeto en ESLORA y PRESEEA

Clasificador BERT en español que decide si un objeto directo lleva marcado diferencial (`con_MDO`, p. ej. «conocí
**a** mi mujer») o no (`sin_MDO`). Se entrena con la muestra anotada a mano de ESLORA y PRESEEA.

**Modelo final**: [johnatanebonilla/mdo-beto-cased](https://huggingface.co/johnatanebonilla/mdo-beto-cased), en
Hugging Face. Es privado: para usarlo hay que iniciar sesión con una cuenta que tenga acceso.

Esta rama contiene cuatro pasos:

1. La comparación de 5 modelos, entrenados en GPU en el clúster de la HU.
2. La segmentación de las oraciones con varios objetos directos y el reentrenamiento del mejor modelo con ellas.
3. La clasificación MDO de todos los objetos de PRESEEA y ESLORA.
4. El cálculo de cuántas oraciones faltan, por clase de animacidad y de definitud, para que la muestra sea
   representativa.

## Qué abrir primero

| Archivo | Qué contiene |
|---|---|
| `resultados/clasificacion/PRESEEA_mdo.xlsx` | Clasificación MDO de los 9 266 objetos de PRESEEA |
| `resultados/clasificacion/ESLORA_mdo.xlsx` | Clasificación MDO de los 48 422 objetos de ESLORA |
| `resultados/original/resumen.md` | Tabla de resultados de los 5 modelos |
| `resultados/representatividad.csv` | Cuántas oraciones faltan por clase de animacidad y de definitud (punto 4) |
| `dataset/segmentacion/muestra/divididas_gemma_v3.xlsx` | Las oraciones segmentadas de la muestra, con la oración completa al lado |

Los Excel de clasificación tienen tres columnas:

- **texto**: el trozo de oración del objeto, con el verbo entre `<< >>` y el objeto entre `[[ ]]`, seguido de
  `| verbo: … ; objeto: …`. Ejemplo: `a mi mujer [[la]] <<conocí,>> | verbo: conocí ; objeto: la`.
- **mdo**: `con_MDO` o `sin_MDO`.
- **seguridad_%**: la probabilidad (0–100) que el modelo da a su respuesta. No está calibrada; sirve para revisar
  primero las más bajas. Si dice **`anotado`**, la fila ya estaba en la muestra anotada a mano: no se clasificó, y
  `mdo` es la anotación manual.

Para volver a la fila del Excel original, usa `resultados/clasificacion/<CORPUS>_mdo_claves.csv`. Tiene las mismas
filas en el mismo orden, con estas columnas:

- `sent_id`, `verbo_id`, `objeto_id`, `Verbo`, `Objeto`;
- `origen`: `modelo` o `anotado`;
- `mdo` y `seguridad_%`;
- `segmentada` y `metodo` (explicados en el punto 2);
- `dom_marcado`: lo que marcó el pipeline, para comparar.

## 1. Comparación de modelos (`resultados/original/`)

- **Datos**: los dos Excel de la muestra (`dataset/raw/muestra/`), repartidos 80/10/10 en train/dev/test
  (`dataset/processed/original/`). Las oraciones repetidas se eliminan; el punto 2 explica por qué.
- **Búsqueda de hiperparámetros**: para cada modelo se prueban learning rate, batch y número de épocas con el 40 % del
  train, y se elige la combinación con mejor resultado en dev.
- **Entrenamiento final**: con todo el train y evaluación en test, 3 veces con semillas distintas (42, 43, 44).

F1 macro en test (media de las 3 semillas; 1 = todo acertado, cuenta igual cada clase):

| Modelo | F1 macro |
|---|---|
| BETO cased (`dccuchile/bert-base-spanish-wwm-cased`) | **0,996** |
| MrBERT-es (`BSC-LT/MrBERT-es`) | 0,995 |
| mBERT (`google-bert/bert-base-multilingual-cased`) | 0,991 |
| BETO uncased (`dccuchile/bert-base-spanish-wwm-uncased`) | 0,976 |
| MrBERT (`BSC-LT/MrBERT`) | 0,933 |
| Línea base: regla de la «a» | 0,946 |
| Línea base: tf-idf + regresión logística | 0,815 |

El mejor modelo es BETO cased. La tabla completa (hiperparámetros, desviación, GPU) está en
`resultados/original/resumen.md`. Cada carpeta `final_<modelo>/` tiene `metricas.json` y `predicciones_test.csv`
(frase, clase real y clase predicha en test) del modelo guardado.

## 2. Segmentación y modelo reentrenado (`dataset/segmentacion/`, `resultados/segmentado/`)

**El problema.** En los Excel de la muestra, una oración larga con varios objetos directos aparece una vez por objeto y
siempre con el mismo texto. El modelo no puede saber a qué objeto se refiere cada fila, y en el punto 1 se
eliminaban por repetidas: 174 filas. PRESEEA y ESLORA tienen el mismo problema.

**La solución.** En esas filas, `Texto oración` pasa a ser solo el trozo de la oración que contiene su verbo y su
objeto. Ninguna otra celda cambia; se comprobó celda a celda.

1. `00a`: localiza en la oración el verbo y el objeto de cada fila. Si el mismo verbo aparece varias veces, se respeta
   el orden de las filas.
2. `00b`: Gemma 4 (31B), un modelo de lenguaje instalado en el clúster, decide dónde cortar cada trozo. Solo devuelve
   números de palabra; el texto del trozo se copia tal cual de la oración original. Las instrucciones que recibe están
   en `prompts/segmentar_mdo_v3.txt` (versión final, tras la revisión).
3. `00c`: comprueba que cada trozo contiene el verbo y el objeto de su fila. Si la respuesta de Gemma no pasa la
   comprobación, el trozo se corta con una regla fija: hasta 10 palabras a cada lado del verbo y el objeto, sin entrar
   en el objeto vecino.

**Revisión.** En la muestra se segmentaron 243 filas de 112 oraciones.

- La revisión manual está en `dataset/segmentacion/muestra/divididas_revisado.xlsx`.
- Su inventario está en `inventario_revision.csv`.
- De las 9 observaciones del revisor, 8 vienen de la anotación original:
  - animacidad: filas 364 y 805;
  - verbo que no rige el objeto: filas 23 y 692;
  - no es CD: filas 565, 672 y 719;
  - *haber* existencial: fila 687.
- La otra era un error de segmentación (fila 808, una repetición) y está corregida.

**Dataset y modelo reentrenado.**

- **Dataset**: `dataset/processed/segmentado/mdo/`, con 2 005 filas (antes 1 834).
  - Se recuperan las 174 filas eliminadas, menos las 3 que el revisor marcó como «no es CD». Estas 3 están en
    `eliminadas.csv`.
  - El reparto se hace por oración original, así que los trozos de una misma oración nunca quedan repartidos entre
    train y test.
- **Modelo**: BETO cased, con lr 5e-5, batch 16 y 2 épocas.
- **Test**: F1 macro de 0,998 ± 0,003 (3 semillas), con las 35 filas segmentadas del test bien clasificadas.
  - Líneas base en el mismo test: regla de la «a» 0,961; tf-idf 0,803.
- **Dónde está el modelo**: en Hugging Face,
  [johnatanebonilla/mdo-beto-cased](https://huggingface.co/johnatanebonilla/mdo-beto-cased) (privado), con una ficha
  que explica cómo usarlo. No está en el repositorio porque pesa 420 MB.

### Archivos de `dataset/segmentacion/<corpus>/` (muestra, preseea, eslora)

| Archivo | Qué es |
|---|---|
| `completo_gemma_v3.xlsx` | Todas las filas, con las mismas celdas que el original; en las oraciones con varios objetos, `Texto oración` es el trozo. Añade `clase_mdo`, `archivo`, `fila_excel`, `segmentada` y `metodo`. |
| `divididas_gemma_v3.xlsx` | Solo las filas segmentadas, con `Texto completo (referencia)` y las columnas de revisión (abajo). |
| `respuestas_gemma_v3.jsonl` | Respuesta de Gemma para cada oración, tal cual. Permite rehacer los xlsx sin volver a correr Gemma. |

Columnas de revisión de `divididas_*`:

- **`metodo`**: cómo se obtuvo el trozo.
  - `gemma`: el corte de Gemma pasó la comprobación.
  - `regla`: se cortó con la regla fija.
  - `regla (centrada en el objeto)`: el verbo anotado está a más de 15 palabras (anotación dudosa), así que se corta
    alrededor del objeto.
  - `regla (forma sin tilde / clítico pegado)`: el objeto es un clítico unido al verbo («verlo», «haciéndolo»,
    «fíjate»).
  - `regla (segmento de Gemma demasiado largo)`: Gemma propuso más de 40 palabras.
  - `sin segmentar`: el verbo o el objeto no aparece en la oración; se deja la oración completa.
- **`aviso`**: comprobaciones automáticas.
  - `verbo y objeto a N palabras (revisar anotación)`.
  - `contiene otro objetivo`: el trozo incluye el objeto de otra fila.
  - `posición corregida`: se tomó otra aparición de la palabra.
  - `haber existencial`.
  - `verbo/objeto no encontrado`.
- **`alerta_gemma`** y **`nota_gemma`**: lo que Gemma señala para revisar. No cambian la anotación.
  - `no_es_cd`: lo anotado como objeto no es CD de ese verbo.
  - `verbo_no_rige`: el objeto depende de otro verbo; se indica cuál.
  - `repeticion`: repetición o reformulación del hablante.
- **Posiciones** (`pos_verbo`, `pos_objeto`, `inicio_segmento`, `fin_segmento`): número de palabra en la oración
  completa, contando desde 0.

## 3. Clasificación de PRESEEA y ESLORA (`resultados/clasificacion/`)

- **Entrada**: la hoja `objetos` de los Excel del pipeline.
  - PRESEEA: `dataset/raw/preseea/`, que son los lotes 001–003 y `resumen_global`.
  - ESLORA: `dataset/raw/eslora/`, los lotes 001–018.
- **Proceso**: primero se segmentan las oraciones con varios objetos, como en el punto 2, y después las clasifica el
  modelo reentrenado.

| | PRESEEA | ESLORA |
|---|---|---|
| Objetos | 9 266 | 48 422 |
| Ya anotados a mano (`anotado`) | 392 | 1 637 |
| Clasificados por el modelo | 8 874 | 46 785 |
| … de ellos `con_MDO` | 385 | 1 861 |
| Filas segmentadas | 7 408 | 29 535 |
| Seguridad < 70 % | 35 | 164 |
| Coincidencia con `dom_marcado` del pipeline (sin clíticos) | 97,6 % | 97,9 % |

**Cautelas**

- **Clíticos** (`lo`, `la`, `te`, `me`, `nos`…): son más o menos el 20 % de los objetos y casi no aparecen en la
  muestra de entrenamiento. Sobre ellos el modelo es menos fiable, y marca `con_MDO` 151 en PRESEEA y 782 en ESLORA.
  Se dejó la salida del modelo tal cual.
- **Errores del pipeline automático.** A veces el «objeto» no es un CD («a mí me gusta», «no hay nadie»). En las
  filas segmentadas, Gemma los señala con `no_es_cd`: 1 739 en PRESEEA y 7 219 en ESLORA.
- **Desacuerdos con `dom_marcado`.** Cuando el pipeline marca y el modelo dice `sin_MDO`, casi siempre es una «a» de
  régimen («jugar al fútbol», «dedicarse a la docencia»), y el modelo acierta.
- **Sin segmentar**: 3 filas de PRESEEA y 18 de ESLORA, porque su verbo o su objeto no aparece en la oración (error
  del pipeline). Se clasifican con la oración completa.

## 4. Qué falta para que la muestra sea representativa (`resultados/representatividad.csv`)

En la muestra, el MDO casi coincide con la animacidad:

- **con_MDO** es humano en 966 de 1 007 filas.
- **sin_MDO** es inanimado en 918 de 998 filas.

Los casos que contradicen ese patrón (con_MDO inanimado, sin_MDO humano, animados) son muy pocos, y con tan pocos el
modelo puede aprender «humano = con_MDO» en vez del marcado. Lo mismo pasa, en menor medida, con «Por determinar».

La tabla cuenta, dentro de cada clase de MDO, cuántas oraciones hay de cada clase y cuántas faltan para llegar a un
mínimo de 100 o de 200 por combinación. Con el reparto 80/10/10 quedan unas 10 o 20 en test.

| MDO | Clase | En la muestra | Faltan (mín. 100) | Faltan (mín. 200) | Candidatos en el corpus* |
|---|---|---|---|---|---|
| con_MDO | humano | 966 | 0 | 0 | 359 |
| con_MDO | inanimado | 27 | **73** | **173** | 560 |
| con_MDO | animado | 14 | **86** | **186** | 4 |
| sin_MDO | humano | 75 | **25** | **125** | 1 252 |
| sin_MDO | inanimado | 918 | 0 | 0 | 28 523 |
| sin_MDO | animado | 5 | **95** | **195** | 101 |
| con_MDO | Definido | 644 | 0 | 0 | 242 |
| con_MDO | Indefinido | 286 | 0 | 0 | 70 |
| con_MDO | Por determinar | 77 | **23** | **123** | — |
| sin_MDO | Definido | 332 | 0 | 0 | 6 498 |
| sin_MDO | Indefinido | 622 | 0 | 0 | 5 676 |
| sin_MDO | Por determinar | 44 | **56** | **156** | — |

\* Objetos de PRESEEA y ESLORA que no están en la muestra y no son clíticos. Su clase MDO es la del clasificador; su
animacidad y definitud son las del pipeline, sin revisar. Sirven para saber dónde buscar, no como etiqueta.

- **Cuántas oraciones nuevas hacen falta.** Para el mínimo de 100 faltan 279 por animacidad y 79 por definitud; para
  el de 200, 679 y 279. Cada oración nueva cuenta a la vez para las dos (por ejemplo, un animado «Por determinar»), así
  que hacen falta entre **279 y 358** oraciones nuevas para el mínimo de 100, y entre **679 y 958** para el de 200.
- **Animados con MDO.** Son la clase más difícil de completar: el pipeline solo encuentra 4 candidatos. Reconoce los
  animales con una lista de lemas, así que habrá más, pero hay que buscarlos a mano (perro, gato, caballo…).
- **con_MDO inanimado.** Los 560 candidatos son lo que el clasificador marcó como con_MDO sobre objetos que el pipeline
  da por inanimados. Muchos serán errores de alguno de los dos, así que hay que revisarlos antes de anotarlos.
- **«Por determinar».** No existe en el pipeline; hay que buscarlo en la anotación manual.
- **Combinaciones de los dos ejes.** Hay muy pocas de sin_MDO humano definido (14) y de con_MDO inanimado indefinido
  (3). El cruce completo de las tres variables lo da `scripts/08_representatividad.py`.

## Estructura

```
dataset/
  raw/muestra/                 Excel de la muestra anotada (con_MDO y sin_MDO)
  raw/preseea/, raw/eslora/    Excel del pipeline (hoja «objetos»)
  processed/original/          train/dev/test (punto 1)
  processed/segmentado/        train/dev/test con las oraciones segmentadas (punto 2)
  segmentacion/                muestra/, preseea/, eslora/: segmentación de cada corpus
prompts/                       instrucciones para Gemma
scripts/
  00a_preparar_segmentos.py    localiza verbo y objeto en las oraciones con varios objetos
  00b_segmentar_gemma.py       Gemma propone los cortes (GPU)
  00c_construir_xlsx.py        valida los cortes y escribe completo_*.xlsx y divididas_*.xlsx
  01_build_dataset.py          particiones train/dev/test (--version original | segmentado)
  02_baseline.py               líneas base (regla de la «a», tf-idf)
  03_busqueda.py               búsqueda de hiperparámetros (GPU)
  04_final.py                  entrenamiento final con 3 semillas y evaluación en test (GPU)
  05_subir_hf.py               sube el modelo final a Hugging Face
  06_informe.py                tabla resumen de resultados
  07_clasificar_mdo.py         clasifica MDO un corpus segmentado (GPU recomendada)
  08_representatividad.py      cuántas oraciones faltan por clase de animacidad y definitud
  common.py, entrenamiento.py  funciones comunes
slurm/                         trabajos para el clúster
resultados/
  original/                    punto 1
  segmentado/                  punto 2
  clasificacion/               punto 3
  representatividad.csv        punto 4
```

## Cómo reproducirlo

En el clúster de la HU, cada sesión empieza con `source slurm/env.sh`, que define `$PY` y entra en la carpeta del
repositorio. Lo marcado como GPU se lanza con `sbatch`; el resto corre en el login o en un PC. La variable
`MDO_VERSION=segmentado` hace que 02–06 usen el dataset segmentado; por defecto usan el original.

```bash
# 1. Comparación de modelos
$PY scripts/01_build_dataset.py --version original
$PY scripts/02_baseline.py
J=$(sbatch --parsable slurm/03_busqueda.sbatch)          # GPU, 5 trabajos: uno por modelo
sbatch --dependency=aftercorr:$J slurm/04_final.sbatch   # GPU
$PY scripts/06_informe.py

# 2. Segmentación de la muestra y reentreno de BETO cased
$PY scripts/00a_preparar_segmentos.py --corpus muestra
sbatch slurm/00_segmentar_gemma.sbatch --corpus muestra  # GPU
$PY scripts/00c_construir_xlsx.py --corpus muestra
$PY scripts/01_build_dataset.py --version segmentado
sbatch slurm/reentreno_segmentado.sbatch                 # GPU
HF_TOKEN=... MDO_VERSION=segmentado $PY scripts/05_subir_hf.py --model beto_cased   # -> mdo-beto-cased

# 3. PRESEEA (ESLORA igual, con --corpus eslora y --time=04:00:00 en la segmentación)
$PY scripts/00a_preparar_segmentos.py --corpus preseea
sbatch slurm/00_segmentar_gemma.sbatch --corpus preseea --max-tokens 4096   # GPU
$PY scripts/00c_construir_xlsx.py --corpus preseea
sbatch slurm/07_clasificar.sbatch --corpus preseea      # GPU

# 4. Representatividad (después de clasificar PRESEEA y ESLORA)
$PY scripts/08_representatividad.py
```

Notas:

- **Archivos intermedios.** Los de `00a` (`tareas.jsonl`, `filas.csv`) no están en el repositorio: se regeneran en
  segundos. Con las respuestas de Gemma guardadas, `00c` y `07` se pueden volver a correr sin Gemma.
- **Sin GPU.** `07` también funciona en un PC, pero más lento, y con el modelo indicado por
  `--modelo <carpeta o repo de Hugging Face>`.
- **CPU frente a GPU.** En la comprobación con PRESEEA cambió 1 etiqueta de 8 874: un caso límite, con seguridad del
  50 %.
- **Versión de scikit-learn.** La partición del dataset segmentado se agrupa por oración (`StratifiedGroupKFold`), y
  esa partición cambia entre versiones de scikit-learn. La oficial se hizo con la 1.7.2, la del clúster. La del
  dataset original sale igual con cualquier versión.
