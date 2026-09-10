# Student Analytics UA

Plataforma de analítica estudiantil para la Universidad Autónoma de Chile.
Piloto en carreras de la **Facultad de Administración y Negocios** (Providencia,
San Miguel, Talca, Temuco).

La especificación completa está en
[Student_Analytics_UA_Master_Context.md](Student_Analytics_UA_Master_Context.md)
(120 secciones). Este README cubre solo lo implementado y cómo correrlo.

---

## Estado

| Sprint | Contenido | Estado |
|---|---|---|
| 0 | Contextualización y especificación | Completo |
| **1** | **Paquete, configuración, generador sintético, tests** | **Completo** |
| **1b** | **Adaptador OULAD + baseline + cockpit Streamlit** | **Completo** |
| **1c** | **Datos abiertos Mineduc/SIES: retención real UA + piso predictivo** | **Completo** |
| **1d** | **Informe metodológico reproducible (Quarto)** | **Completo** |
| **1e** | **Trayectoria escolar previa al ingreso + ablación del piso** | **Completo** |
| **1f** | **Benchmark universitario: CNED, selectividad de admisión, titulación** | **Completo** |
| **1g** | **Serie longitudinal: matrícula 2007–2026, retención y titulación por cohorte** | **Completo** |
| 2 | Capa analítica: silver, `mart_student_course_week`, `mart_student_week` | Pendiente |
| 3 | Segmentación (≥3 alternativas) | Pendiente |
| 4 | Modelo 1 — riesgo de reprobación (≥3 algoritmos) | Pendiente |
| 5 | Producto completo: Student 360, drivers SHAP | Pendiente |
| 6 | Modelo 2 — desenganche académico | Pendiente |

**Sin accesos a Banner/Canvas todavía.** El pipeline corre sobre datos
sintéticos y sobre dos fuentes reales abiertas (OULAD y Mineduc/SIES). La
calibración del generador está anclada a la retención **medida** de la propia
UA, calculada desde datos abiertos.

---

## Estrategia de datos antes de tener accesos

Cuatro fuentes, cuatro problemas distintos:

| Fuente | Resuelve | No resuelve |
|---|---|---|
| Esquemas públicos de Canvas Data 2 y Banner ODS | Que el pipeline reciba las **columnas reales** | No trae datos |
| **OULAD** (Open University, 28.785 estudiantes reales) | ¿Funciona la predicción temprana? ¿Cuán temprano? | Sin asistencia, sin escala 1–7, sin sedes |
| **Mineduc / SIES abiertos** (matrícula 2007–2026, titulados 2007–2025, admisión 2021–2026, más escolares) | Retención real de la UA; piso predictivo nacional; benchmark y seguimiento por cohorte | No se puede unir al RUT de Banner |
| **CNED INDICES** (institucional 2005–2025) | Cuerpo docente, infraestructura, acreditación | Se une por nombre, no por código |
| **Generador sintético** (este repo) | Forma UA completa: sedes, carreras, asistencia, notas 1.0–7.0 | No valida nada por sí solo |

El objetivo de esta fase no es tener un modelo. Es tener un pipeline cuyo
contrato de entrada calce con Banner/Canvas lo bastante bien como para que la
migración a datos reales sea un cambio de configuración. Ese contrato vive en
[config/data_contracts.yml](config/data_contracts.yml), donde cada campo declara
su origen real esperado y su nivel de confianza (`confirmed` / `likely` /
`verify`).

---

## Uso

Para publicar en Streamlit Community Cloud, la entrada es `streamlit_app.py`.
Incluye los agregados públicos necesarios en `deploy/data/` y dependencias
verificadas en `requirements.txt`. Consulta la
[configuración de despliegue](deploy/README.md).

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -e ".[modeling,ui,dev]"

# 1. Datos sintéticos (forma UA)
python scripts/generate_synthetic.py                 # volumen de config (10.000)
python scripts/generate_synthetic.py --students 500  # smoke test

# 2. Datos reales de referencia (~45 MB comprimidos)
python scripts/download_oulad.py

# 2b. Bases abiertas de Mineduc/SIES
python scripts/download_mineduc.py     # escolares y socioeconómicas
python scripts/analyze_mineduc.py      # retención UA + piso + ablación
python scripts/build_findings.py       # hallazgos a CSV para el informe

# 3. Curva de poder predictivo vs anticipación, sobre cualquier fuente
python scripts/validate_lead_time.py --source synthetic --out data/results
python scripts/validate_lead_time.py --source oulad     --out data/results

# 4. Cockpit
streamlit run src/student_analytics/ui/app.py

# 5. Informe metodológico (fuentes, descriptivas, metodología)
python scripts/descriptive_stats.py
quarto render documentacion/informe_metodologico.qmd            # HTML + PDF
quarto render documentacion/informe_metodologico.qmd --to typst # solo PDF

pytest -q                                            # 143 tests
```

---

## La herramienta de visualización

### Hallazgos (vista inicial)

La app abre en **Hallazgos**, que responde en vez de dejar preguntar. Reúne en
una pantalla lo que el proyecto encontró: el argumento central, la posición real
de la UA en continuidad y selectividad, el contraste entre los dos indicadores
de titulación, y las trampas de datos que cambiaron un resultado.

**Ninguna cifra está escrita en la vista.** Todas se calculan en
[`modeling/findings.py`](src/student_analytics/modeling/findings.py) desde los
mismos parquet que alimentan el informe metodológico, que las consume vía
`scripts/build_findings.py`. Es la única forma de que el tablero y el documento
no se contradigan cuando cambie una base — y ya pasó: la cifra escrita a mano
sobre denominadores siguió ahí meses después de que el código cambiara.

Cada función devuelve `None` cuando falta su insumo en vez de fallar, así que la
vista funciona con el proyecto armado a medias y dice qué script correr.

### Benchmark UA

Desde la barra lateral. Los pares se
identifican por perfil de ingreso **sin utilizar la retención como variable de
clustering**. Se puede comparar con los cinco vecinos más cercanos, con el mismo
cluster o con un conjunto manual; la UA permanece como referencia.

```bash
python scripts/download_matricula.py   # matrícula 2007-2026 (18 GB crudo, 367 MB slim)
python scripts/download_titulados.py $(seq 2007 2025)   # titulados, ~6 MB por año
python scripts/download_paes.py        # admisión 2021-2026
python scripts/download_cned.py        # recursos institucionales (3,3 MB, opcional)
python scripts/build_retention.py      # 19 transiciones de continuidad
python scripts/build_cohorts.py        # titulación por cohorte de ingreso
python scripts/build_benchmark.py      # perfiles de las cohortes disponibles
streamlit run src/student_analytics/ui/app.py
```

Nada de esto es obligatorio de una vez: cada script detecta lo que falta y se
salta lo que no está. Con solo dos años de matrícula el benchmark funciona
igual, con una transición y sin titulación por cohorte.

### Recursos institucionales (CNED)

`scripts/download_cned.py` baja la base **INDICES Institucional 2005–2025** del
Consejo Nacional de Educación y `build_benchmark.py` la incorpora si está
presente. Aporta, por institución y sede: cuerpo docente por jornada y nivel de
grado (permite calcular JCE y % con doctorado), inmuebles y m² construidos,
laboratorios y PC para estudiantes, bibliotecas, año de creación, pertenencia al
CRUCH y años de acreditación.

Con eso, más la selectividad de admisión y la titulación, el perfil llega a
**101 variables**, de las cuales **43 son graficables** en el mapa de
posicionamiento y **39 entran a la distancia**.

Tres cosas que hay que saber de esta fuente:

1. **Los códigos de institución del CNED no son los del SIES** — la Universidad
   Autónoma es 1037 en CNED y 31 en matrícula. La unión se hace por nombre
   normalizado y calza 51 de 51 universidades; hay un test que falla si esa
   cobertura baja del 95%.
2. **Las bibliotecas están en dos hojas** (2005–2018 y 2019–2025) porque cambió
   el instrumento de medición. Se usa solo la vigente: concatenarlas produciría
   un salto en 2019 que es metodológico, no real.
3. **La columna "Tradicional" del CNED marca pertenencia al CRUCH, no
   antigüedad.** Desde la Ley 21.091 el CRUCH admite privadas posteriores a
   1981, y en 2019 entraron Diego Portales, Alberto Hurtado y Los Andes. Por eso
   la variable se llama `cruch`.

Los recursos entran como **descriptivos**: aparecen en los ejes del mapa pero no
participan en la distancia ni en los clusters, así que no cambian los pares.

### Selectividad de admisión (PAES y PDT)

`scripts/download_paes.py` baja los puntajes de admisión y `build_benchmark.py`
los cruza con la cohorte de ingreso **por MRUN**. A diferencia de los recursos,
la selectividad **sí participa en la distancia**: con ella activada, «comparable»
deja de significar solo «ofrece carreras parecidas» y pasa a significar también
«recibe estudiantes parecidos».

Tres cosas que hay que saber de esta fuente:

1. **Solo existe desde 2021.** La PSU (2004–2020) no está publicada como
   microdato abierto en ninguna sección del portal del Mineduc; hay que pedirla
   al DEMRE. Las cohortes anteriores a 2021 se comparan sin selectividad, y el
   bloque se omite repartiendo su peso entre los demás.
2. **Hay un corte de escala en 2023.** La Prueba de Transición (2021–2022) iba
   de 150 a 850 con media 500; la PAES va de 100 a 1000 con media ~610. Los
   puntajes crudos **no son comparables** entre ambos lados: un promedio
   institucional salta unos 100 puntos sin que haya cambiado nada real. Por eso
   se expone `paes_percentil_promedio`, el percentil nacional dentro de cada
   cohorte, que sí cruza el corte. Ese es el eje correcto para mirar la serie.
3. **La cobertura es la advertencia.** No todos los ingresantes rinden la prueba
   —vías especiales, extranjeros, continuidad de estudios—, así que el promedio
   describe solo a quienes sí. `paes_cobertura` acompaña siempre al indicador.

El clustering estandariza dentro de cada cohorte y las agrupa por separado, así
que el cambio de escala no contamina la selección de pares.

El agrupamiento usa cinco bloques con igual peso: tamaño de cohorte/sedes,
distribución por áreas, distribución regional, jornada/modalidad y
selectividad. Compara
K-means, mezcla gaussiana diagonal y jerárquico Ward, con 2–6 grupos, tamaño
mínimo de grupo y un **límite de concentración**: se descartan las particiones
donde un solo grupo reúne más del 50% de las universidades.

Ese límite no es cosmético. Sin él la silueta elige siempre k=2, que aísla
cuatro instituciones atípicas y deja al 92% restante —la UA incluida— en un
único grupo: silueta 0,33 frente a ~0,11 del resto, porque separar atípicos
produce cortes limpios. Con el límite la selección pasa a Ward con 5–6 grupos y
el de la UA reúne 9 universidades (18%), que es lo que hace usable el modo
«mismo cluster». La silueta mide separación, no utilidad. La aplicación presenta diagnóstico de silueta, estabilidad
entre cohortes, distancias por bloque, distribuciones originales y mapa PCA.
Los filtros del resultado no modifican los pares identificados por perfil.

La comparación muestra brechas frente a pares y frente al resto del sistema,
excluyendo a la UA de ambas referencias. La sección de retención aplica el mismo
criterio: la brecha de cada universidad se calcula contra el sistema sin ella
misma, para que una institución grande no atenúe su propia referencia. Incluye una referencia estandarizada
con la mezcla de áreas de la UA y cobertura explícita de áreas comunes. Se pueden
descargar la tabla CSV y la ficha metodológica JSON. Los perfiles ya incluyen
selectividad (en la distancia) y recursos, cuerpo docente, acreditación y
titulación (descriptivos); sigue sin incorporarse actividad de investigación.
La similitud es parcial y las brechas son descriptivas, no causales.

Configuración: `config/benchmark.yml`. Artefactos locales:
`data/results/benchmark_profiles.parquet` y `benchmark_manifest.json`.
La generación comprueba denominadores contra los agregados de retención; la
app verifica sus huellas SHA-256. Detalles en
[la metodología del benchmark](documentacion/benchmark_universitario.md).

### Sección complementaria: retención universitaria

En la barra lateral, **Sección → Retención universitaria** permite comparar las
universidades presentes en las matrículas públicas descargadas. Para preparar
los agregados (sin entrenar modelos):

```bash
python scripts/build_retention.py
streamlit run src/student_analytics/ui/app.py
```

Con la matrícula histórica completa se obtienen **19 transiciones**, de
2007→2008 a 2025→2026. La vista incluye selección de universidades, área de
conocimiento, mínimo de inscripciones, comparación entre cohortes, detalle por
sede y carrera y descarga CSV. Distingue continuidad en la misma carrera y
universidad, en la misma universidad y en cualquier institución (incluidos
IP/CFT).

Al mirar la serie completa hay que tener presente que **la continuidad de
carrera no es comparable antes de 2009**: falta `cod_carrera` en el 24,5% de la
cohorte 2007 y el 17,7% de la de 2008, y sin código la fila no puede calzar
aunque la persona haya seguido en la misma carrera. Los otros dos niveles no
dependen de ese campo. La app avisa cuando se elige ese indicador en una cohorte
con cobertura baja.

El denominador son inscripciones de pregrado con ingreso a la carrera en el año
de cohorte y MRUN disponible; una persona puede contar en más de una carrera.
Se eliminan duplicados exactos de las columnas leídas. Los registros sin MRUN se
cuentan aparte. Las tasas agregadas se ponderan por inscripciones. La referencia
nacional respeta el área y cohorte, pero no la selección de universidades ni el
mínimo de tamaño. Son cálculos descriptivos propios, no un ranking de calidad ni
predicciones semanales. El artefacto `data/results/retention_universities.parquet`
contiene únicamente agregados y se carga independientemente del cockpit.

La sección **Alerta temprana** mantiene tres vistas:

**Cockpit de alerta temprana.** La pieza central no es el AUC sino la **curva de
capacidad**: un deslizador que responde *si esta semana alcanzo a contactar al
20% de mis estudiantes, ¿a qué fracción de los que van a reprobar llego?*. Es la
métrica del benchmark ULagos y la única que se traduce en una decisión. Debajo,
la lista priorizada con las señales que la generaron.

**Estudiante 360.** Trayectoria del riesgo semana a semana por asignatura. Lo que
importa no es el nivel sino la pendiente: un riesgo alto y estable es un caso
distinto de uno que se está acelerando.

**Desempeño del modelo.** Curva de lead time, AUC, Brier y la tabla completa.

Se eligió **Streamlit** sobre Shiny for Python porque el documento maestro ya lo
especifica en el stack (§96) y para un cockpit de tablas, filtros y drill-down es
más directo. Shiny sería mejor con reactividad compleja o con un equipo que
venga de R. Para la entrega final a las direcciones de carrera conviene evaluar
**Power BI**: ya tiene autenticación institucional y gobernanza de acceso.

Las bandas de riesgo usan una paleta de estado validada para daltonismo y van
siempre con **icono + etiqueta**, nunca solo con color.

---

## Decisiones de diseño

**El generador se auto-calibra contra anclas, no contra constantes.**
`config/synthetic.yml` declara tasas objetivo (retención 84,1% — **medida** para
la UA desde datos abiertos; reprobación por asignatura 20%) y el código resuelve
sus parámetros internos por bisección para satisfacerlas. Cambiar el peso de una señal no altera la tasa
global de reprobación, que es lo que permite probar robustez sin re-tunear todo.

**Orden causal no invertible.** Perfil latente → comportamiento semanal →
agregados de cierre → riesgo latente → resultado. El resultado nunca se sortea
primero. Si se hiciera al revés, cualquier modelo obtendría un AUC excelente y no
sabríamos si es porque el pipeline funciona o porque el generador filtró la
respuesta.

**Dos targets separados: `fail_by_grade` y `fail_by_attendance`.** Si la
universidad reprueba por inasistencia, la asistencia no es un *predictor* de
reprobación: es parcialmente el *mecanismo* del target. Mezclarlos produce un
modelo que aprendió el reglamento, no el fenómeno.

**Calidad de Canvas desde el día 1.** La adopción docente es propiedad del
*curso*. En cursos de baja adopción las features Canvas se emiten como NULL
explícito, no como cero: sin esto, todo estudiante de un curso sin Canvas
aparece como desenganchado.

**Ausencia de dato ≠ cero.** Un conteo ausente es un cero real; un promedio
ausente es desconocido. Imputar 0 en `mean_grade` equivale a inventar un 1.0.

**Las variables previas al ingreso predicen débil, a propósito.** El generador
reproduce el hallazgo central del benchmark ULagos (PAES y NEM casi no predicen)
para que el pipeline lo redescubra en vez de asumirlo. Hay un test que falla si
esa correlación se vuelve fuerte.

---

## Resultados sobre datos sintéticos

Regresión logística, validación temporal (semestres 1–4 entrenan, 5–6 testean),
3.000 estudiantes, cohorte fija de 19.842 casos y prevalencia 23,4%:

| Semana de scoring | AUC | Top 10% | Top 20% | Semanas restantes |
|---|---|---|---|---|
| 3 | 0,652 | 14,8% | 28,9% | 15 |
| 5 | 0,681 | 16,5% | 30,9% | 13 |
| 8 | 0,717 | 19,2% | 36,3% | 10 |
| 12 | 0,749 | 22,3% | 41,5% | 6 |

*Top 20% = de los estudiantes que efectivamente reprobaron, qué fracción queda
dentro del 20% priorizado por el modelo esa semana. Baseline aleatorio = 20%.*

**Estas cifras no son un hallazgo sobre la UA.** Miden que el pipeline recupera
la señal que el generador inyectó, y que la dificultad del problema simulado está
en el rango que reporta la literatura. El número que importa saldrá de datos
reales.

## Resultados sobre OULAD (datos reales)

El **mismo** pipeline, sin cambiar una línea de la lógica de features ni de
evaluación — solo el adaptador de ingesta. 28.785 estudiantes reales en 32.593
inscripciones (estudiante × módulo × presentación), validación temporal.

**Cohorte fija**: 13.144 casos y prevalencia 32,9% en *todas* las semanas, así
que lo único que cambia entre filas es la información disponible.

| Semana | % del curso | AUC | Top 10% | Top 20% | Semanas restantes |
|---|---|---|---|---|---|
| 4 | 10% | 0,697 | 15,5% | 33,6% | 35 |
| 8 | 21% | 0,753 | 19,4% | 37,5% | 31 |
| 12 | 31% | 0,801 | 26,4% | 43,6% | 27 |
| 16 | 41% | 0,824 | 27,7% | 45,9% | 23 |
| 20 | 51% | 0,840 | 27,7% | 46,5% | 19 |
| 26 | 67% | 0,859 | 29,0% | 48,7% | 13 |

**OULAD no tiene asistencia** (es educación a distancia) — justamente la variable
que ULagos identificó como clave. Estos números son un **piso**, no un techo.

### Por qué cohorte fija

Excluir a quien ya se dio de baja es correcto (no es un caso a predecir), pero
recalcular esa exclusión cada semana hace que la población cambie: en modo
`rolling` el n va de 15.735 a 13.144 y la prevalencia de 44% a 33%. Entonces la
mejora del AUC podría venir de más información **o** de una población distinta.

Se midió con `--compare-population`: los deltas de AUC son ≤ 0,008 en todas las
semanas, así que la pendiente **no** venía del cambio de población. Pero la
prevalencia variable hacía las filas no comparables y el lift sí se movía (Top
20% en la semana 4: 30,3% rolling vs 33,6% fija). Por eso `fixed` es el default.

```bash
python scripts/validate_lead_time.py --source oulad --compare-population
```

Un hallazgo del baseline: hasta la semana 12 el driver dominante es
`prior_attempts` (intentos previos en el módulo), una variable de historial; desde
la semana 16 pasa a ser `late` (entregas atrasadas), una señal de comportamiento.
Es el mismo patrón de ULagos — al principio solo sirve la historia previa, y las
señales conductuales toman el relevo cuando ya hay conducta que observar.

---

## Resultados sobre datos abiertos de Mineduc / SIES

```bash
python scripts/download_mineduc.py    # bases escolares y socioeconómicas
python scripts/analyze_mineduc.py     # retención UA + piso predictivo
```

Estas bases están desagregadas **a nivel individual** y usan **MRUN**, un
identificador ficticio pero estable entre bases y años. Chile es de los pocos
países OCDE que publica esto abiertamente.

### Retención real de la UA (Matrícula 2024 → 2025, cruce por MRUN)

Calculada, no citada. Cohorte de ingreso 2024, área Administración y Comercio:

| | Misma carrera y universidad | En cualquier institución |
|---|---|---|
| **Administración y Comercio (n=680)** | **84,1%** | 92,1% |
| Toda la UA, pregrado (n=6.475) | 85,8% | 93,7% |
| Todas las universidades del país (n=153.391) | 82,0% | 92,1% |

Por sede, dentro del área del piloto:

| Sede | n | Retención |
|---|---|---|
| Providencia | 251 | 88,8% |
| Temuco | 173 | 85,5% |
| Talca | 157 | 80,9% |
| El Llano (San Miguel) | 99 | 74,7% |

La brecha de **14 puntos entre Providencia y El Llano** es un hallazgo real, no
simulado, y sugiere que el modelo debería considerar la sede explícitamente.

**El orden entre sedes no es estable de un año a otro.** Con las tres cohortes
disponibles del área la brecha va de 14,1 puntos (2024) a 10,2 (2023) y 6,9
(2025), y El Llano marca 80,7%, 74,7% y 79,6% en esos años. Con cerca de cien
inscripciones por sede, parte de ese movimiento es ruido muestral. Lo que
sostiene el argumento es la dispersión persistente, no el ranking de un año:
concluir «El Llano es la sede con problemas» desde una sola cohorte sería un
error.

### La serie larga: 19 transiciones de continuidad (2007→2008 … 2025→2026)

Con la matrícula histórica completa, la retención deja de ser una foto y pasa a
ser una serie. Como cada año son ~900 MB de CSV y la máquina de trabajo tiene
~5 GB de RAM libre, `download_matricula.py` escribe una **capa slim en parquet**
—367 MB para los 20 años— que es lo que consume todo el análisis longitudinal.

Son dos parquet por año. El segundo, `seguimiento_YYYY`, no está filtrado por
tipo de institución, y esa decisión no es cosmética: la continuidad «en el
sistema» significa seguir en educación superior, así que si el año siguiente se
buscara solo entre universidades, quien se cambia a un CFT o a un IP se contaría
como deserción.

**Una trampa que había que resolver:** `cod_carrera` falta en el **24,5% de la
cohorte 2007** y el **17,7% de la de 2008**, y bajo el 2% desde 2009. Sin
código, la fila no puede calzar a nivel de carrera aunque la persona haya
seguido en la misma, así que la continuidad *de carrera* aparece en 53,9% en vez
de ~72% sin que nadie hubiera desertado. Los niveles de universidad (76,5%) y de
sistema (83,8%) no dependen de ese campo y sí son comparables en esos años. El
script avisa solo y la app también.

### Titulación por cohorte de ingreso

Cruzando la cohorte de ingreso contra las bases de titulados por MRUN se
responde lo que los indicadores transversales no pueden: de quienes **entraron**,
qué proporción llegó a titularse. Se mide en tres niveles anidados —de su
carrera, en su universidad, en alguna universidad—; la brecha entre el segundo y
el tercero es traslado, y sin ese nivel todo traslado se contaría como fracaso.

El problema central es la **censura por derecha**, y tiene dos niveles:

1. **Por estudiante.** Cada ingresante entra al denominador solo si su propio
   horizonte —duración nominal más la holgura— cabe en los datos disponibles.
2. **Por institución.** Corregir el tiempo no corrige la composición: cuando
   quedan pocos años de seguimiento caen primero las carreras largas, así que
   una universidad con mucha ingeniería o medicina queda descrita solo por sus
   carreras cortas y su tasa sube sin haber titulado mejor.

Ese segundo filtro no es teórico. Sin él la UA aparecía **primera del sistema**
en la cohorte 2018 con 70,4% y percentil 98; con el piso al 90% de cobertura esa
cohorte se anula y la última comparable pasa a ser 2017.

### El piso predictivo de las variables previas al ingreso

Cruce PAES 2024 → Matrícula 2024 → Matrícula 2025 por MRUN. Predice no
continuar en la misma carrera, usando **solo** NEM, ranking, notas de enseñanza
media, puntajes PAES, dependencia del colegio, sexo y año de egreso:

| Población | n | Prevalencia | AUC | Top 20% |
|---|---|---|---|---|
| Todo el sistema | 184.279 | 18,3% | 0,623 | 31,5% (1,57×) |
| **Solo universidades** | 135.050 | 16,4% | **0,617** | 31,6% (1,58×) |

### ¿Y si agregamos todo el expediente escolar?

Se incorporaron tres bases más de Mineduc: **asistencia mensual de enseñanza
media** (marzo–diciembre, individual), **condición SEP** (prioritario/preferente)
y **asignaciones de becas y créditos** (quintil, decil DFE, gratuidad, FSCU).
Ablación con validación temporal, solo universidades:

| Modelo | AUC | Δ vs base | Top 20% |
|---|---|---|---|
| A. Ficha de admisión (base) | 0,617 | — | 31,6% |
| B. + asistencia de enseñanza media | 0,622 | +0,005 | 32,2% |
| C. + socioeconómico | 0,631 | +0,014 | 33,2% |
| **D. + ambos** | **0,636** | **+0,019** | 33,8% |
| E. Solo asistencia de enseñanza media | 0,543 | −0,073 | 26,9% |

Tres lecturas:

1. **La asistencia escolar casi no transfiere.** Sola alcanza AUC 0,543, apenas
   sobre el azar. No contradice a ULagos: la asistencia importa *dentro* de la
   institución y del período en curso, no como rasgo que cruza una transición
   institucional.
2. **Las variables socioeconómicas aportan +0,014.** Ese "poco" resuelve el
   dilema ético empíricamente: excluirlas del scoring cuesta casi nada.
3. **Todo junto llega a 0,636** — sigue por debajo de lo que un baseline
   conductual alcanza en la semana 4.

Con **validación temporal** (cohorte 2023 entrena, n=129.512 → cohorte 2024
testea, n=135.050) el AUC de universidades es **0,615**: idéntico al del split
aleatorio. El piso es estable y se sostiene sobre cohortes futuras.

**Esto replica el hallazgo de ULagos con 135.000 estudiantes en vez de dos
cohortes**, y es el argumento central del proyecto:

> Todo lo que el Estado de Chile sabe de un estudiante antes de que pise la
> universidad —NEM, ranking, cinco pruebas PAES, dependencia del colegio,
> asistencia mensual de 4° medio, decil de ingreso oficial, condición de
> prioritario, tipo de financiamiento— alcanza **AUC 0,636**.
>
> El baseline conductual sobre OULAD llega a **0,697 en la semana 4**, con el
> 10% del curso transcurrido y **sin asistencia**.

Ese es el caso para pedir acceso a Banner y Canvas: el valor no está en lo que
ya se sabe del estudiante al matricularse.

*Caveat: solo el 53% de los ingresantes 2024 aparece en PAES 2024 (los ingresos
a IP/CFT y por vías alternativas no la rinden), así que la población analizada
son ingresantes con PAES — el perfil relevante para la UA, pero no el sistema
completo.*

---

## Documentación

[`documentacion/informe_metodologico.qmd`](documentacion/informe_metodologico.qmd)
— informe reproducible con todas las fuentes y sus licencias, los insumos
documentales, estadísticas descriptivas por fuente, la metodología completa
(grano, anti-leakage, validación temporal, cohorte fija, calibración por anclas)
y las limitaciones. Ninguna cifra está escrita a mano: todas salen de
`documentacion/datos/`, que `scripts/descriptive_stats.py` regenera.

Se genera en **dos formatos** desde el mismo fuente:

| Salida | Cómo | Identidad visual |
|---|---|---|
| `informe_metodologico.html` | tema `cosmo` + `estilos.css` | Franja con logo sobre el título, logo embebido en base64 |
| `informe_metodologico.pdf` | **Typst** (no requiere LaTeX) | Encabezado corrido con logo en las 14 páginas |

El PDF sale por Typst, que viene incorporado en Quarto: no hace falta instalar
TinyTeX ni una distribución LaTeX. Los colores de marca (`#E2211C` del escudo,
`#3D3935` del texto) se muestrearon del propio logo y se verificó su contraste
sobre blanco — 4,70:1 y 11,44:1, ambos WCAG AA.

Complementan el informe:

- [`documentacion/benchmark_universitario.md`](documentacion/benchmark_universitario.md)
  — metodología del benchmark: bloques, distancia, clusters y sus límites.
- `documentacion/datos/inventario_perfil_benchmark.csv` — **una fila por
  variable del perfil**, con su origen, si entra al clustering, si es graficable
  y las advertencias que no se deducen del dato (escalas no comparables,
  denominadores, centinelas). Lo regenera `scripts/build_inventory.py` y **no se
  edita a mano**: un inventario desactualizado es peor que no tenerlo, porque se
  lee como si fuera cierto.
- `documentacion/datos/inventario_campos_fuentes_benchmark.csv` — cada campo
  crudo de cada base descargada.

---

## Preguntas bloqueantes

Están en `open_questions` de [config/data_contracts.yml](config/data_contracts.yml).
Las tres que cambian el diseño:

1. **¿Usa la UA el módulo Banner Attendance Tracking?** Es opcional. Sin él solo
   existe `SFRSTCR_LAST_ATTEND`, que no da grano semanal — y se cae medio Caso 2.
2. **¿Existen notas parciales fechadas**, o solo la nota final del período?
3. **¿Contempla el reglamento reprobación por inasistencia**, y con qué umbral?

---

## Marco legal

La **Ley 21.719** de Protección de Datos Personales entra en plena vigencia el
**1 de diciembre de 2026**, dentro del horizonte de este proyecto. Consagra el
derecho a no ser objeto de decisiones basadas exclusivamente en tratamiento
automatizado, con garantías de explicación, intervención humana y revisión.

Consecuencias de diseño, no de cumplimiento: el humano en el loop deja de ser
buena práctica y pasa a ser requisito; la explicabilidad por caso deja de ser
opcional. Conviene levantar base de licitud y evaluación de impacto con la
Dirección Jurídica **antes** de que el modelo toque datos reales.

---

## Estructura

```
config/          settings.yml, synthetic.yml, benchmark.yml, data_contracts.yml
src/student_analytics/
    config.py               carga y validacion de YAML
    logging_setup.py
    synthetic/
        profiles.py         7 perfiles latentes de estudiante
        generator.py        generador reproducible y auto-calibrado
    ingestion/
        oulad.py            OULAD -> esquema canonico
        retention.py        continuidad en tres niveles anidados
        cohortes.py         titulacion por cohorte de ingreso (longitudinal)
        titulados.py        titulacion de la promocion que egresa (transversal)
        paes.py             selectividad de admision; PDT y PAES
        cned.py             recursos institucionales y cuerpo docente
    features/
        builder.py          features por semana, agnostico a la fuente
    modeling/
        lead_time.py        poder predictivo vs anticipacion
        benchmark.py        bloques, distancia y clusters de pares
        findings.py         hallazgos, calculados una sola vez
    ui/
        app.py              cockpit Streamlit
        hallazgos.py        vista de hallazgos (la inicial)
scripts/
    generate_synthetic.py
    download_oulad.py
    download_mineduc.py     bases escolares y socioeconomicas
    download_matricula.py   matricula 2007-2026 + capa slim en parquet
    download_titulados.py   titulados 2007-2025
    download_paes.py        admision 2021-2026 (la PSU no es abierta)
    download_cned.py        INDICES institucional del CNED
    analyze_mineduc.py      retencion real UA + piso predictivo nacional
    build_retention.py      19 transiciones de continuidad
    build_cohorts.py        seguimiento longitudinal por MRUN
    build_benchmark.py      perfiles universidad-cohorte
    build_inventory.py      inventario de variables (se regenera, no se edita)
    build_findings.py       hallazgos a CSV para el informe
    validate_lead_time.py   --source synthetic | oulad
tests/           tests: generador, leakage, OULAD, UI, benchmark, cohortes
docs/            documentos base del proyecto (postulación, ULagos, títulos)
documentacion/
    informe_metodologico.qmd   fuentes, descriptivas, metodología, resultados
    datos/                     artefactos que el informe cita (CSV pequeños)
```

El feature builder y el evaluador **no saben de que fuente vienen los datos**.
Corren igual sobre el generador sintetico y sobre OULAD. Esa es la propiedad que
debe hacer barata la migracion a Banner/Canvas: implementar un tercer adaptador
en `ingestion/`, no reescribir el pipeline.
