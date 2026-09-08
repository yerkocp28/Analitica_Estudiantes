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
| **Mineduc / SIES abiertos** (individual, vía MRUN) | Retención real de la UA; piso predictivo nacional | No se puede unir al RUT de Banner |
| **Generador sintético** (este repo) | Forma UA completa: sedes, carreras, asistencia, notas 1.0–7.0 | No valida nada por sí solo |

El objetivo de esta fase no es tener un modelo. Es tener un pipeline cuyo
contrato de entrada calce con Banner/Canvas lo bastante bien como para que la
migración a datos reales sea un cambio de configuración. Ese contrato vive en
[config/data_contracts.yml](config/data_contracts.yml), donde cada campo declara
su origen real esperado y su nivel de confianza (`confirmed` / `likely` /
`verify`).

---

## Uso

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
pip install -e ".[modeling,ui,dev]"

# 1. Datos sintéticos (forma UA)
python scripts/generate_synthetic.py                 # volumen de config (10.000)
python scripts/generate_synthetic.py --students 500  # smoke test

# 2. Datos reales de referencia (~45 MB comprimidos)
python scripts/download_oulad.py

# 2b. Bases abiertas de Mineduc/SIES (~1,3 GB); retencion real y piso predictivo
python scripts/download_mineduc.py
python scripts/analyze_mineduc.py

# 3. Curva de poder predictivo vs anticipación, sobre cualquier fuente
python scripts/validate_lead_time.py --source synthetic --out data/results
python scripts/validate_lead_time.py --source oulad     --out data/results

# 4. Cockpit
streamlit run src/student_analytics/ui/app.py

pytest -q                                            # 58 tests
```

---

## La herramienta de visualización

`streamlit run src/student_analytics/ui/app.py` — tres vistas:

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
python scripts/download_mineduc.py    # ~1,3 GB comprimidos
python scripts/analyze_mineduc.py
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

### El piso predictivo de las variables previas al ingreso

Cruce PAES 2024 → Matrícula 2024 → Matrícula 2025 por MRUN. Predice no
continuar en la misma carrera, usando **solo** NEM, ranking, notas de enseñanza
media, puntajes PAES, dependencia del colegio, sexo y año de egreso:

| Población | n | Prevalencia | AUC | Top 20% |
|---|---|---|---|---|
| Todo el sistema | 184.279 | 18,3% | 0,626 | 31,8% (1,59×) |
| **Solo universidades** | 135.050 | 16,4% | **0,615** | 31,5% (1,58×) |

**Esto replica el hallazgo de ULagos con 135.000 estudiantes en vez de dos
cohortes**, y es el argumento central del proyecto:

> Toda la ficha de admisión junta alcanza **AUC 0,615**. El baseline sobre
> comportamiento en OULAD llega a **0,697 en la semana 4** — con el 10% del
> curso transcurrido y **sin asistencia**. Los datos de comportamiento
> intrasemestral superan a la ficha de admisión completa antes del primer mes.

Ese es el caso para pedir acceso a Banner y Canvas: el valor no está en lo que
ya se sabe del estudiante al matricularse.

*Caveats: solo el 53% de los ingresantes 2024 aparece en PAES 2024 (los ingresos
a IP/CFT y por vías alternativas no la rinden). Split aleatorio, no temporal —
hay una sola cohorte, y si acaso eso favorece al modelo, así que el piso real es
aún más bajo.*

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
config/          settings.yml, synthetic.yml, data_contracts.yml
src/student_analytics/
    config.py               carga y validacion de YAML
    logging_setup.py
    synthetic/
        profiles.py         7 perfiles latentes de estudiante
        generator.py        generador reproducible y auto-calibrado
    ingestion/
        oulad.py            OULAD -> esquema canonico
    features/
        builder.py          features por semana, agnostico a la fuente
    modeling/
        lead_time.py        poder predictivo vs anticipacion
    ui/
        app.py              cockpit Streamlit
scripts/
    generate_synthetic.py
    download_oulad.py
    download_mineduc.py     bases abiertas Mineduc/SIES
    analyze_mineduc.py      retencion real UA + piso predictivo nacional
    validate_lead_time.py   --source synthetic | oulad
tests/           58 tests: generador, leakage, OULAD, UI
docs/            documentos base del proyecto
```

El feature builder y el evaluador **no saben de que fuente vienen los datos**.
Corren igual sobre el generador sintetico y sobre OULAD. Esa es la propiedad que
debe hacer barata la migracion a Banner/Canvas: implementar un tercer adaptador
en `ingestion/`, no reescribir el pipeline.
