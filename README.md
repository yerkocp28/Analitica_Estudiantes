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
| 2 | Capa analítica: silver, `mart_student_course_week`, `mart_student_week` | Pendiente |
| 3 | Segmentación (≥3 alternativas) | Pendiente |
| 4 | Modelo 1 — riesgo de reprobación | Pendiente |
| 5 | Producto: cockpit y Student 360 | Pendiente |
| 6 | Modelo 2 — desenganche académico | Pendiente |

**Sin accesos a Banner/Canvas todavía.** Todo lo que existe corre sobre datos
sintéticos calibrados contra estadísticas públicas del sistema chileno.

---

## Estrategia de datos antes de tener accesos

Tres fuentes, tres problemas distintos:

| Fuente | Resuelve | No resuelve |
|---|---|---|
| Esquemas públicos de Canvas Data 2 y Banner ODS | Que el pipeline reciba las **columnas reales** | No trae datos |
| **OULAD** (Open University, 32k estudiantes reales) | ¿Funciona la predicción temprana? ¿Cuán temprano? | Sin asistencia, sin escala 1–7, sin sedes |
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
pip install -e ".[modeling,dev]"

python scripts/generate_synthetic.py                 # volumen de config (10.000)
python scripts/generate_synthetic.py --students 500  # smoke test
python scripts/validate_lead_time.py                 # curva poder vs anticipación
pytest -q
```

---

## Decisiones de diseño

**El generador se auto-calibra contra anclas, no contra constantes.**
`config/synthetic.yml` declara tasas objetivo (retención 82,8% — ancla SIES 2022;
reprobación por asignatura 20%) y el código resuelve sus parámetros internos por
bisección para satisfacerlas. Cambiar el peso de una señal no altera la tasa
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
3.000 estudiantes, prevalencia de reprobación 24,5%:

| Semana de scoring | AUC | Top 10% | Top 20% | Semanas restantes |
|---|---|---|---|---|
| 3 | 0,661 | 16,4% | 30,4% | 15 |
| 5 | 0,687 | 16,8% | 31,9% | 13 |
| 8 | 0,722 | 18,9% | 36,1% | 10 |
| 12 | 0,751 | 22,8% | 41,3% | 6 |

*Top 20% = de los estudiantes que efectivamente reprobaron, qué fracción queda
dentro del 20% priorizado por el modelo esa semana. Baseline aleatorio = 20%.*

**Estas cifras no son un hallazgo sobre la UA.** Miden que el pipeline recupera
la señal que el generador inyectó, y que la dificultad del problema simulado está
en el rango que reporta la literatura. El número que importa saldrá de datos
reales.

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
    config.py            carga y validación de YAML
    logging_setup.py
    synthetic/
        profiles.py      7 perfiles latentes de estudiante
        generator.py     generador reproducible y auto-calibrado
scripts/
    generate_synthetic.py
    validate_lead_time.py    diagnóstico: poder predictivo vs anticipación
tests/           40 tests, incluida invarianza anti-leakage
docs/            documentos base del proyecto
```
