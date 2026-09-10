# Plan de avance con datos públicos

Supuesto de trabajo: **los accesos a Banner y Canvas se retrasan y no hay fecha
comprometida.** Este documento define qué se puede cumplir igual, qué no, y cómo
dejar todo listo para que la llegada de los datos internos sea una conexión y no
una reescritura.

---

## 1. El hallazgo que hace viable el plan

El proyecto se venía tratando a sí mismo como bloqueado. No lo está tanto:

> **98.812 ingresantes reales de la Universidad Autónoma**, en 19 cohortes
> (2007–2025), con su resultado de continuidad **observado**, están disponibles
> hoy en datos públicos. De la cohorte 2024 —6.475 personas— el **88%** tiene
> puntaje de admisión y el **89%** tiene NEM.

No son estudiantes sintéticos ni una muestra nacional: son los estudiantes de la
UA, identificados por MRUN, con su sede, su carrera, su área, su año de ingreso y
si continuaron o no. Eso permite entrenar y evaluar modelos **sobre la población
real del proyecto**, con validación temporal genuina de 19 cohortes.

Lo que sigue faltando es el **grano semanal dentro del curso** —notas parciales,
asistencia, actividad en Canvas—, que es exactamente lo que Banner y Canvas
aportan y ninguna fuente pública sustituye.

### Consecuencia para el alcance

| Se puede hoy, con estudiantes UA reales | Requiere Banner/Canvas |
|---|---|
| Segmentar la población de ingreso | Segmentar por conducta durante el semestre |
| Predecir **no continuidad** al año siguiente | Predecir **reprobación** de una asignatura |
| Perfilar riesgo al momento de matricular | Actualizar el riesgo semana a semana |
| Medir brechas por sede, carrera y área | Detectar deterioro y desenganche |

---

## 2. Los compromisos, releídos

| Compromiso | Plazo | Estado | Vía sin datos internos |
|---|---|---|---|
| Permisos de acceso VRA | mes 2 | **Bloqueado** | Ninguna. Es gestión, no técnica |
| Segmentos de estudiantes obtenidos | mes 3 | Alcanzable | Segmentación sobre los 98.812 ingresantes UA reales |
| ≥3 algoritmos, modelo priorizado 1 | mes 4 | Alcanzable con matices | Modelo de no continuidad sobre UA real + validación del pipeline semanal sobre OULAD |
| ≥3 algoritmos, modelo priorizado 2 | mes 6 | **Parcial** | La estructura sí; el desenganche conductual no |

**Lo que hay que decir explícitamente en el informe de avance:** el modelo de
reprobación por asignatura y el de desenganche **no se pueden entrenar** sobre
estudiantes UA sin los sistemas internos. Lo que se entrega en su lugar es un
modelo de no continuidad sobre la misma población real, más la demostración de
que el pipeline semanal funciona sobre datos conductuales reales (OULAD). Eso es
defendible; presentar un modelo sintético como si fuera un resultado, no.

---

## 3. El principio de arquitectura: la costura ya existe

Todo el pipeline aguas abajo consume **un diccionario de DataFrames en esquema
canónico**, no una fuente:

```python
build_features_at_week(data: dict[str, pd.DataFrame], week: int)
```

Hoy ese diccionario lo producen dos adaptadores —`load_oulad()` y el generador
sintético— y `config/data_contracts.yml` ya declara, campo por campo, de qué
tabla de Banner o Canvas saldrá cada uno y con qué nivel de confianza
(`confirmed` / `likely` / `verify`).

**Conectar los datos internos consiste en escribir un tercer adaptador que
devuelva el mismo diccionario.** Nada de features, marts, modelos ni tablero
debería cambiar. Todo el trabajo de este plan debe respetar esa costura.

### La pieza que falta para que la conexión sea segura

Hoy nada impide que un adaptador nuevo devuelva un esquema parecido pero mal:
una columna con otro nombre, una semana base distinta, una nota en escala 0–100
en vez de 1,0–7,0. Se descubriría tarde y en forma de resultados raros.

**Entregable transversal E0 — Test de contrato de adaptador.** Un único test
parametrizado que se corre contra *cualquier* adaptador y verifica el esquema, los
tipos, los rangos, las claves y el grano declarados en `data_contracts.yml`. Los
adaptadores existentes deben pasarlo hoy; el de Banner tendrá que pasarlo el día
uno. Es la pieza de mayor retorno del plan: convierte la migración de un ejercicio
de fe en uno con semáforo.

---

## 4. Fase A — Lo que se construye ahora

### A1 · Capa analítica: los marts (§21 del documento maestro)

Construir `mart_student_course_week` y `mart_student_week` **sobre el esquema
canónico**, alimentados por OULAD y por el generador. Son la pieza que hoy no
existe y que el proyecto declara pendiente desde el sprint 2.

`mart_student_week` debe traer las agregaciones que el documento maestro
especifica: `courses_enrolled`, `courses_at_risk`, `mean_grade`, `min_grade`,
`mean_attendance`, `min_attendance`, `submission_rate`, `missing_assignments`,
`canvas_activity`, `days_since_last_activity`, `academic_load_risk`.

Las columnas que hoy no tienen fuente real quedan **declaradas y vacías**, no
omitidas: así el día que llegue Canvas se pueblan sin alterar el esquema. Ausencia
de dato no es cero, y el mart debe distinguirlo.

### A2 · Feature Registry versionado (§22)

Cada feature declara nombre, versión, fuente, ventana temporal y
`available_from_week`. Es lo que permite reproducir exactamente qué vio un modelo
en una fecha dada, y es prerrequisito del Prediction Store.

### A3 · Prediction Store (§24)

Con el esquema exacto del documento maestro: `prediction_id`, `student_id`,
`course_id`, `period`, `week`, `model_name`, `model_version`, `score_date`,
`risk_score`, `risk_band`, `top_driver_1..3`, `feature_set_version`. **Nunca
sobrescribe.** Habilita trayectoria de riesgo, backtesting, auditoría y
comparación de versiones — y es requisito para poder explicar una decisión bajo la
Ley 21.719.

### A4 · Segmentación de estudiantes UA reales — cumple el mes 3

Sobre los 98.812 ingresantes reales, con features previas al ingreso: puntaje de
admisión, NEM, ranking, dependencia del colegio, asistencia de enseñanza media,
decil de ingreso, condición SEP, tipo de financiamiento, sede, carrera y área.

El documento maestro exige **≥3 alternativas**. Ya existe la maquinaria de
comparación en `modeling/benchmark.py` —K-means, Ward, mezcla gaussiana, con
silueta y techo de concentración— y es directamente reutilizable.

Dos cuidados que este proyecto ya aprendió y no debe repetir:

- La silueta sola elige siempre la partición que aísla atípicos. **Exigir techo de
  concentración**, como en el benchmark.
- Los segmentos deben describirse por **lo que implican para la acción**, no solo
  por su centroide. Un segmento que no cambia lo que hace una dirección de carrera
  es un segmento inútil aunque tenga buena silueta.

### A5 · Modelo de no continuidad sobre UA real — el mes 4, por la vía honesta

Tres algoritmos como mínimo (regresión logística regularizada, gradient boosting,
random forest), validación **temporal** por cohorte —entrenar en cohortes
antiguas, testear en la reciente—, y evaluación por **curva de capacidad**, no por
AUC: *si contacto al 20% de los ingresantes, ¿a qué fracción de los que se van
alcanzo?*

**Este modelo tiene un techo conocido y hay que declararlo por adelantado: AUC
≈ 0,64.** Ya está medido sobre 135.050 casos nacionales. Presentarlo como si
fuera a rendir más sería deshonesto; presentarlo *como el piso que el modelo 1
deberá superar cuando lleguen los datos internos* es exactamente el argumento del
proyecto, ahora sobre la población propia.

Complemento: correr el mismo pipeline semanal sobre OULAD para demostrar que la
maquinaria entrega cuando hay conducta. Son dos evidencias distintas y hay que
mantenerlas separadas en el informe.

### A6 · Student 360 y drivers (§41)

La ficha por estudiante, alimentada desde el Prediction Store. Con datos públicos
muestra trayectoria entre años; con Banner mostrará trayectoria entre semanas. La
vista es la misma, cambia la granularidad de la serie.

Los drivers con SHAP, **presentados como correlación observada y no como causa**,
que es lo que el documento maestro exige (§91) y lo que la Ley 21.719 obligará a
poder explicar.

### A7 · Evaluación de impacto y base de licitud — no depende de datos

La Ley 21.719 entra en plena vigencia el **1 de diciembre de 2026**. Exige base de
licitud, evaluación de impacto, derecho a explicación, intervención humana y no
ser objeto de decisiones exclusivamente automatizadas.

**Esto se puede y se debe hacer ahora, con la Dirección Jurídica, antes de que el
modelo toque un dato real.** Hoy no existe y es la única tarea del plan que no
tiene ninguna dependencia técnica. Si el acceso se destraba y esto no está listo,
se convierte en el nuevo bloqueo.

### A8 · Model cards

Una por modelo: propósito, población, features, métricas, limitaciones conocidas,
usos previstos y **usos explícitamente desaconsejados**. Ya hay una carpeta
`docs/model_cards` esperando.

---

## 5. Fase B — El día que lleguen los datos internos

La secuencia, en orden, y ninguna etapa empieza antes de que la anterior pase:

1. **Responder las seis preguntas de `open_questions`** en
   `config/data_contracts.yml`. Tres cambian el diseño del modelo:
   - ¿Usa la UA el módulo **Banner Attendance Tracking**? Es opcional. Sin él solo
     existe `SFRSTCR_LAST_ATTEND`, que no da grano semanal y se cae medio Caso 2.
   - ¿Hay **notas parciales fechadas** en `SHRTCKG` o solo la nota final del período?
   - ¿Contempla el reglamento **reprobación por inasistencia**, y con qué umbral?
     De la respuesta depende si la asistencia es predictor o mecanismo del target.

   Y tres condicionan la ingeniería y el calendario:
   - ¿Existe **Banner ODS / datawarehouse**, o hay que consultar las tablas
     transaccionales?
   - ¿Qué tenant de **Canvas Data 2** está habilitado y quién administra las
     credenciales DAP?
   - ¿**Cuántos años de historia** hay en Canvas? Esto acota directamente cuántas
     cohortes son entrenables, así que conviene preguntarlo **ahora** y no el día
     de la conexión: si Canvas tiene dos años, el plan de modelado cambia.

   Las tres últimas **no requieren el acceso concedido**: son preguntas a la
   contraparte técnica y se pueden hacer hoy, en paralelo a la gestión del permiso.

2. **Escribir `ingestion/banner.py` y `ingestion/canvas.py`** devolviendo el
   diccionario canónico.

3. **Correr el test de contrato (E0).** Si no pasa, el problema está en el
   adaptador, no aguas abajo. Ese es todo el punto.

4. **Reentrenar apuntando a `bronze/`.** Marts, features, modelos, Prediction Store
   y tablero no se tocan.

5. **Comparar contra el piso.** El modelo con datos internos debe superar
   claramente el AUC ≈ 0,64 del modelo de la fase A. Si no lo supera, el problema
   es el modelo o los datos, y hay que investigarlo antes de desplegar nada.

---

## 6. Riesgos del plan

**El riesgo dominante no es técnico.** Es que el retraso del acceso se prolongue
hasta consumir los plazos del fondo. El plan mitiga el costo, no la causa: hay que
escalar el permiso como gestión, con el informe metodológico como argumento —el
resultado de 0,636 contra 0,697 es precisamente la evidencia de que el acceso no
es burocracia sino la condición del proyecto.

**Riesgo de sustitución.** Presentar el modelo de no continuidad como si fuera el
modelo comprometido. No lo es, y hay que decirlo en cada entrega.

**Riesgo de deriva.** Este proyecto ya derivó una vez hacia el benchmark
institucional, que es buen trabajo pero no es lo que el fondo financió. La fase A
está deliberadamente acotada a piezas que el sprint 2 en adelante exige, y que
**no se botan** cuando lleguen los datos.

**Riesgo de sobreajuste al piso.** Optimizar mucho un modelo cuyo techo es 0,64 es
tiempo perdido. Basta con que sea correcto, honesto y esté bien evaluado.

---

## 7. Qué NO intentar

- **Más fuentes públicas.** Hay rendimientos decrecientes claros: el expediente
  completo previo al ingreso suma 0,019 de AUC sobre la ficha de admisión sola.
- **Reconstruir asistencia o notas parciales desde datos públicos.** No existen.
  Cualquier proxy sería una invención presentada como medición.
- **Comprar la PSU al DEMRE para extender la selectividad.** Mejora el benchmark,
  que no es el entregable comprometido.
