# Piloto FAN independiente

Implementación complementaria de la propuesta de cierre: perfiles de ingreso,
no continuidad al año siguiente y no titulación en un horizonte definido.
Los modelos de continuidad, titulación y perfiles de ingreso se entrenan y
evalúan con datos públicos reales de la UA.
Los seis casos ficticios de la demostración no participan en ningún resultado.
La nueva vista **Datos institucionales FAN** incorpora el extracto académico
universitario 2024–2025 como análisis descriptivo por período. Sus resultados
no se mezclan con las métricas de esos modelos.

## Abrir

Desde la raíz del repositorio, usando el entorno existente:

```powershell
.venv\Scripts\python.exe -m piloto_fan.build
.venv\Scripts\python.exe -m streamlit run piloto_fan/app.py --server.port 8502
```

Si los resultados ya están generados, basta el segundo comando o ejecutar
`piloto_fan/abrir.ps1`. La aplicación estará en http://localhost:8502.
El puerto diferente permite abrirla junto a la aplicación original.

Para instalar en otro entorno: `python -m pip install -r piloto_fan/requirements.txt`.
Se requieren las fuentes locales descritas abajo para regenerar los modelos.
No se descargan nuevas fuentes durante la generación.

## Aislamiento

- Todo el código nuevo está en `piloto_fan/`.
- Las fuentes se abren solo para lectura. No se cambian los conjuntos ni las
  etiquetas originales: la depuración y el seguimiento se recalculan aquí.
- El entrenamiento y los derivados individuales permanecen en este piloto.
  La integración posterior añade exportadores explícitos en `scripts/`,
  documentación de corridas y vistas de agregados en la aplicación existente.
  `data/interim/` se excluye en `.gitignore` para proteger el extracto.
- Cada ejecución crea `piloto_fan/artifacts/<fecha-uuid>/`. No sobrescribe corridas.
- `artifacts/latest.json` apunta a la última ejecución terminada y verificada.
- La carpeta de artefactos está ignorada por Git: contiene modelos y datos
  individuales locales. La app sirve resultados agregados, no los parquet de personas.
- Este piloto no actualiza automáticamente el Streamlit publicado. Los scripts
  `export_pilot_summary.py`, `export_institutional_aggregates.py` y
  `package_deployment.py` preparan explícitamente los agregados para esa app;
  véase [el contrato de publicación](../deploy/README.md#controles-de-los-agregados-institucionales).

## Fuentes reutilizadas

| Fuente existente | Uso |
|---|---|
| `data/results/ua_ingresantes.parquet` | Perfil y primer ingreso observado a la UA |
| `data/interim/matricula/matricula_*.parquet` | Recalcular continuidad e historia de matrícula UA |
| `data/external/titulados/titulados_*/*.csv` | Observar títulos de pregrado de la UA por año |

La generación conserva SHA-256 de cada insumo y verifica al terminar que no
haya cambiado durante la corrida. Los archivos originales no se reescriben.

## Población y límites

El alcance usa **un catálogo explícito FAN**, institución 31, área Administración,
plan regular según el código de la fuente y nivel profesional. La página oficial
de la facultad acredita Ingeniería Comercial, Ingeniería en Administración e
Ingeniería en Control de Gestión. El Plan de Desarrollo institucional de 2019
documenta además Auditoría e Ingeniería en Control de Gestión, Contador Público
Auditor e Ingeniería de Ejecución en Administración. Los enlaces y fecha de consulta
están en `config.json` y en la aplicación. Se conservan como seis nombres distintos:
no se inventan equivalencias entre carreras antiguas y actuales.

El catálogo delimita este piloto y **no representa toda la historia de FAN**:
no incluye sus antiguos programas técnicos, posgrados ni Administración Pública.
Las variantes de nombres no documentadas se excluyen y quedan en la auditoría.
La ratificación institucional de carreras y períodos de pertenencia sigue pendiente.
No se equipara `SEDE SANTIAGO` con San Miguel automáticamente.
El rótulo `CAMPUS EL LLANO` se muestra como **San Miguel / El Llano**, según
la página oficial de ese campus, enlazada en Calidad y método. Es un cambio
de presentación; las claves originales y la sede histórica Santiago se conservan.

El grano es una persona en su primer ingreso observado a UA dentro de la ventana
2007–2025. Se excluyen MRUN ausentes, ingresos posteriores y primeras entradas
ambiguas con más de una inscripción. No equivale necesariamente al primer ingreso
de toda su vida al sistema de educación superior. La cohorte inicial puede tener
truncamiento por falta de historia anterior a 2007.

Los 52 registros sin MRUN de la fuente se excluyen aquí: el cero de seguimiento
heredado no se considera un resultado real. Las sedes conservan su denominación
original. Las coberturas se muestran por cohorte.

### Segmentación

Compara K-means, mezcla gaussiana diagonal y K-means bisectriz para 2–5 grupos.
Usa percentiles de admisión, NEM y ranking, duración nominal y jornada.
Imputa las variables numéricas con la mediana, las estandariza y codifica jornada.
No usa resultado, sede ni carrera para definir los grupos.

Aprende preprocesamiento y grupos solo en las cohortes anteriores a las dos
últimas; asigna las posteriores sin reajustar. Elige la mayor silueta entre
particiones que respetan tamaño y concentración. La numeración se congela con
el tamaño de los grupos de entrenamiento. Los perfiles describen resultados
observados, pero no diagnostican necesidades ni explican causas.

NEM y ranking son percentiles dentro de la cohorte UA de la fuente existente;
admisión utiliza su referencia nacional. La imputación puede influir en la
partición: se informa la cobertura por grupo. La variación total entre tamaños
de grupos mide cambio de composición, no estabilidad individual ni validación
de acciones de apoyo.

### Caso 1: continuidad

Población de ingreso desde 2021. Resultado: no observar matrícula UA en el año
siguiente. Se recalcula desde matrícula, no se confía en la etiqueta original.
No observar UA no distingue abandono definitivo de traslado a otra institución.

Las variables son los tres percentiles, duración nominal, carrera, sede y
jornada al ingreso. No se incorporan datos socioeconómicos con cobertura limitada
a dos años ni variables semanales que no existen para la UA.

### Caso 2: titulación

Se evalúa en el año `ingreso + 3`, incluyendo matrícula observada hasta ese año.
Elegibles: personas matriculadas UA en ese corte, con duración nominal mayor a
tres y hasta ocho años, sin título UA observado desde su ingreso hasta el corte.
El horizonte es `ceil(ingreso + duración nominal + 2)`. El resultado es no
observar título de pregrado UA entre el corte y ese horizonte.

Las variables describen años matriculados, años sin matrícula UA, cantidad de
códigos de carrera, sedes y jornadas observadas hasta el corte, más duración y
perfil inicial. Un cambio de código puede ser administrativo; no se interpreta
automáticamente como cambio real de carrera.

Se conservan todos los años de titulación: un título previo no oculta uno
posterior. Si falta un año de seguimiento o el horizonte es posterior a los
archivos disponibles, el resultado queda desconocido, nunca negativo.
El test usa un año de corte cuya población elegible tiene resultado completo.
Este resultado está condicionado a seguir matriculado en el corte y no es la
tasa de titulación de la cohorte completa de ingreso.

### Comparación predictiva

Ambos casos comparan regresión logística regularizada, random forest y gradient
boosting. La selección usa captura al 20% en validación; los desempates usan Brier
y AUC. Después se reajusta el ganador con datos maduros y se evalúa una sola vez
en la fecha reservada. No se optimizan hiperparámetros con el test.

No basta ordenar las cohortes: cada resultado usado para entrenar o seleccionar
debe estar maduro en la fecha simulada. Se asume que una etiqueta se conoce en
`año del resultado + 1`. Este supuesto es conservador respecto del año observado,
pero **no reconstruye las fechas reales de publicación de los datos abiertos**.
El análisis es una reconstrucción de registros académicos, no un backtest de un
servicio que realmente hubiera existido en esas fechas.

Se incluyen una probabilidad constante aprendida en entrenamiento y una regla
de tasas históricas por carrera con contracción de 30 casos hacia el promedio.
Captura y precisión tratan los empates mediante selección fraccional esperada,
para no fabricar capacidad predictiva con el orden de las filas.

Se informa AUC, precisión promedio (AP), Brier, curva de calibración, capacidad y
bootstrap por persona del test. Los intervalos no incluyen incertidumbre por
deriva entre cohortes ni errores de las fuentes. El estado `señal retrospectiva`
requiere mejora frente a la regla histórica y el azar en captura con el intervalo
configurado, además de mejorar Brier frente a la constante de entrenamiento.
Si no, queda `experimental`. Ninguno autoriza uso individual operativo.

## Artefactos y actualización

Cada ejecución incluye:

- `report.json` e `informe_ejecutivo.html`: resultados agregados para revisión.
- `*_model_card.json`: propósito, variables, partición temporal, resultados y límites.
- `feature_registry.json`: versión, variables, momento de observación y exclusiones.
- `*_model.joblib`: modelos y transformaciones ajustadas; solo cargar archivos
  generados por este piloto en un entorno confiable.
- `*_predictions.parquet`: test reservado, identificador de predicción, fecha y
  versiones. No se publican ni se muestran en la aplicación.
- Conjuntos derivados y asignaciones de segmentos, de uso local.
- `config.json` y `manifest.json`: configuración, versiones y huellas de fuentes,
  código y artefactos.

Para actualizar tras disponer de nuevas fuentes locales, volver a ejecutar
`python -m piloto_fan.build`. La cadencia es anual, acorde con los registros;
no se generan alertas semanales a partir de datos anuales. La corrida previa
permanece intacta si una nueva falla. No se requiere entrenar desde Streamlit.

## Verificación

La revisión de usabilidad incorpora filtros dependientes de sede/campus, carrera
y jornada al ingreso, con botón de restablecimiento. Resumen y Perfiles agregan
selección de cohorte. Las vistas predictivas usan exclusivamente la fecha de test
disponible: al cambiar los filtros se recalculan sus métricas y la prioridad
dentro del subgrupo, sin reentrenar ni reemplazar selecciones vacías por el total.

Las comparaciones de algoritmos, estado general del modelo y auditoría global
se identifican como no filtradas. Se distingue cohorte de ingreso de fecha de
evaluación y horizonte. La cifra de bootstrap al 20% se etiqueta como referencia
fija aunque el deslizador elija otra capacidad. Se controlan ausencia de casos,
grupos pequeños y grupos con una sola clase (AUC no calculable).

Formato: coma decimal, punto de miles, porcentajes explícitos y “—” para ausentes.
Los CSV filtrados llevan BOM UTF-8, separador punto y coma, coma decimal y contexto
de ejecución, cohorte y filtros. No exportan MRUN ni predicciones individuales.
Los informes HTML/JSON de la barra lateral se etiquetan como generales sin filtros.

```powershell
.venv\Scripts\python.exe -m pytest piloto_fan/tests -q
git diff --exit-code
```

Las pruebas cubren exclusión de identidades ausentes, primer ingreso,
reconstrucción de continuidad, censura y huecos de titulados, ausencia de datos
futuros en las variables, madurez de etiquetas, independencia de selección
respecto del test, empates, segmentación y navegación/exportaciones.
Las pruebas de artefactos y app requieren generar primero los resultados.

## Pendientes de cierre institucional

La implementación técnica no reemplaza el acuerdo de alcance ni la validación
con las direcciones. El archivo `validacion_institucional.json` contiene el
protocolo de prueba y los compromisos pendientes, sin simular su cumplimiento.

Prioridades de las próximas sesiones:

1. Confirmar carreras, niveles, planes y equivalencias de sedes con FAN.
2. Acordar los dos casos, horizonte y capacidad de revisión de interés.
3. Evaluar interpretación y utilidad con representantes de las cuatro sedes.
4. Registrar responsables, frecuencia de actualización y aceptación o ajustes.
5. Adjuntar el informe agregado a la rendición y documentar el acceso VRA pendiente.

El benchmark publicado se enlaza como complemento. OULAD y el cockpit semanal
siguen disponibles en la aplicación existente. El acceso a Banner/Canvas,
contactos individuales, acciones preventivas y medición causal del impacto no
se declaran implementados por este piloto.

## Fuente institucional 2024–2025

Actualización del 20 de octubre de 2026. El usuario confirmó que la universidad
generó el Excel ubicado en `data/interim/Matricula e Historia Académica Pregrado 2024 2025.xlsx`.
La recepción permite trabajar con historia académica real; no confirma la
fecha de corte, los filtros de extracción ni una conexión directa a los sistemas.

### Lo que entendimos y cómo se aplica

El objetivo es complementar el benchmark y los modelos públicos con una
herramienta de gestión académica FAN. Se mantienen las preguntas pendientes
con una forma explícita de continuar. No se simulan notas ni matrículas y no se
declaran aprobaciones institucionales inexistentes.

El archivo contiene 569.148 registros académicos y 133.975 filas de matrícula/plan.
La etiqueta FAN incluye 4.710 personas en 2024–2025. El alcance propuesto para los
cuatro campus físicos contiene 4.308 personas; aún no equivale a matrícula efectiva.
La población es distinta de los 9.556 primeros ingresos históricos del piloto público.

Implementado:

- Adaptador independiente `institucional.py`, con claves de origen verificadas,
  interpretación de coma decimal y conservación de códigos no numéricos.
- Matrícula por persona–período–programa, resultados por persona–período–NRC
  y resumen por persona–período. No se fuerzan datos anuales/semestrales a una
  estructura semanal.
- Vinculación sin multiplicar resultados. Se atribuyen notas a carrera/campus
  solo si hay un único plan institucional en ese período. Los planes múltiples
  y los registros sin coincidencia se conservan en la auditoría.
- Identificador local estable mediante HMAC. La clave se conserva en
  `artifacts/institucional/.identity_key`, ignorada por Git; los derivados no
  contienen PIDM original. No se vinculan estos identificadores a MRUN.
- Corridas independientes en `artifacts/institucional/<run_id>/`, manifiesto con
  hashes de fuente/configuración/artefactos y puntero propio `latest.json`.
  No se reemplazan las corridas de los modelos públicos.
- Nueva vista **Datos institucionales FAN**, disponible sin los artefactos del
  piloto público. Alcance, año, régimen, período, campus, carrera, jornada y
  estado del plan tienen filtros propios y botón de restablecimiento.
- Panorama académico, comparaciones por asignatura, códigos de resultado,
  auditoría general y preguntas pendientes. Descargas agregadas con filtros,
  UTF-8 BOM, coma decimal y separador punto y coma; no se exportan personas.

Para generar desde el Excel original:

```powershell
.venv\Scripts\python.exe -m piloto_fan.institucional
.venv\Scripts\python.exe -m streamlit run piloto_fan/app.py --server.port 8502
```

Seleccionar **Datos institucionales FAN** en la barra lateral. No hay que
regenerar los modelos públicos. La generación institucional lee todo el libro;
la app utiliza los derivados ya verificados y no lee el Excel en cada filtro.

Los valores iniciales son el último año disponible y régimen semestral.
La alternativa «Todo lo etiquetado FAN» permite explorar Online e intercambio
sin incorporarlos silenciosamente al alcance propuesto.

### Preguntas necesarias, con avance habilitado

El registro editable es `institucional_pendientes.json`: pregunta, responsable
sugerido, prioridad, estado, respuesta, trabajo habilitado mientras se espera y
limitación actual. Las preguntas también están en el tablero y se descargan
como texto. Las respuestas se registran allí y se generan nuevas corridas cuando
corresponda; no se sobrescriben resultados anteriores ni se aplican cambios
semánticos automáticamente.

| ID | Pregunta a la universidad | Mientras se confirma |
|---|---|---|
| P01 | ¿Fecha de corte, filtros de extracción y cobertura de estados/períodos? | Presencia observada en 2024–2025; fecha de corte no informada. |
| P02 | ¿Significado de RI/RR/RM/RC/CR/CC/P/REPROB y reglas de aprobación? | Notas numéricas y códigos separados; «notas bajo 4» no se llama reprobación oficial. |
| P03 | ¿Unidad de CREDIT_HOURS y equivalencia por plan con SCT? | Conteos y promedios simples; sin ponderar por esa magnitud. |
| P04 | ¿Cómo atribuir asignaturas cuando hay planes simultáneos y cómo interpretar prioridades/secuencias? | Se conservan los casos ambiguos fuera de comparaciones académicas por carrera. |
| P05 | ¿Calendario real, cierres, disponibilidad de notas y vigencia histórica de estados? | Comparaciones descriptivas por régimen; sin reconstruir semanas ni fechas inventadas. |
| P06 | ¿Catálogo FAN, campus y tratamiento de Online, intercambio y excepciones? | Alcance propuesto y universo etiquetado FAN separados, sin aprobación supuesta. |
| P07 | ¿Estados de matrícula efectiva y registros de egreso, titulación, suspensión y traslado? | Filtros por estado informado; ausencia no equivale a deserción. |
| P08 | ¿Catálogo y equivalencias de asignaturas, secciones y mallas? | Códigos/nombres originales; sin equivalencias ni repitencia histórica inventadas. |
| P09 | ¿Historia 2022–2023, actualización 2026 y responsable/cadencia de entrega? | Ventana 2024–2025 explícita; sin alertas sobre estudiantes actuales. |
| P10 | ¿Dos casos prioritarios, decisión apoyada, capacidad de revisión y criterios de aceptación? | Diagnóstico y propuesta de evaluación; sin afirmar validación de usuarios. |

### Interpretación de los indicadores

- Promedio simple y porcentaje bajo 4 usan únicamente notas numéricas entre 1 y 7.
  Los códigos, los ausentes y las notas fuera de rango no se transforman en cero.
- La cobertura numérica informa qué fracción de resultados atribuibles contiene
  una nota numérica. Los registros académicos observados no prueban el total de
  cursos inscritos, y la ausencia de historia no implica cero carga.
- Las personas se cuentan una sola vez en la selección; los subtotales por
  período/carrera/campus pueden superponerse. No se promedian porcentajes de
  grupos para obtener un total.
- Los filtros de campus se refieren al plan. Las diferencias con el campus
  de la asignatura se señalan; no se corrigen sin una regla institucional.
- Se omiten grupos con menos de 10 personas y estadísticas numéricas sin
  suficientes personas con nota. Los CSV contienen los mismos agregados de la vista.
- La auditoría general y las preguntas están identificadas como no filtradas.

### Próximos entregables

1. Ratificar reglas y alcance con las respuestas, manteniendo los diagnósticos utilizables.
2. Revisar con las direcciones los perfiles académicos y resultados de los modelos
   experimentales descritos a continuación.
3. Priorizar formalmente los casos y confirmar etiquetas, estados y disponibilidad
   temporal. Los experimentos bajo supuestos no sustituyen esa validación.
4. Validar tareas y decisiones con las cuatro sedes y acordar acceso institucional
   y actualización. La app pública no recibe este Excel ni sus derivados individuales.

El extracto no contiene Canvas, asistencia ni notas parciales fechadas. Los modelos
anteriores conservan sus métricas y no han sido reentrenados con esta fuente.

## Modelos institucionales experimentales

La sección **Modelos institucionales** utiliza exclusivamente los derivados del
Excel institucional. No modifica ni combina los modelos públicos por MRUN.

```powershell
.venv\Scripts\python.exe -m piloto_fan.modelos_institucionales
```

El comando entrena desde la última integración verificada, no vuelve a leer
las hojas del Excel ni cambia el puntero de los datos. Guarda modelos,
predicciones locales seudonimizadas, fichas por caso, registro de variables,
comparaciones y manifiesto en `artifacts/institucional_modelos/<run_id>/`.
El puntero propio se actualiza solo después de verificar los artefactos.
La interfaz carga resultados precalculados y no entrena al cambiar filtros.

### Población y variables

Estudiantes del alcance FAN propuesto, con un único plan, régimen semestral y
al menos una nota numérica en el semestre de origen. Se usan resultados de
ese semestre: cantidad observada, promedio, dispersión, cobertura numérica y
fracción bajo 4, además de carrera, campus y jornada.

No se incluyen identificadores, estados administrativos, secuencias, carga de
unidad desconocida ni resultados del semestre objetivo. Los estudiantes sin
historia numérica previa no pertenecen a esta evaluación.

### Objetivos, algoritmos y partición

- **Resultado académico:** al menos una nota numérica bajo 4 en el siguiente
  semestre. Se confirma un positivo cuando se observa esa nota. Un negativo
  exige historia observada completamente numérica sin notas bajo 4; códigos,
  resultados fuera de rango o ausencia de historia impiden afirmar un negativo.
  Los desconocidos no se imputan como éxito ni fracaso.
- **Presencia en matrícula:** no observar ningún registro UA en el siguiente
  semestre ordinario, considerando todos los estados y carreras. No equivale
  a deserción ni matrícula efectiva; egreso y cambios de régimen pueden influir.
- Cada caso compara regresión logística, random forest y gradient boosting.
  Se elige por captura al 20% en validación, con desempate Brier/AUC.
- Entrenamiento: **2024-1→2024-2**; validación: **2024-2→2025-1**;
  prueba: **2025-1→2025-2**. El ganador se reajusta con los dos conjuntos
  anteriores antes de evaluar la prueba reservada.
- Se supone que las notas finales estuvieron disponibles al cierre del semestre,
  antes del siguiente. No conocemos sus fechas reales; es una reconstrucción
  retrospectiva hipotética y no un backtest de alertas que hayan operado.
- La misma persona puede reaparecer en períodos sucesivos; se informa ese
  solapamiento. La prueba contiene una fila por persona, con bootstrap por
  persona. No se hace partición aleatoria por registros.
- Se compara con reglas simples y con el mismo algoritmo usando únicamente
  carrera/campus/jornada para medir el aporte incremental de las notas.
  El test no selecciona algoritmos ni modifica sus hiperparámetros.

### Segmentación académica

Compara K-means, mezcla gaussiana diagonal y K-means bisectriz con 2–5 grupos.
Aprende imputación, escalamiento y segmentos en 2024-1; asigna los semestres
posteriores sin reajustar. Usa las cinco variables académicas numéricas, sin
carrera, campus, jornada ni resultados futuros. Exige mínimo 80 personas y
máximo 80% del entrenamiento por grupo; selecciona por silueta.
La numeración queda fijada por el tamaño de los grupos del entrenamiento.
Los perfiles describen resultados observados, no riesgos ni necesidades diagnosticadas.

### Lectura y límites

La interfaz separa perfiles, resultado académico y presencia en matrícula.
Los filtros predictivos corresponden a carrera/campus/jornada del semestre de
origen del test; no cambian su fecha ni reentrenan. Las comparaciones de
algoritmos y los intervalos generales se identifican como no filtrados, con
referencia fija al 20%. Se informa descalibración, pocos eventos, cambio de
frecuencia y falta de mejora de Brier frente a una probabilidad constante.

Los resultados y tamaños exactos se leen de la corrida, evitando cifras
duplicadas en la documentación. La primera evaluación muestra señal útil en
resultado académico y un caso de matrícula con muy pocos eventos y problemas
de calibración. Ambos permanecen **experimentales**. Los pendientes institucionales
siguen abiertos y no se genera scoring de alumnos actuales.
