# Evaluación de impacto en protección de datos — Ley 21.719

**Borrador técnico para la Dirección Jurídica.** Versión 1, 21 de octubre de 2026.

> Este documento **no es una opinión legal**. Lo redacta el equipo técnico del
> proyecto para describir con precisión qué datos se tratan hoy, cómo y con qué
> salvaguardas, de modo que la Dirección Jurídica pueda evaluarlo. Las
> calificaciones jurídicas —base de licitud aplicable, si el tratamiento exige
> evaluación de impacto formal, si procede consulta previa— corresponden a esa
> dirección, no a este equipo.

## Por qué ahora

La Ley 21.719 entra en plena vigencia el **1 de diciembre de 2026**. Consagra el
derecho a no ser objeto de decisiones basadas exclusivamente en tratamiento
automatizado, con garantías de explicación, intervención humana y revisión.

El proyecto **todavía no toma ninguna decisión sobre personas**, y ese es
precisamente el momento de hacer esto: cuando aún no hay nada que corregir. El
objetivo es llegar a la vigencia con la base de licitud establecida y los
controles acordados, no con un modelo en producción y la conversación pendiente.

---

## 1. Qué se trata hoy

El proyecto maneja **tres cuerpos de datos** con naturaleza jurídica distinta.
Conviene no tratarlos como uno solo.

### 1.1 Datos abiertos del Estado (Mineduc, CNED)

Matrícula 2007–2026, titulados 2007–2025, admisión 2021–2026 y bases escolares.
Publicados por el Ministerio de Educación bajo licencia de datos abiertos de
gobierno.

Están desagregados a nivel individual y usan **MRUN**, un identificador
enmascarado y estable entre bases y años. **No existe tabla de equivalencia con
el RUT**, y por tanto no es posible —ni se intenta— vincular estos registros con
una persona concreta de la universidad.

Volumen: 98.812 ingresantes de la UA en 19 cohortes.

### 1.2 Extracto institucional de la propia universidad

`Matrícula e Historia Académica Pregrado 2024 2025.xlsx`, generado por la
universidad y recibido el 20 de octubre de 2026. **Este es el cuerpo sensible**:
569.148 registros académicos y 133.975 de matrícula, de **40.590 personas**
identificables dentro de los sistemas institucionales.

Contiene calificaciones, estados de matrícula, carrera, campus y jornada. No
contiene nombre, RUT, dirección, contacto ni datos de salud.

### 1.3 Datos que no son de personas reales de la UA

OULAD, un conjunto público del Reino Unido bajo CC BY 4.0 con identificadores ya
anonimizados por su origen, y un generador sintético propio. Ninguna persona de
la Universidad Autónoma está en ellos.

---

## 2. Finalidad

**Identificar tempranamente a estudiantes en riesgo académico para ofrecerles
apoyo**, y caracterizar el desempeño institucional comparado.

Lo que la finalidad **no** incluye, y conviene dejarlo por escrito porque limita
usos futuros: no es seleccionar, excluir ni desmatricular; no es evaluar
docentes; no es asignar beneficios ni retirarlos; no es alimentar ningún proceso
disciplinario.

---

## 3. Salvaguardas ya implementadas

No son intenciones: están en el código y cubiertas por pruebas automatizadas.

| Salvaguarda | Cómo está implementada |
|---|---|
| **Seudonimización del extracto** | El identificador interno (PIDM) se reemplaza por HMAC-SHA256 con clave local. La clave vive fuera del control de versiones; ningún derivado contiene el PIDM original |
| **Sin vinculación entre cuerpos** | Los identificadores institucionales **no** se cruzan con MRUN. Son universos separados a propósito |
| **La interfaz solo muestra agregados** | Las aplicaciones no exponen personas. Las descargas llevan agregados con filtros; no exportan identificadores ni predicciones individuales |
| **Mínimo de personas por celda** | Los cortes con menos de 10 personas no se muestran |
| **Nada individual sale del repositorio** | Las carpetas con datos de personas están excluidas del control de versiones. El paquete publicado contiene únicamente agregados institucionales |
| **Exclusión de variables sensibles** | Las variables socioeconómicas no entran al scoring. La ablación mostró que aportan +0,014 de AUC: excluirlas cuesta casi nada y evita discriminar por origen |
| **Sin scoring de alumnos actuales** | Los modelos son reconstrucciones retrospectivas. Ninguno genera puntajes sobre estudiantes en curso |

### 3.1 La decisión sobre variables socioeconómicas

Merece explicarse porque es una decisión de diseño con fundamento empírico, no
una precaución genérica.

Se midió cuánto aportan al modelo el decil de ingreso oficial, la condición de
prioritario y el tipo de financiamiento: **+0,014 de AUC** sobre la ficha de
admisión. Es decir, usarlas para priorizar estudiantes mejora el modelo de forma
marginal mientras introduce un riesgo serio de que el apoyo se asigne por origen
socioeconómico.

**Se usan para análisis de equidad agregado, nunca para priorizar personas.**

---

## 4. Riesgos identificados y su tratamiento

### 4.1 Decisión automatizada sobre una persona — riesgo alto

Es el riesgo que la ley aborda directamente. Si un modelo ordenara estudiantes y
alguien actuara sobre esa lista sin criterio propio, habría una decisión
basada exclusivamente en tratamiento automatizado.

**Tratamiento comprometido:** el modelo prioriza, **no decide**. La revisión y
cualquier acción son de la dirección de carrera, que debe poder apartarse de la
lista sin justificar nada. Esto ya está declarado en la aplicación y debe quedar
en el protocolo operativo que la facultad apruebe.

**Pendiente:** no existe todavía un registro que acredite la intervención humana
caso a caso. Cuando haya operación real, la `Prediction Store` debería registrar
quién revisó, cuándo y qué decidió — incluyendo cuando decidió no actuar.

### 4.2 Derecho a explicación — riesgo alto

**Tratamiento:** cada modelo tiene una *model card* con propósito, variables,
partición temporal, resultados y límites. Los *drivers* por caso se calculan y
se presentan **como correlación observada, nunca como causa**.

**Pendiente:** falta decidir qué se le explica al estudiante, en qué lenguaje y
por qué vía. Explicar un gradient boosting a una persona no es lo mismo que
documentarlo para una auditoría.

### 4.3 Estigmatización por etiqueta — riesgo medio

Clasificar a alguien como "en riesgo" puede afectar cómo lo tratan docentes y
pares, y puede volverse profecía autocumplida.

**Tratamiento:** la segmentación describe grupos por perfil de ingreso y
**nunca por resultado** —hay una prueba que verifica que cambiar el resultado no
mueve los segmentos—. Los perfiles describen lo observado; no diagnostican
necesidades ni explican causas.

**Pendiente:** acordar quién ve las listas y quién no.

### 4.4 Uso para fines distintos del comprometido — riesgo medio

Un modelo de riesgo académico sirve técnicamente para seleccionar admisión o
para decidir a quién no renovar un beneficio. Esos usos están fuera de la
finalidad y deben quedar **prohibidos explícitamente** en el acuerdo.

**Tratamiento:** cada *model card* declara usos previstos y **usos prohibidos**.

### 4.5 Reidentificación — riesgo bajo pero no nulo

La seudonimización por HMAC es reversible **por quien tenga la clave**. No es
anonimización.

**Pendiente:** definir custodia de la clave, rotación, y qué ocurre con los
derivados si se rota.

### 4.6 Calidad del dato que lleva a una conclusión falsa — riesgo medio

Este proyecto ya encontró catorce trampas de datos que habrían producido
conclusiones falsas con toda naturalidad: centinelas leídos como valores,
cambios de escala leídos como cambios reales, grano no único que abre los joins.

Una decisión sobre una persona apoyada en un dato mal leído es un daño concreto.

**Tratamiento:** contrato de datos verificado en código, 233 pruebas
automatizadas, y la disciplina de que ninguna cifra publicada se escriba a mano.

---

## 5. Lo que falta decidir — para la Dirección Jurídica

Son las preguntas que este equipo no puede responder y que condicionan el resto.

1. **Base de licitud.** ¿Interés legítimo de la institución, ejecución del
   vínculo académico, consentimiento? De la respuesta depende si hace falta
   recabar algo del estudiante y en qué momento.
2. **¿Este tratamiento exige evaluación de impacto formal** en los términos de
   la ley, y debe inscribirse en algún registro?
3. **Información al estudiante.** ¿Qué debe saber, cuándo y por qué medio? ¿Basta
   el reglamento académico o requiere aviso específico?
4. **Ejercicio de derechos.** ¿Por qué canal puede un estudiante acceder,
   rectificar, oponerse o pedir revisión humana? ¿Quién responde y en qué plazo?
5. **Plazo de conservación** de predicciones y derivados, y qué se hace al
   vencer.
6. **Encargados de tratamiento.** El repositorio está en GitHub (privado) y una
   aplicación con **solo agregados públicos** puede publicarse en Streamlit
   Community Cloud. ¿Requiere eso alguna formalización?
7. **El extracto institucional.** ¿Su entrega al equipo del proyecto está
   cubierta por algún acto administrativo, o requiere uno?

---

## 6. Estado y siguiente paso

| | |
|---|---|
| Base de licitud | **No establecida** |
| Evaluación de impacto formal | **No realizada** — este documento es el insumo |
| Información al estudiante | **No definida** |
| Canal de ejercicio de derechos | **No definido** |
| Registro de intervención humana | **No implementado** (no hay operación todavía) |
| Salvaguardas técnicas | **Implementadas y verificadas** (sección 3) |
| Decisiones automatizadas en producción | **Ninguna** |

**El proyecto no está incumpliendo hoy**, porque no toma decisiones sobre
personas. Lo que está en riesgo es llegar al 1 de diciembre con un modelo listo
para operar y sin el marco que lo habilita.

**Siguiente paso:** reunión con la Dirección Jurídica para resolver los siete
puntos de la sección 5. Es la única tarea del proyecto sin dependencias técnicas
y con fecha dura.

---

## Anexo · Dónde verificar cada afirmación

| Afirmación | Dónde |
|---|---|
| Seudonimización por HMAC | `piloto_fan/institucional.py` |
| Exclusión de estado como variable | `piloto_fan/tests/test_modelos_institucionales.py` |
| Segmentación sin usar el resultado | `tests/test_segmentation.py` |
| Aporte de variables socioeconómicas | `documentacion/datos/ablacion_piso.csv` |
| Trampas de datos encontradas | `documentacion/datos/hallazgo_trampas.csv` |
| Model cards | `piloto_fan/artifacts/<corrida>/*_model_card.json` |
| Veredicto de cada modelo | `data/results/modelos_resumen.json` |
| Qué sale publicado | `deploy/data/` y `deploy/README.md` |
