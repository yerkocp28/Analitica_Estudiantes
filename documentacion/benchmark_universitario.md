# Benchmark universitario de la Universidad Autónoma

## Propósito y alcance

Comparar la continuidad de las cohortes de ingreso a carrera de la UA con otras
universidades, seleccionando comparadores por perfil antes de observar sus
resultados. Es un benchmark descriptivo de cohortes, no una clasificación global
de calidad institucional ni un estimador causal del efecto de una universidad.

La app abre en **Benchmark UA**. Conserva las secciones de alerta temprana y
exploración nacional de retención. El clustering es institucional: no segmenta
estudiantes ni reemplaza los modelos de riesgo académico del cockpit.

## Fuentes y universo

- Matrículas públicas Mineduc/SIES 2023, 2024 y 2025 ya descargadas en
  `data/external/mineduc/matricula_<año>`.
- Cohortes de ingreso 2023 y 2024: `nivel_global = Pregrado`,
  `tipo_inst_1 = Universidades`, `anio_ing_carr_ori = año de cohorte`.
- El grano es la inscripción de ingreso a carrera: una persona puede contribuir
  más de una inscripción. No equivale necesariamente a primer ingreso al sistema.
- Se eliminan duplicados según las mismas columnas canónicas utilizadas por
  `build_retention.py`. Se comprueba que los totales institucionales de los
  perfiles coincidan con `n + sin_mrun` de los resultados de retención.
- El perfil incluye inscripciones sin MRUN. El seguimiento excluye esas filas
  del denominador y conserva su número como indicador de cobertura.
- Los archivos exportados contienen agregados institucionales, sin MRUN.

## Variables de comparabilidad

| Bloque | Variables | Peso predeterminado |
|---|---|---:|
| Escala | log(1 + inscripciones de ingreso), log(1 + sedes distintas por código) | 25% |
| Áreas | Proporción de inscripciones por área de conocimiento | 25% |
| Regiones | Proporción de inscripciones por región de sede | 25% |
| Docencia | Proporciones por jornada y por modalidad | 25% |

No se incorporan resultados de continuidad, notas finales ni características
posteriores al seguimiento. Tampoco se incorporan selectividad PAES, acreditación,
recursos económicos, investigación o perfil socioeconómico: los pares son
similares en las dimensiones disponibles, no necesariamente en todas las relevantes.

Para los bloques de proporciones se aplica raíz cuadrada. Cada bloque se centra
y divide por la raíz de su varianza total entre universidades; las variables de
escala se estandarizan individualmente antes de normalizar el bloque. Se multiplica
por la raíz del peso configurado. Así cada bloque variable contribuye el mismo
peso a la varianza total con los valores predeterminados. Un bloque constante no
aporta distancia. No se estandariza individualmente cada categoría, para evitar
amplificar artificialmente categorías raras.

La distancia es euclidiana en el espacio completo transformado. No se expresa
como probabilidad o porcentaje. La categoría de datos ausentes se conserva como
`Sin información` y se publican coberturas. La PCA de dos componentes solo sirve
para visualizar; se indica qué proporción de la variación representa.

## Agrupamiento y elección de pares

Las universidades con al menos 100 inscripciones de ingreso entran al ajuste,
con igual peso institucional. Los valores se configuran en `config/benchmark.yml`.

Se contrastan 15 candidatos: K-means, mezcla gaussiana con covarianza diagonal y
jerárquico Ward, para k de 2 a 6. K-means utiliza 20 inicializaciones; mezcla
gaussiana, 5 y regularización 0,0001. Semilla 42 para las alternativas aleatorias.
Se descartan particiones con menos de tres universidades en algún grupo, número
efectivo de grupos distinto a k, silueta indefinida o mezcla sin convergencia.
Entre las restantes se elige la mayor silueta, favoreciendo menor k en un empate.
Si no existe una solución admisible, permanecen los vecinos y selección manual;
no se inventa un cluster.

La silueta es un diagnóstico interno de separación, no validación sustantiva de
comparabilidad. La app advierte separación débil por debajo de 0,25 como umbral
orientativo; no se trata de una regla universal. También muestra tamaños de
grupos, explicaciones por bloque y estabilidad temporal. Referencia técnica:
[evaluación de clustering de scikit-learn](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation).

Tres modos de selección:

1. **Más cercanas por perfil**: cinco vecinos por defecto, cantidad configurable;
   pueden pertenecer a distintos clusters. Es la vista inicial para una comparación
   acotada y fácil de revisar.
2. **Mismo cluster que la UA**: todas las universidades del grupo seleccionado por
   el modelo, excepto la propia UA.
3. **Selección manual**: pares elegidos por el usuario entre los perfiles elegibles.

La UA siempre es la referencia. Cambiar el área o definición del resultado no
recalcula la selección de pares. Cambiar la cohorte sí modifica los perfiles,
las escalas, la partición y los vecinos.

La estabilidad se informa mediante coincidencia de los vecinos más cercanos
entre las dos cohortes y el índice Rand ajustado de las particiones sobre
universidades comunes. No se presupone igualdad de etiquetas de cluster entre
años. La PCA tampoco se usa para establecer correspondencias temporales.

## Resultados comparados y referencias

Se observa continuidad al año siguiente:

- **Misma carrera y universidad**: coincidencia de MRUN, código de institución y
  código de carrera. No exige la misma sede.
- **Misma universidad**: coincidencia de MRUN y código de institución.
- **Sistema**: el MRUN aparece en cualquier institución, incluidos IP y CFT.

No observar matrícula al año siguiente no prueba abandono definitivo. Cambios de
códigos pueden reducir artificialmente la continuidad en carrera.

Los promedios se obtienen sumando numeradores y denominadores, no promediando
porcentajes institucionales. La referencia de pares excluye a la UA. La referencia
del resto del sistema incluye todas las universidades con seguimiento en el área
y cohorte, incluso las que no cumplen el mínimo de tamaño para clustering, y
excluye a la UA.

**Ajuste por composición de áreas.** La referencia ajustada aplica las proporciones
de inscripciones de cada área de la UA a las tasas de los pares de esas áreas.
Solo utiliza áreas con denominador positivo en ambos lados. Publica qué fracción
de la UA queda cubierta y compara con la retención UA recalculada en ese mismo
soporte común. Una ausencia no se imputa como cero. No controla otras diferencias
de selección, contexto o recursos.

En la evolución se conserva el listado de pares seleccionado en la cohorte del
filtro. La ponderación cambia con los tamaños observados de cada año. Se informa
cuántos pares tienen datos; dos cohortes no permiten inferir una tendencia de
largo plazo. El cuadro por áreas muestra todas las áreas de la UA, aunque se haya
seleccionado un área en el filtro superior.

Los porcentajes provienen de registros administrativos. No se presentan intervalos
de muestreo como si se tratara de una muestra aleatoria; esto no elimina errores
de cobertura, registro, codificación o incertidumbre al generalizar a futuras cohortes.

## Reproducibilidad y validación

```bash
python scripts/build_retention.py
python scripts/build_benchmark.py
streamlit run src/student_analytics/ui/app.py
python -m pytest -q
```

`benchmark_profiles.parquet` conserva los perfiles por universidad y cohorte.
`benchmark_manifest.json` registra fuente, tamaño y fecha de archivo, generación
UTC y huellas SHA-256 de perfiles y retención. La app rechaza combinaciones de
artefactos cuyas huellas no coincidan. Las huellas corresponden a los artefactos,
no a los CSV originales; estos se identifican por nombre, tamaño y fecha.

Los modelos se calculan y almacenan en caché de Streamlit por perfiles y
configuración; no dependen de predicciones semanales. La ficha descargable
registra configuración, versión de scikit-learn, variables por bloque, modelo,
cohorte, filtros y códigos de los pares. La tabla CSV incluye el contexto de
comparación. Para reproducir exactamente una ejecución se deben conservar los
artefactos, la configuración, esta versión del código y el entorno de librerías.

Las pruebas cubren independencia respecto de outcomes, reproducibilidad,
exclusión de la UA de referencias y vecinos, denominadores, estandarización por
áreas, soporte común incompleto, cohortes insuficientes y navegación de la app.
