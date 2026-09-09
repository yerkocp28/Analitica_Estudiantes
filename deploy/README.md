# Despliegue de Student Analytics UA

Entrada: `streamlit_app.py`. Entorno: Python **3.11**, dependencias de
`requirements.txt`. No se requieren secretos para consultar los datos incluidos.

## Streamlit Community Cloud

En https://share.streamlit.io/ selecciona **Create app** y configura:

| Campo | Valor |
|---|---|
| Repository | `yerkocp28/Analitica_Estudiantes` |
| Branch | `master` |
| Main file path | `streamlit_app.py` |
| Advanced settings → Python | `3.11` |

Pulsa **Deploy**. La URL final se confirma en Streamlit al crear la aplicación;
el subdominio opcional depende de disponibilidad. Los siguientes pushes a la rama
desplegada actualizan la app.

## Datos distribuidos

`deploy/data/` contiene una instantánea explícita de siete archivos:

- Perfiles universitarios y retención: agregados de matrícula pública Mineduc/SIES
  y recursos institucionales CNED.
- Manifiesto con trazabilidad y SHA-256 de los dos agregados.
- Métricas y predicciones sintéticas para la demostración del cockpit.
- Métricas y predicciones derivadas de OULAD, con identificadores públicos
  anonimizados de ese dataset; no son estudiantes de la Universidad Autónoma.

Fuentes: [Mineduc Datos Abiertos](https://datosabiertos.mineduc.cl/),
[CNED INDICES](https://cned.cl/institucional/bases-de-datos/),
[OULAD](https://analyse.kmi.open.ac.uk/open_dataset), distribuido bajo CC BY 4.0.
OULAD: Kuzilek, J., Hlosta, M. y Zdrahal, Z. (2017), *Open University Learning
Analytics dataset*. Las predicciones y variables semanales son transformaciones
del proyecto; no representan datos internos de la UA.

La app usa únicamente estos artefactos y no descarga las fuentes originales.
El entrenamiento del clustering es pequeño y queda en caché. Los resultados
semanales del cockpit están precalculados.

## Actualizar los datos

Tras regenerar localmente los resultados:

```bash
python scripts/package_deployment.py
python -m pytest tests/test_deployment.py -q
```

Revisa y versiona `deploy/data/` completo para mantener el manifiesto y los
agregados sincronizados. No se debe copiar todo `data/` al repositorio.

Prueba de la misma entrada que usa la nube:

```bash
streamlit run streamlit_app.py
```

Para seguir trabajando con resultados locales recién generados utiliza la
entrada habitual `streamlit run src/student_analytics/ui/app.py`. También puedes
definir `STUDENT_ANALYTICS_RESULTS_DIR` para seleccionar otro directorio.
