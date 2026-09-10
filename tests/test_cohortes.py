"""Tests del seguimiento longitudinal de cohortes."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from student_analytics.ingestion.cohortes import (
    adjuntar_cohorte,
    agregar_por_institucion,
    cargar_cohorte,
    seguir_cohorte,
)

ROOT = Path(__file__).resolve().parents[1]
SLIM = ROOT / "data/interim/matricula"
RESULTADOS = ROOT / "data/results/cohort_completion.parquet"


def _cohorte(n=10, **cambios):
    """Ingresantes 2010 de una carrera de 10 semestres (5 años nominales)."""
    base = pd.DataFrame({
        "mrun": [f"m{i}" for i in range(n)],
        "cod_inst": "1", "nomb_inst": "U. A", "cod_carrera": "C1",
        "nomb_carrera": "Enfermería", "area_conocimiento": "Salud",
        "anio_ing_carr_ori": 2010, "dur_total_carr": 10,
        "tipo_plan_carr": "Plan Regular", "gen_alu": "1", "nomb_sede": "Central",
        "nominal_anios": 5.0, "cohorte": 2010,
    })
    for k, v in cambios.items():
        base[k] = v
    return base


def _titulos(filas):
    return pd.DataFrame(filas, columns=["mrun", "cod_inst", "cod_carrera", "anio_titulo"])


def test_la_nominal_viene_en_semestres():
    """10 semestres son 5 años: con holgura 2, el caso cierra en 2017."""
    seg = seguir_cohorte(_cohorte(1), _titulos([("m0", "1", "C1", 2015)]),
                         2010, ultimo_anio=2025, holgura=2)
    assert seg.horizonte.iloc[0] == pytest.approx(2017)
    assert seg.anios_misma_carrera.iloc[0] == pytest.approx(5)
    assert seg.misma_carrera_en_plazo.iloc[0] == 1


def test_la_censura_saca_del_denominador_a_quien_no_alcanza_a_observarse():
    """Sin esto, las cohortes recientes parecen pésimas sólo por ser recientes."""
    seg = seguir_cohorte(_cohorte(1), _titulos([]), 2010, ultimo_anio=2016, holgura=2)
    assert seg.observable.iloc[0] == 0          # necesitaba datos hasta 2017
    seg = seguir_cohorte(_cohorte(1), _titulos([]), 2010, ultimo_anio=2017, holgura=2)
    assert seg.observable.iloc[0] == 1


def test_titularse_fuera_de_plazo_no_cuenta_en_plazo_pero_si_como_titulado():
    seg = seguir_cohorte(_cohorte(1), _titulos([("m0", "1", "C1", 2020)]),
                         2010, ultimo_anio=2025, holgura=2)
    assert seg.misma_carrera.iloc[0] == 1        # se tituló
    assert seg.misma_carrera_en_plazo.iloc[0] == 0   # pero a los 10 años


def test_los_tres_niveles_estan_anidados():
    """Cambiar de carrera o de universidad no puede reducir el conteo."""
    c = _cohorte(3)
    t = _titulos([
        ("m0", "1", "C1", 2015),   # misma carrera
        ("m1", "1", "C9", 2015),   # otra carrera, misma universidad
        ("m2", "7", "C9", 2015),   # otra universidad
    ])
    seg = seguir_cohorte(c, t, 2010, ultimo_anio=2025, holgura=2)
    assert seg.misma_carrera_en_plazo.sum() == 1
    assert seg.misma_universidad_en_plazo.sum() == 2
    assert seg.sistema_en_plazo.sum() == 3


def test_un_titulo_previo_al_ingreso_no_es_de_esta_cohorte():
    """Quien ya tenía un título aparece en la base con un año anterior."""
    seg = seguir_cohorte(_cohorte(2), _titulos([("m0", "1", "C1", 2008),
                                                ("m1", "1", "C1", 2010)]),
                         2010, ultimo_anio=2025, holgura=2)
    assert seg.misma_carrera.sum() == 0


def test_un_titulo_previo_no_esconde_al_posterior_que_si_cuenta():
    """Regresión: el filtro por año va ANTES del mínimo por clave.

    Quien ya traía un título de otra carrera tiene dos registros. Si se toma
    el mínimo primero, gana el título viejo, el estudiante se anula entero y
    el título que sí corresponde a esta cohorte desaparece. Como el nivel
    `sistema` agrega más títulos por persona, se envenena más que los otros
    y las tasas dejan de estar anidadas.
    """
    t = _titulos([
        ("m0", "9", "C7", 2006),   # título previo, en otra universidad
        ("m0", "1", "C1", 2015),   # el que corresponde a esta cohorte
    ])
    seg = seguir_cohorte(_cohorte(1), t, 2010, ultimo_anio=2025, holgura=2)
    assert seg.misma_carrera_en_plazo.iloc[0] == 1
    assert seg.sistema_en_plazo.iloc[0] == 1
    assert seg.anios_sistema.iloc[0] == pytest.approx(5)


def test_el_anidamiento_resiste_titulos_previos():
    c = _cohorte(3)
    t = _titulos([("m0", "5", "C4", 2005), ("m0", "1", "C1", 2015),
                  ("m1", "1", "C1", 2015), ("m2", "1", "C1", 2015)])
    seg = seguir_cohorte(c, t, 2010, ultimo_anio=2025, holgura=2)
    assert (seg.sistema_en_plazo >= seg.misma_universidad_en_plazo).all()
    assert (seg.misma_universidad_en_plazo >= seg.misma_carrera_en_plazo).all()


def test_se_toma_el_primer_titulo():
    seg = seguir_cohorte(_cohorte(1), _titulos([("m0", "1", "C1", 2019),
                                                ("m0", "1", "C1", 2015)]),
                         2010, ultimo_anio=2025, holgura=2)
    assert seg.anios_misma_carrera.iloc[0] == pytest.approx(5)


def test_la_tasa_se_calcula_solo_sobre_observables():
    c = pd.concat([_cohorte(4), _cohorte(4, dur_total_carr=30, nominal_anios=15.0)
                   .assign(mrun=[f"x{i}" for i in range(4)])], ignore_index=True)
    t = _titulos([("m0", "1", "C1", 2015), ("m1", "1", "C1", 2015)])
    seg = seguir_cohorte(c, t, 2010, ultimo_anio=2017, holgura=2)
    agg = agregar_por_institucion(seg, minimo=1, min_cobertura=0.5)
    assert agg.titulacion_cohorte_n.iloc[0] == 4          # los de 15 años no
    assert agg.titulacion_cohorte_carrera.iloc[0] == pytest.approx(.5)


def test_con_pocos_observables_se_anula_la_tasa_pero_queda_el_conteo():
    seg = seguir_cohorte(_cohorte(5), _titulos([]), 2010, ultimo_anio=2025)
    agg = agregar_por_institucion(seg, minimo=30)
    assert agg.titulacion_cohorte_n.iloc[0] == 5
    assert np.isnan(agg.titulacion_cohorte_carrera.iloc[0])
    assert np.isnan(agg.titulacion_cohorte_anios.iloc[0])


def test_la_cobertura_baja_anula_la_tasa_de_esa_institucion():
    """La censura no golpea parejo: se lleva primero las carreras largas.

    Una universidad con mucha carrera larga queda descrita sólo por sus
    carreras cortas y su tasa sube sin que haya titulado mejor. El piso
    nacional no la detecta, porque es una composición propia de ella.
    """
    largas = _cohorte(8, dur_total_carr=14, nominal_anios=7.0).assign(
        mrun=[f"L{i}" for i in range(8)])
    c = pd.concat([_cohorte(2), largas], ignore_index=True)
    t = _titulos([("m0", "1", "C1", 2015), ("m1", "1", "C1", 2015)])
    seg = seguir_cohorte(c, t, 2010, ultimo_anio=2017, holgura=2)
    assert seg.observable.sum() == 2                 # sólo las cortas
    laxo = agregar_por_institucion(seg, minimo=1, min_cobertura=0.1)
    assert laxo.titulacion_cohorte_carrera.iloc[0] == pytest.approx(1.0)
    estricto = agregar_por_institucion(seg, minimo=1, min_cobertura=0.9)
    assert np.isnan(estricto.titulacion_cohorte_carrera.iloc[0])
    # El conteo y la cobertura se conservan: dicen por qué se anuló.
    assert estricto.titulacion_cohorte_n.iloc[0] == 2
    assert estricto.titulacion_cohorte_cobertura.iloc[0] == pytest.approx(.2)


def _cohortes_tabla(filas):
    return pd.DataFrame(filas, columns=[
        "cohorte", "cod_inst", "nomb_inst", "titulacion_cohorte_n",
        "titulacion_cohorte_carrera", "titulacion_cohorte_universidad",
        "titulacion_cohorte_sistema", "titulacion_cohorte_anios"])


def test_el_perfil_recibe_la_ultima_cohorte_observable():
    """El perfil es 2024, que no se puede observar: se trae la más reciente."""
    cohortes = _cohortes_tabla([
        (2015, "1", "U. A", 100, .5, .55, .7, 5.0),
        (2017, "1", "U. A", 120, .6, .65, .75, 5.0)])
    perfil = pd.DataFrame({"cod_inst": ["1"], "nomb_inst": ["U. A"]})
    unido = adjuntar_cohorte(perfil, cohortes)
    assert unido.titulacion_cohorte_anio.iloc[0] == 2017
    assert unido.titulacion_cohorte_universidad.iloc[0] == pytest.approx(.65)


def test_la_cohorte_del_perfil_es_una_sola_para_todos():
    """Comparar una universidad en 2017 contra otra en 2018 no es un benchmark.

    En la cohorte más nueva sobreviven sólo las universidades de carreras
    cortas. Si cada una trajera su propia última cohorte, el ranking mezclaría
    años. Se exige que la cohorte elegida cubra al grueso del sistema.
    """
    cohortes = _cohortes_tabla([
        (2017, "1", "U. A", 100, .5, .55, .7, 5.0),
        (2017, "2", "U. B", 100, .4, .45, .6, 5.0),
        (2017, "3", "U. C", 100, .3, .35, .5, 5.0),
        # En 2018 sólo una alcanza a cerrarse.
        (2018, "1", "U. A", 100, np.nan, np.nan, np.nan, np.nan),
        (2018, "2", "U. B", 100, np.nan, np.nan, np.nan, np.nan),
        (2018, "3", "U. C", 100, .9, .95, .95, 4.0)])
    perfil = pd.DataFrame({"cod_inst": ["1", "2", "3"],
                           "nomb_inst": ["U. A", "U. B", "U. C"]})
    unido = adjuntar_cohorte(perfil, cohortes)
    assert unido.titulacion_cohorte_anio.eq(2017).all()
    # La U. C no se queda con su 95% de 2018, que no es comparable.
    assert unido.set_index("cod_inst").loc["3", "titulacion_cohorte_universidad"] \
        == pytest.approx(.35)


def test_sin_cohortes_el_perfil_pasa_intacto():
    perfil = pd.DataFrame({"cod_inst": ["1"], "nomb_inst": ["U. A"]})
    assert adjuntar_cohorte(perfil, pd.DataFrame()).equals(perfil)


# ----------------------------------------------------------------------
# Sobre las bases reales
# ----------------------------------------------------------------------
@pytest.mark.skipif(not (SLIM / "matricula_2010.parquet").exists(),
                    reason="matrícula histórica no descargada")
def test_la_cohorte_real_sale_del_anio_de_ingreso_no_del_periodo():
    c = cargar_cohorte(SLIM / "matricula_2010.parquet", 2010)
    assert c.anio_ing_carr_ori.eq(2010).all()
    assert c.tipo_plan_carr.eq("Plan Regular").all()
    assert len(c) > 50_000
    # Nadie ingresa a una carrera de duración nula o negativa.
    assert c.nominal_anios.dropna().gt(0).all()


@pytest.mark.skipif(not RESULTADOS.exists(), reason="cohortes no construidas")
def test_las_tasas_por_cohorte_son_plausibles_y_estan_anidadas():
    d = pd.read_parquet(RESULTADOS)
    for col in ("carrera", "universidad", "sistema"):
        assert d[f"titulacion_cohorte_{col}"].dropna().between(0, 1).all()
    sub = d.dropna(subset=["titulacion_cohorte_carrera",
                           "titulacion_cohorte_universidad",
                           "titulacion_cohorte_sistema"])
    assert (sub.titulacion_cohorte_universidad >= sub.titulacion_cohorte_carrera).all()
    assert (sub.titulacion_cohorte_sistema >= sub.titulacion_cohorte_universidad).all()
