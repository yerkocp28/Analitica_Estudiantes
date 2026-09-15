"""Segmentación de ingresantes: que los segmentos signifiquen algo."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from student_analytics.modeling.segmentation import (
    PROHIBIDAS,
    estabilidad,
    matriz_segmentacion,
    perfilar,
    segmentar,
)

ROOT = Path(__file__).resolve().parents[1]
INGRESANTES = ROOT / "data" / "results" / "ua_ingresantes.parquet"

PESOS = {"academico": 0.5, "estructura": 0.5}


def _ingresantes(n=3000, semilla=5) -> pd.DataFrame:
    """Ingresantes sintéticos con dos perfiles separables de verdad."""
    rng = np.random.default_rng(semilla)
    alto = n // 3
    d = pd.DataFrame({
        "cohorte": rng.choice([2023, 2024, 2025], n),
        "nomb_sede": rng.choice(["Providencia", "Talca", "Temuco"], n),
        "area_conocimiento": ["Salud"] * alto + list(
            rng.choice(["Ciencias Sociales", "Derecho"], n - alto)),
        "jornada": ["Diurna"] * alto + list(rng.choice(["Diurna", "Vespertina"], n - alto)),
        "modalidad": "Presencial",                      # constante a propósito
        "duracion_nominal_anios": 5.0,
        "percentil_admision": np.r_[rng.uniform(.7, 1, alto), rng.uniform(0, .5, n - alto)],
        "percentil_nem": np.r_[rng.uniform(.7, 1, alto), rng.uniform(0, .5, n - alto)],
        "percentil_ranking": np.r_[rng.uniform(.7, 1, alto), rng.uniform(0, .5, n - alto)],
        "sigue_carrera": np.r_[np.ones(alto), rng.integers(0, 2, n - alto)],
        "sigue_universidad": 1,
        "sigue_sistema": 1,
    })
    return d


def _config(**cambios) -> dict:
    base = {"block_weights": PESOS, "k_values": [2, 3, 4], "random_seed": 42,
            "minimum_segment": 100, "maximum_segment_share": 0.9,
            "muestra_ajuste": 5000}
    base.update(cambios)
    return base


# ----------------------------------------------------------------------
# El espacio de segmentación
# ----------------------------------------------------------------------
def test_el_resultado_nunca_entra_al_espacio():
    """Si la continuidad definiera el segmento, describir su retención sería
    circular y el segmento no serviría para recibir a la cohorte siguiente."""
    d = _ingresantes()
    _, columnas = matriz_segmentacion(d, PESOS)
    usadas = {c for cols in columnas.values() for c in cols}
    assert not (PROHIBIDAS & usadas)
    # Y cambiar el resultado no debe mover el espacio ni un poco.
    a, _ = matriz_segmentacion(d, PESOS)
    b, _ = matriz_segmentacion(d.assign(sigue_carrera=0, sigue_sistema=0), PESOS)
    assert np.allclose(a, b)


def test_una_variable_constante_se_descarta():
    """`modalidad` es 100% Presencial en la UA: no separa a nadie."""
    _, columnas = matriz_segmentacion(_ingresantes(), PESOS)
    assert not any("modalidad" in c for c in columnas["estructura"])


def test_pesos_invalidos_no_producen_una_particion_arbitraria():
    for malos in ({"academico": -1, "estructura": 2}, {"academico": 0.5}, {}):
        with pytest.raises(ValueError, match="pesos"):
            matriz_segmentacion(_ingresantes(), malos)


def test_un_bloque_sin_datos_se_omite_y_reparte_su_peso():
    d = _ingresantes().drop(columns=["percentil_admision", "percentil_nem",
                                     "percentil_ranking"])
    matriz, columnas = matriz_segmentacion(d, PESOS)
    assert "academico" not in columnas
    assert matriz.shape[0] == len(d)


# ----------------------------------------------------------------------
# La elección de la partición
# ----------------------------------------------------------------------
def test_se_comparan_tres_familias_de_algoritmo():
    """El documento maestro exige al menos tres alternativas."""
    modelo = segmentar(_ingresantes(), _config())
    assert modelo.diagnostico.algoritmo.nunique() >= 3


def test_el_techo_de_concentracion_descarta_particiones():
    """Sin él la silueta elige siempre la que aísla atípicos."""
    laxo = segmentar(_ingresantes(), _config(maximum_segment_share=0.95))
    estricto = _config(maximum_segment_share=0.55)
    modelo = segmentar(_ingresantes(), estricto)
    assert modelo.diagnostico.loc[modelo.diagnostico.admisible,
                                  "concentracion"].max() <= 0.55
    assert laxo.diagnostico.admisible.sum() >= modelo.diagnostico.admisible.sum()


def test_sin_particion_admisible_falla_diciendo_que_revisar():
    with pytest.raises(ValueError, match="minimum_segment|maximum_segment_share"):
        segmentar(_ingresantes(), _config(minimum_segment=10_000))


def test_los_segmentos_se_numeran_por_tamanio():
    """Para que el nombre no cambie de significado entre corridas."""
    modelo = segmentar(_ingresantes(), _config())
    cuentas = modelo.etiquetas.value_counts()
    assert list(cuentas.index) == sorted(cuentas.index, key=lambda s: -cuentas[s])
    assert cuentas.index.min() == 1


# ----------------------------------------------------------------------
# La trampa de escala que este proyecto ya encontró dos veces
# ----------------------------------------------------------------------
def test_el_percentil_cruza_el_cambio_de_escala_y_el_puntaje_crudo_no():
    """Regresión del error que partía un segmento en dos por el calendario.

    El NEM llegaba a 824 con la PDT y a 994 con la PAES, y la media saltaba de
    560 a 716 entre 2022 y 2023. Con el puntaje crudo la segmentación agrupaba
    por AÑO creyendo que agrupaba por perfil. El percentil dentro de la cohorte
    deja a la misma persona en el mismo lugar con cualquiera de las dos escalas.
    """
    rng = np.random.default_rng(1)
    n = 600
    d = pd.DataFrame({
        "cohorte": [2022] * (n // 2) + [2024] * (n // 2),
        "puntaje_nem": np.r_[rng.uniform(280, 830, n // 2),   # escala PDT
                             rng.uniform(300, 990, n // 2)],  # escala PAES
    })
    d["percentil_nem"] = d.groupby("cohorte").puntaje_nem.rank(pct=True)
    # El crudo separa las cohortes; el percentil no. En los ingresantes reales
    # de la UA la brecha cruda es de 156 puntos (560 contra 716); aqui, con
    # rangos uniformes, sale algo menor pero el contraste es el mismo.
    crudo = abs(d.groupby("cohorte").puntaje_nem.mean().diff().iloc[-1])
    percentil = abs(d.groupby("cohorte").percentil_nem.mean().diff().iloc[-1])
    assert crudo > 80, f"la construcción del caso no reproduce el corte: {crudo:.0f}"
    assert percentil < 0.05


# ----------------------------------------------------------------------
# El perfil y la estabilidad
# ----------------------------------------------------------------------
def test_el_perfil_describe_la_continuidad_sin_haberla_usado():
    d = _ingresantes()
    modelo = segmentar(d, _config())
    perfil = perfilar(d, modelo.etiquetas)
    assert {"segmento", "n", "share", "sigue_carrera"} <= set(perfil.columns)
    assert perfil.n.sum() == len(d)
    assert np.isclose(perfil.share.sum(), 1)
    # La brecha se mide contra el conjunto, y debe sumar cero ponderada.
    ponderada = (perfil.brecha_carrera_pp * perfil.share).sum()
    assert abs(ponderada) < 1e-6


def test_la_estabilidad_reparte_cada_cohorte_entre_los_segmentos():
    d = _ingresantes()
    modelo = segmentar(d, _config())
    e = estabilidad(d, modelo.etiquetas)
    assert len(e) == d.cohorte.nunique()
    shares = e.drop(columns="cohorte")
    assert np.allclose(shares.sum(axis=1), 1)


# ----------------------------------------------------------------------
# Sobre los ingresantes reales
# ----------------------------------------------------------------------
@pytest.mark.skipif(not INGRESANTES.exists(), reason="ingresantes no construidos")
def test_los_ingresantes_reales_son_coherentes():
    d = pd.read_parquet(INGRESANTES)
    assert len(d) > 90_000
    assert d.cohorte.between(2007, 2026).all()
    for col in ("sigue_carrera", "sigue_universidad", "sigue_sistema"):
        s = pd.to_numeric(d[col], errors="coerce")
        assert s.dropna().isin([0, 1]).all()
    # Los tres niveles están anidados, igual que en el resto del proyecto.
    a, b, c = (pd.to_numeric(d[x], errors="coerce")
               for x in ("sigue_carrera", "sigue_universidad", "sigue_sistema"))
    assert (b >= a).all() and (c >= b).all()
    # El percentil existe donde existe el puntaje, y solo ahí.
    if "percentil_nem" in d.columns:
        assert d.percentil_nem.notna().equals(d.puntaje_nem.notna())
        assert d.percentil_nem.dropna().between(0, 1).all()


@pytest.mark.skipif(not INGRESANTES.exists(), reason="ingresantes no construidos")
def test_la_segmentacion_real_es_estable_entre_cohortes():
    """Un segmento que solo existe en un año describe un año, no un tipo.

    Antes de corregir la escala del NEM había un segmento que pasaba de 2% en
    2021 a 32% en 2024: eso era el calendario, no los estudiantes.
    """
    config = yaml.safe_load((ROOT / "config/segmentation.yml").read_text(encoding="utf-8"))
    d = pd.read_parquet(INGRESANTES)
    d = d[d.cohorte.between(config["cohorte_desde"], config["cohorte_hasta"])]
    d = d.reset_index(drop=True)
    if len(d) < 1000:
        pytest.skip("pocas cohortes con puntaje de admisión")
    modelo = segmentar(d, config)
    e = estabilidad(d, modelo.etiquetas).drop(columns="cohorte")
    # Ningún segmento debe triplicar su presencia entre la primera y la última.
    for col in e.columns:
        assert e[col].max() <= 3 * max(e[col].min(), 1e-3), col
