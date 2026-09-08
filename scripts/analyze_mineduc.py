"""Analisis de las bases abiertas de Mineduc/SIES.

    python scripts/analyze_mineduc.py

Produce dos resultados que el proyecto necesita antes de tener accesos:

1. RETENCION REAL DE LA UA, por sede y carrera de Administracion y Comercio.
   Se calcula cruzando Matricula 2024 con Matricula 2025 por MRUN. Reemplaza
   el ancla nacional generica de config/synthetic.yml por la cifra de la
   propia universidad.

2. PISO PREDICTIVO DE LAS VARIABLES PREVIAS AL INGRESO, a escala nacional.
   Se une PAES 2024 (notas de ensenanza media, NEM, ranking, puntajes) con
   quienes entraron a primer anio en 2024, y se predice si siguen
   matriculados en 2025. Es replicar el hallazgo de ULagos con ~200.000
   estudiantes en vez de dos cohortes de una universidad.

   Ese numero es el argumento central del proyecto: si las variables de
   ingreso solas alcanzan un AUC bajo, entonces el valor esta en los datos
   de comportamiento intrasemestral (Banner/Canvas) y no en la ficha de
   admision -- que es exactamente lo que se pide autorizar.

NOTA SOBRE MRUN: es un identificador ficticio pero estable entre bases y
anios. Permite seguir trayectorias, pero NO puede unirse al RUT que tiene
Banner. Esto sirve para calibrar y para argumentar, no para generar features
de estudiantes concretos de la UA.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from student_analytics.logging_setup import get_logger  # noqa: E402

log = get_logger("analyze_mineduc")

D = REPO_ROOT / "data" / "external" / "mineduc"
OUT = REPO_ROOT / "data" / "results"

MAT_COLS = ["mrun", "cat_periodo", "anio_ing_carr_ori", "nivel_global", "tipo_inst_1",
            "cod_inst", "nomb_inst", "nomb_sede", "cod_carrera", "nomb_carrera",
            "area_conocimiento", "gen_alu"]

PAES_COLS = ["MRUN", "COD_SEXO", "RBD", "DEPENDENCIA", "NOMBRE_REGION_EGRESO",
             "ANYO_DE_EGRESO", "PROMEDIO_NOTAS", "PTJE_NEM", "PTJE_RANKING",
             "CLEC_MAX", "MATE1_MAX", "MATE2_MAX", "HCSOC_MAX", "CIEN_MAX"]

# En estas bases el 0 significa "no rindio", no "saco cero". Tratarlo como
# puntaje real desplaza todas las medias y arruina cualquier modelo.
SCORE_COLS = ["PROMEDIO_NOTAS", "PTJE_NEM", "PTJE_RANKING",
              "CLEC_MAX", "MATE1_MAX", "MATE2_MAX", "HCSOC_MAX", "CIEN_MAX"]


def _find(sub: str, pattern: str) -> Path:
    hits = sorted((D / sub).rglob(pattern))
    if not hits:
        raise FileNotFoundError(f"No se encontro {pattern} en {D / sub}. "
                                "Corre antes: python scripts/download_mineduc.py")
    return hits[0]


# Los archivos pesan ~1 GB cada uno y la maquina tiene poca RAM libre, asi
# que los tipos importan: texto repetido como `category` y solo las columnas
# que se van a usar. Leerlos como str genera varios GB de objetos Python.
MAT_DTYPES = {
    "mrun": "float64",            # entero con posibles nulos
    "cat_periodo": "float32",
    "anio_ing_carr_ori": "float32",
    "cod_inst": "category", "cod_carrera": "category", "cod_sede": "category",
    "nivel_global": "category", "tipo_inst_1": "category", "gen_alu": "category",
    "nomb_inst": "category", "nomb_sede": "category", "nomb_carrera": "category",
    "area_conocimiento": "category",
}


def load_matricula(year: int, cols: list[str] | None = None) -> pd.DataFrame:
    """Lee la matricula de un anio con solo las columnas pedidas."""
    path = _find(f"matricula_{year}", "*.csv")
    cols = cols or MAT_COLS
    log.info("Leyendo %s (%.0f MB, %d columnas)", path.name,
             path.stat().st_size / 1e6, len(cols))
    df = pd.read_csv(path, sep=";", encoding="utf-8", usecols=cols,
                     dtype={k: v for k, v in MAT_DTYPES.items() if k in cols})
    log.info("  %s filas, %.0f MB en memoria", f"{len(df):,}",
             df.memory_usage(deep=True).sum() / 1e6)
    return df


def _clave(df: pd.DataFrame) -> pd.Series:
    """Clave persona+institucion+carrera como texto.

    Se usa una clave de pandas en vez de un set de tuplas de Python: con
    ~1,3 millones de filas la diferencia de memoria es de cientos de MB.
    """
    # Un puñado de filas viene sin mrun. Se marcan con -1 para que no
    # calcen con nadie, en vez de romper la conversion a entero.
    mrun = df["mrun"].fillna(-1).astype("int64").astype(str)
    return mrun + "|" + df["cod_inst"].astype(str) + "|" + df["cod_carrera"].astype(str)


def _decimal_comma(s: pd.Series) -> pd.Series:
    """Estas bases usan coma decimal (6,5 en vez de 6.5)."""
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


# ----------------------------------------------------------------------
# 1. Retencion real de la UA
# ----------------------------------------------------------------------
def retencion_ua(m24: pd.DataFrame, m25: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("1. RETENCION REAL — Universidad Autonoma de Chile (Matricula 2024 -> 2025)")
    print("=" * 78)

    es_ua = m24["nomb_inst"].str.contains("AUTONOMA", case=False, na=False)
    ua = m24[es_ua & (m24["nivel_global"] == "Pregrado")].copy()
    if ua.empty:
        print("  No se encontro la universidad en la base.")
        return
    print(f"  Institucion: {ua['nomb_inst'].iloc[0]}")
    print(f"  Matricula total de pregrado 2024: {len(ua):,}")
    print(f"  Sedes: {sorted(ua['nomb_sede'].dropna().unique())}")

    # Cohorte de primer anio: ingresaron en 2024.
    nuevos = ua[ua["anio_ing_carr_ori"] == 2024]
    print(f"  Cohorte de primer anio 2024: {len(nuevos):,}")

    # Retencion en el SISTEMA (sigue en cualquier institucion) y en el
    # PROGRAMA (misma institucion y carrera). La brecha entre ambas es la
    # movilidad, que no es lo mismo que desercion.
    en_sistema = pd.Index(m25["mrun"].dropna().unique())
    misma_carrera = pd.Index(_clave(m25.dropna(subset=["mrun"])).unique())

    def _tasas(df: pd.DataFrame) -> tuple[float, float, int]:
        # Sin mrun no se puede saber si la persona siguio: se excluye del
        # denominador en vez de contarla como desertora.
        df = df.dropna(subset=["mrun"])
        if df.empty:
            return float("nan"), float("nan"), 0
        sis = df["mrun"].isin(en_sistema).mean()
        prog = _clave(df).isin(misma_carrera).mean()
        return float(sis), float(prog), len(df)

    sis, prog, n = _tasas(nuevos)
    print(f"\n  Retencion al 2do anio (toda la UA, cohorte 2024, n={n:,}):")
    print(f"    misma carrera y universidad : {prog:6.1%}   <- definicion SIES")
    print(f"    en cualquier institucion    : {sis:6.1%}   (incluye movilidad)")

    # Foco del piloto: Administracion y Comercio.
    adm = nuevos[nuevos["area_conocimiento"].str.contains(
        "Administraci", case=False, na=False)]
    sis_a, prog_a, n_a = _tasas(adm)
    print(f"\n  Solo area Administracion y Comercio (n={n_a:,}):")
    print(f"    misma carrera y universidad : {prog_a:6.1%}")
    print(f"    en cualquier institucion    : {sis_a:6.1%}")

    if n_a:
        print("\n  Por sede:")
        for sede, g in adm.groupby("nomb_sede"):
            s, p, k = _tasas(g)
            if k >= 20:
                print(f"    {str(sede)[:34]:34s} n={k:5,}  misma carrera {p:6.1%}")

        print("\n  Por carrera (n>=30):")
        for carr, g in adm.groupby("nomb_carrera"):
            s, p, k = _tasas(g)
            if k >= 30:
                print(f"    {str(carr)[:40]:40s} n={k:5,}  misma carrera {p:6.1%}")

    # Referencia del sistema, para leer las cifras en contexto.
    univ = m24[(m24["nivel_global"] == "Pregrado")
               & (m24["tipo_inst_1"] == "Universidades")
               & (m24["anio_ing_carr_ori"] == 2024)]
    s_u, p_u, n_u = _tasas(univ)
    print(f"\n  Referencia — todas las universidades del pais (n={n_u:,}):")
    print(f"    misma carrera y universidad : {p_u:6.1%}")
    print(f"    en cualquier institucion    : {s_u:6.1%}")


# ----------------------------------------------------------------------
# 2. Piso predictivo de las variables previas al ingreso
# ----------------------------------------------------------------------
def piso_predictivo(m24: pd.DataFrame, m25: pd.DataFrame) -> None:
    print("\n" + "=" * 78)
    print("2. PISO PREDICTIVO DE LAS VARIABLES PREVIAS AL INGRESO (nacional)")
    print("=" * 78)

    path = _find("paes_2024_puntajes", "*.csv")
    log.info("Leyendo %s (%.0f MB)", path.name, path.stat().st_size / 1e6)
    paes = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=PAES_COLS,
                       dtype=str, low_memory=False)
    paes["mrun"] = pd.to_numeric(paes["MRUN"], errors="coerce")
    for c in SCORE_COLS:
        paes[c] = _decimal_comma(paes[c])
        # 0 = no rindio esa prueba. Es ausencia de dato, no un puntaje.
        paes.loc[paes[c] <= 0, c] = np.nan
    log.info("  %s inscritos PAES 2024", f"{len(paes):,}")

    # Poblacion: entraron a primer anio de pregrado en 2024.
    nuevos = m24[(m24["nivel_global"] == "Pregrado")
                 & (m24["anio_ing_carr_ori"] == 2024)][
        ["mrun", "cod_inst", "cod_carrera", "tipo_inst_1", "area_conocimiento"]
    ].drop_duplicates("mrun")

    df = nuevos.merge(paes, on="mrun", how="inner")
    print(f"\n  Cruce por MRUN: {len(df):,} de {len(nuevos):,} ingresantes 2024 "
          f"({len(df) / len(nuevos):.0%}) aparecen en PAES 2024")
    if len(df) < 5000:
        print("  Cruce insuficiente; se omite el modelo.")
        return

    misma = pd.Index(_clave(m25.dropna(subset=["mrun"])).unique())
    df["retenido"] = _clave(df).isin(misma)
    df["desertor"] = (~df["retenido"]).astype(int)
    print(f"  No retenidos en la misma carrera al 2025: {df['desertor'].mean():.1%}")

    feats = SCORE_COLS + ["sexo_f", "dep_municipal", "dep_particular_sub",
                          "dep_particular_pag", "anios_desde_egreso"]
    df["sexo_f"] = (df["COD_SEXO"] == "2").astype(int)
    dep = pd.to_numeric(df["DEPENDENCIA"], errors="coerce")
    df["dep_municipal"] = (dep == 1).astype(int)
    df["dep_particular_sub"] = (dep == 2).astype(int)
    df["dep_particular_pag"] = (dep == 3).astype(int)
    df["anios_desde_egreso"] = 2024 - pd.to_numeric(df["ANYO_DE_EGRESO"], errors="coerce")

    def _ajustar(sub: pd.DataFrame, etiqueta: str) -> tuple[float, pd.Series | None]:
        if len(sub) < 5000 or sub["desertor"].nunique() < 2:
            print(f"\n  {etiqueta}: n={len(sub):,}, insuficiente")
            return float("nan"), None
        X = sub[feats]
        X = X.fillna(X.median()).fillna(0).to_numpy(dtype=float)
        y = sub["desertor"].to_numpy()
        Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=42,
                                              stratify=y)
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"))
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:, 1]
        auc = roc_auc_score(yte, p)

        def captura(pct: float) -> float:
            k = max(1, int(round(len(p) * pct)))
            return float(yte[np.argsort(-p)[:k]].sum() / yte.sum())

        print(f"\n  {etiqueta}")
        print(f"    n_train={len(ytr):,}  n_test={len(yte):,}  "
              f"prevalencia={yte.mean():.1%}")
        print(f"    AUC = {auc:.3f}")
        for pct in (0.05, 0.10, 0.20, 0.50):
            print(f"    top {pct:4.0%} -> captura {captura(pct):5.1%} de los no retenidos"
                  f"   ({captura(pct) / pct:.2f}x sobre azar)")
        return auc, pd.Series(model[-1].coef_[0], index=feats).sort_values()

    print("\n  Modelo: regresion logistica, SOLO variables previas al ingreso")
    print("  (NEM, ranking, notas EM, puntajes PAES, dependencia, sexo, anio egreso)")

    auc, coefs = _ajustar(df, "TODO EL SISTEMA (universidades, IP y CFT)")

    # La UA es una universidad: el numero directamente comparable excluye
    # IP y CFT, cuyas dinamicas de admision y permanencia son distintas.
    solo_u = df[df["tipo_inst_1"] == "Universidades"]
    auc_u, _ = _ajustar(solo_u, "SOLO UNIVERSIDADES  <- el comparable para la UA")

    if coefs is not None:
        print("\n  Coeficientes del modelo de sistema "
              "(positivo = mas riesgo de no continuar):")
        for name, v in coefs.items():
            print(f"    {name:22s} {v:+.3f}")

    print("\n  " + "-" * 74)
    print("  LECTURA: este es el PISO. Es lo que se puede predecir sin mirar una")
    print("  sola semana de comportamiento universitario. Todo lo que Banner y")
    print("  Canvas aporten por encima de esta linea es el valor del proyecto.")
    print("  (Split aleatorio, no temporal: hay una sola cohorte. Si acaso, eso")
    print("  favorece al modelo, asi que el piso real es aun mas bajo.)")

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"poblacion": "sistema", "n": len(df), "auc": auc},
        {"poblacion": "solo_universidades", "n": len(solo_u), "auc": auc_u},
    ]).to_parquet(OUT / "piso_preingreso_nacional.parquet", index=False)


def main() -> int:
    # De 2025 solo se necesita saber quien sigue y en que carrera.
    m25 = load_matricula(2025, ["mrun", "cod_inst", "cod_carrera"])
    m24 = load_matricula(2024)
    retencion_ua(m24, m25)
    piso_predictivo(m24, m25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
