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

    # Los desgloses se guardan como artefacto para que el informe
    # metodologico los cite sin recalcular sobre 3 GB de CSV.
    doc = REPO_ROOT / "documentacion" / "datos"
    doc.mkdir(parents=True, exist_ok=True)
    filas_sede, filas_carr = [], []

    if n_a:
        print("\n  Por sede:")
        for sede, g in adm.groupby("nomb_sede", observed=True):
            s, p, k = _tasas(g)
            if k >= 20:
                print(f"    {str(sede)[:34]:34s} n={k:5,}  misma carrera {p:6.1%}")
                filas_sede.append({"sede": str(sede), "n": k,
                                   "retencion_misma_carrera": round(p, 6),
                                   "retencion_sistema": round(s, 6)})

        print("\n  Por carrera (n>=30):")
        for carr, g in adm.groupby("nomb_carrera", observed=True):
            s, p, k = _tasas(g)
            if k >= 30:
                print(f"    {str(carr)[:40]:40s} n={k:5,}  misma carrera {p:6.1%}")
                filas_carr.append({"carrera": str(carr), "n": k,
                                   "retencion_misma_carrera": round(p, 6),
                                   "retencion_sistema": round(s, 6)})

    pd.DataFrame(filas_sede).to_csv(doc / "ua_retencion_por_sede.csv",
                                    index=False, encoding="utf-8")
    pd.DataFrame(filas_carr).to_csv(doc / "ua_retencion_por_carrera.csv",
                                    index=False, encoding="utf-8")

    # Referencia del sistema, para leer las cifras en contexto.
    univ = m24[(m24["nivel_global"] == "Pregrado")
               & (m24["tipo_inst_1"] == "Universidades")
               & (m24["anio_ing_carr_ori"] == 2024)]
    s_u, p_u, n_u = _tasas(univ)
    print(f"\n  Referencia — todas las universidades del pais (n={n_u:,}):")
    print(f"    misma carrera y universidad : {p_u:6.1%}")
    print(f"    en cualquier institucion    : {s_u:6.1%}")

    pd.DataFrame([
        {"poblacion": "UA - Administracion y Comercio", "n": n_a,
         "retencion_misma_carrera": round(prog_a, 6),
         "retencion_sistema": round(sis_a, 6)},
        {"poblacion": "UA - todo el pregrado", "n": n,
         "retencion_misma_carrera": round(prog, 6),
         "retencion_sistema": round(sis, 6)},
        {"poblacion": "Todas las universidades del pais", "n": n_u,
         "retencion_misma_carrera": round(p_u, 6),
         "retencion_sistema": round(s_u, 6)},
    ]).to_csv(doc / "ua_retencion_resumen.csv", index=False, encoding="utf-8")


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

    feats = SCORE_COLS + ["sexo_f", "dep_publico", "dep_particular_sub",
                          "dep_particular_pag", "dep_adm_delegada",
                          "anios_desde_egreso"]
    df["sexo_f"] = (df["COD_SEXO"] == "2").astype(int)
    dep = pd.to_numeric(df["DEPENDENCIA"], errors="coerce")
    # Codigos oficiales del diccionario de la base (ER_Alumnos_PAES):
    # 1 Corp. Municipal, 2 Municipal, 3 Part. Subvencionado,
    # 4 Part. Pagado, 5 Corp. de Adm. Delegada, 6 Servicio Local.
    # Las tres formas de administracion publica (1, 2 y 6) se agrupan.
    df["dep_publico"] = dep.isin([1, 2, 6]).astype(int)
    df["dep_particular_sub"] = (dep == 3).astype(int)
    df["dep_particular_pag"] = (dep == 4).astype(int)
    df["dep_adm_delegada"] = (dep == 5).astype(int)
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


# ----------------------------------------------------------------------
# 3. Validacion temporal del piso: cohorte 2023 entrena, 2024 testea
# ----------------------------------------------------------------------
COHORT_COLS = ["mrun", "cod_inst", "cod_carrera", "anio_ing_carr_ori",
               "nivel_global", "tipo_inst_1"]


def load_paes(year: int) -> pd.DataFrame | None:
    """Lee PAES de un anio y normaliza nombres y tipos.

    El nombre de la prueba y de los archivos cambio entre anios, pero las
    columnas relevantes se mantienen. Se toman las que existan.
    """
    try:
        path = _find(f"paes_{year}_puntajes", "*.csv")
    except FileNotFoundError:
        log.warning("PAES %s no descargado; se omite la validacion temporal", year)
        return None
    head = pd.read_csv(path, sep=";", encoding="utf-8-sig", nrows=1)
    cols = [c for c in PAES_COLS if c in head.columns]
    faltan = set(PAES_COLS) - set(cols)
    if faltan:
        log.warning("PAES %s no trae: %s", year, sorted(faltan))
    log.info("Leyendo PAES %s (%.0f MB)", year, path.stat().st_size / 1e6)
    df = pd.read_csv(path, sep=";", encoding="utf-8-sig", usecols=cols,
                     dtype=str, low_memory=False)
    df["mrun"] = pd.to_numeric(df["MRUN"], errors="coerce")
    for c in SCORE_COLS:
        if c in df.columns:
            df[c] = _decimal_comma(df[c])
            # 0 = no rindio esa prueba: ausencia de dato, no un puntaje.
            df.loc[df[c] <= 0, c] = np.nan
        else:
            df[c] = np.nan
    return df


def _construir_cohorte(mat: pd.DataFrame, mat_sig: pd.DataFrame,
                       paes: pd.DataFrame, year: int) -> pd.DataFrame:
    """Ingresantes de `year` con features PAES y si continuaron al year+1."""
    nuevos = mat[(mat["nivel_global"] == "Pregrado")
                 & (mat["anio_ing_carr_ori"] == year)][
        ["mrun", "cod_inst", "cod_carrera", "tipo_inst_1"]].drop_duplicates("mrun")
    df = nuevos.merge(paes, on="mrun", how="inner")
    misma = pd.Index(_clave(mat_sig.dropna(subset=["mrun"])).unique())
    df["desertor"] = (~_clave(df).isin(misma)).astype(int)

    df["sexo_f"] = (df["COD_SEXO"] == "2").astype(int) if "COD_SEXO" in df else 0
    dep = pd.to_numeric(df.get("DEPENDENCIA"), errors="coerce")
    # Codigos oficiales del diccionario de la base (ER_Alumnos_PAES):
    # 1 Corp. Municipal, 2 Municipal, 3 Part. Subvencionado,
    # 4 Part. Pagado, 5 Corp. de Adm. Delegada, 6 Servicio Local.
    # Las tres formas de administracion publica (1, 2 y 6) se agrupan.
    df["dep_publico"] = dep.isin([1, 2, 6]).astype(int)
    df["dep_particular_sub"] = (dep == 3).astype(int)
    df["dep_particular_pag"] = (dep == 4).astype(int)
    df["dep_adm_delegada"] = (dep == 5).astype(int)
    df["anios_desde_egreso"] = year - pd.to_numeric(
        df.get("ANYO_DE_EGRESO"), errors="coerce")
    return df


# ----------------------------------------------------------------------
# 3b. Trayectoria escolar previa al ingreso
# ----------------------------------------------------------------------
# Tasas de asistencia mensuales, de marzo a diciembre.
MESES_ASIS = [f"tasa_asistencia_{m}" for m in range(3, 13)]

FEATS_ASISTENCIA = ["asis_em_anual", "asis_em_pendiente", "asis_em_min",
                    "asis_em_ultimo_trimestre"]
FEATS_SOCIO = ["quintil_ingreso", "decil_dfe", "gratuidad", "fscu",
               "beca_arancel", "prioritario", "preferente"]


def cargar_asistencia_escolar(year: int) -> pd.DataFrame | None:
    """Asistencia de ensenanza media del anio `year`, por MRUN.

    Deriva cuatro features del panel mensual. La mas interesante no es el
    nivel sino la PENDIENTE: un estudiante que baja de 95% en marzo a 70%
    en noviembre ya se estaba desenganchando antes de matricularse. Es el
    mismo fenomeno que el proyecto quiere detectar dentro del semestre.
    """
    try:
        path = _find(f"asistencia_{year}", "*.csv")
    except FileNotFoundError:
        log.warning("Asistencia escolar %s no descargada", year)
        return None
    cols = ["mrun", "tasa_asistencia_anual"] + MESES_ASIS
    log.info("Leyendo asistencia escolar %s (%.0f MB)", year, path.stat().st_size / 1e6)
    df = pd.read_csv(path, sep=";", encoding="utf-8", usecols=cols,
                     dtype={"mrun": "float64"}, low_memory=False)

    for c in ["tasa_asistencia_anual"] + MESES_ASIS:
        df[c] = _decimal_comma(df[c])

    mensual = df[MESES_ASIS]
    # Pendiente por minimos cuadrados sobre los meses observados. Un valor
    # negativo grande = deterioro sostenido durante el anio escolar.
    x = np.arange(len(MESES_ASIS), dtype=float)
    m = mensual.to_numpy(dtype=float)
    obs = ~np.isnan(m)
    n_obs = obs.sum(axis=1)
    xs = np.where(obs, x, np.nan)
    mx = np.nanmean(xs, axis=1)
    my = np.nanmean(np.where(obs, m, np.nan), axis=1)
    cov = np.nansum((xs - mx[:, None]) * (m - my[:, None]), axis=1)
    var = np.nansum((xs - mx[:, None]) ** 2, axis=1)
    pendiente = np.where((n_obs >= 3) & (var > 0), cov / np.where(var == 0, 1, var), np.nan)

    out = pd.DataFrame({
        "mrun": df["mrun"],
        "asis_em_anual": df["tasa_asistencia_anual"],
        "asis_em_pendiente": pendiente,
        "asis_em_min": mensual.min(axis=1),
        # Ultimos tres meses del anio escolar: lo mas cercano al ingreso.
        "asis_em_ultimo_trimestre": mensual[MESES_ASIS[-3:]].mean(axis=1),
    })
    # Un estudiante puede aparecer en varias filas (cambio de colegio).
    out = out.dropna(subset=["mrun"]).groupby("mrun", as_index=False).mean()
    log.info("  %s estudiantes con asistencia escolar", f"{len(out):,}")
    return out


def cargar_socioeconomico(year_sep: int, year_becas: int) -> pd.DataFrame | None:
    """Vulnerabilidad SEP y nivel socioeconomico oficial, por MRUN."""
    partes = []
    try:
        p_sep = _find(f"sep_{year_sep}", "*.csv")
        sep = pd.read_csv(p_sep, sep=";", encoding="utf-8",
                          usecols=["MRUN", "PRIORITARIO_ALU", "PREFERENTE_ALU"],
                          dtype=str, low_memory=False)
        sep["mrun"] = pd.to_numeric(sep["MRUN"], errors="coerce")
        sep["prioritario"] = pd.to_numeric(sep["PRIORITARIO_ALU"], errors="coerce")
        sep["preferente"] = pd.to_numeric(sep["PREFERENTE_ALU"], errors="coerce")
        sep = (sep.dropna(subset=["mrun"])[["mrun", "prioritario", "preferente"]]
               .groupby("mrun", as_index=False).max())
        log.info("  SEP %s: %s estudiantes", year_sep, f"{len(sep):,}")
        partes.append(sep)
    except FileNotFoundError:
        log.warning("SEP %s no descargado", year_sep)

    try:
        p_bec = _find(f"becas_{year_becas}", "*.csv")
        bec = pd.read_csv(p_bec, sep=";", encoding="utf-8", dtype=str, low_memory=False)
        bec["mrun"] = pd.to_numeric(bec["MRUN"], errors="coerce")
        bec["quintil_ingreso"] = pd.to_numeric(bec.get("QUINTIL_INGRESO"), errors="coerce")
        bec["decil_dfe"] = pd.to_numeric(bec.get("DECIL_DFE"), errors="coerce")
        # Esta base cubre gratuidad, becas de arancel y el Fondo Solidario.
        # El CAE NO aparece: lo administra la Comision Ingresa y se publica
        # aparte. Una columna `cae` aqui seria constante en cero y llevaria
        # a concluir que "el CAE no predice" cuando nunca estuvo en los datos.
        ben = bec.get("BENEFICIO_BECA_FSCU", pd.Series(dtype=str)).fillna("")
        bec["gratuidad"] = ben.eq("GRATUIDAD").astype(int)
        bec["fscu"] = ben.eq("FSCU").astype(int)
        bec["beca_arancel"] = (~ben.isin(["GRATUIDAD", "FSCU", ""])).astype(int)
        bec = (bec.dropna(subset=["mrun"])
               [["mrun", "quintil_ingreso", "decil_dfe", "gratuidad", "fscu",
                 "beca_arancel"]]
               .groupby("mrun", as_index=False).max())
        log.info("  Becas %s: %s estudiantes", year_becas, f"{len(bec):,}")
        partes.append(bec)
    except FileNotFoundError:
        log.warning("Becas %s no descargado", year_becas)

    if not partes:
        return None
    out = partes[0]
    for extra in partes[1:]:
        out = out.merge(extra, on="mrun", how="outer")
    return out


def _enriquecer(df: pd.DataFrame, asis: pd.DataFrame | None,
                socio: pd.DataFrame | None) -> pd.DataFrame:
    """Agrega las features escolares, dejando NaN donde no hay match."""
    if asis is not None:
        df = df.merge(asis, on="mrun", how="left")
    for c in FEATS_ASISTENCIA:
        if c not in df.columns:
            df[c] = np.nan
    if socio is not None:
        df = df.merge(socio, on="mrun", how="left")
    for c in FEATS_SOCIO:
        if c not in df.columns:
            df[c] = np.nan
    # gratuidad/cae/prioritario ausentes = no tiene el beneficio, que es
    # informacion real; los deciles ausentes si son desconocidos.
    # Ausencia en estas bases significa "no tiene el beneficio" o "no fue
    # clasificado", que es informacion real. Los deciles ausentes, en
    # cambio, si son desconocidos y se imputan con la mediana de train.
    for c in ("gratuidad", "fscu", "beca_arancel", "prioritario", "preferente"):
        df[c] = df[c].fillna(0)
    return df


def piso_temporal(m25: pd.DataFrame) -> None:
    """Igual que `piso_predictivo`, pero con validacion temporal real.

    El resultado anterior usaba un split aleatorio sobre una sola cohorte,
    lo que mezcla estudiantes del mismo anio entre train y test. Aca la
    cohorte 2023 entrena y la 2024 testea: ninguna persona ni ningun anio
    aparece en ambos lados, que es la unica forma de estimar como se
    comportaria el modelo sobre una cohorte futura.
    """
    print("\n" + "=" * 78)
    print("3. PISO PREDICTIVO CON VALIDACION TEMPORAL (cohorte 2023 -> cohorte 2024)")
    print("=" * 78)

    paes23 = load_paes(2023)
    paes24 = load_paes(2024)
    if paes23 is None or paes24 is None:
        return
    try:
        m23 = load_matricula(2023, COHORT_COLS)
    except FileNotFoundError as exc:
        log.warning("%s", exc)
        return
    m24 = load_matricula(2024, COHORT_COLS)

    tr = _construir_cohorte(m23, m24, paes23, 2023)
    te = _construir_cohorte(m24, m25, paes24, 2024)
    del m23

    # Trayectoria escolar del anio ANTERIOR al ingreso de cada cohorte.
    log.info("Cargando trayectoria escolar previa al ingreso")
    tr = _enriquecer(tr, cargar_asistencia_escolar(2022),
                     cargar_socioeconomico(2022, 2023))
    te = _enriquecer(te, cargar_asistencia_escolar(2023),
                     cargar_socioeconomico(2023, 2024))

    print(f"\n  Cohorte 2023 (entrena): n={len(tr):,}  desertan {tr['desertor'].mean():.1%}")
    print(f"  Cohorte 2024 (testea):  n={len(te):,}  desertan {te['desertor'].mean():.1%}")
    print(f"\n  Cobertura de las fuentes nuevas en la cohorte de test:")
    for c, etiq in [("asis_em_anual", "asistencia de ensenanza media"),
                    ("decil_dfe", "decil de ingreso (DFE)"),
                    ("quintil_ingreso", "quintil de ingreso")]:
        print(f"    {etiq:34s} {te[c].notna().mean():6.1%}")
    print(f"    {'prioritario (SEP)':34s} {te['prioritario'].mean():6.1%}")
    print(f"    {'gratuidad':34s} {te['gratuidad'].mean():6.1%}")

    BASE_FEATS = SCORE_COLS + ["sexo_f", "dep_publico", "dep_particular_sub",
                               "dep_particular_pag", "dep_adm_delegada",
                               "anios_desde_egreso"]

    def _evaluar(a: pd.DataFrame, b: pd.DataFrame, feats: list[str],
                 etiqueta: str, verboso: bool = True) -> tuple[float, dict]:
        if len(a) < 5000 or len(b) < 5000:
            print(f"\n  {etiqueta}: muestra insuficiente")
            return float("nan"), {}
        # La mediana se toma de TRAIN. Usar la del conjunto completo seria
        # dejar que el test influya en la imputacion.
        med = a[feats].median()
        Xa = a[feats].fillna(med).fillna(0).to_numpy(dtype=float)
        Xb = b[feats].fillna(med).fillna(0).to_numpy(dtype=float)
        ya, yb = a["desertor"].to_numpy(), b["desertor"].to_numpy()
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, class_weight="balanced"))
        model.fit(Xa, ya)
        p = model.predict_proba(Xb)[:, 1]
        auc = float(roc_auc_score(yb, p))

        def captura(pct: float) -> float:
            k = max(1, int(round(len(p) * pct)))
            return float(yb[np.argsort(-p)[:k]].sum() / yb.sum())

        caps = {f"captura_{int(pct*100):02d}": captura(pct)
                for pct in (0.05, 0.10, 0.20, 0.50)}
        if verboso:
            print(f"\n  {etiqueta}")
            print(f"    entrena n={len(ya):,}  testea n={len(yb):,}  "
                  f"prevalencia test={yb.mean():.1%}")
            print(f"    AUC = {auc:.3f}")
            for pct in (0.05, 0.10, 0.20, 0.50):
                print(f"    top {pct:4.0%} -> captura {captura(pct):5.1%}"
                      f"   ({captura(pct) / pct:.2f}x sobre azar)")
        return auc, caps

    auc_sis, _ = _evaluar(tr, te, BASE_FEATS, "TODO EL SISTEMA (solo ficha de admision)")
    tru = tr[tr["tipo_inst_1"] == "Universidades"]
    teu = te[te["tipo_inst_1"] == "Universidades"]
    auc_u, _ = _evaluar(tru, teu, BASE_FEATS,
                        "SOLO UNIVERSIDADES  <- el comparable para la UA")

    # --- Ablacion: cuanto aporta cada bloque nuevo --------------------
    print("\n" + "=" * 78)
    print("4. ABLACION — cuanto sube el piso al agregar trayectoria escolar")
    print("=" * 78)
    print("  Poblacion: solo universidades, validacion temporal 2023 -> 2024.")

    bloques = {
        "A. Ficha de admision (base)": BASE_FEATS,
        "B. Base + asistencia de ens. media": BASE_FEATS + FEATS_ASISTENCIA,
        "C. Base + socioeconomico": BASE_FEATS + FEATS_SOCIO,
        "D. Base + ambos": BASE_FEATS + FEATS_ASISTENCIA + FEATS_SOCIO,
        "E. Solo asistencia de ens. media": FEATS_ASISTENCIA,
    }
    filas = []
    print(f"\n  {'Modelo':40s} {'AUC':>7} {'delta':>8} {'Top 20%':>9}")
    print("  " + "-" * 68)
    base_auc = None
    for etiqueta, feats in bloques.items():
        auc, caps = _evaluar(tru, teu, feats, etiqueta, verboso=False)
        if base_auc is None:
            base_auc = auc
        delta = auc - base_auc
        print(f"  {etiqueta:40s} {auc:>7.3f} {delta:>+8.3f} "
              f"{caps.get('captura_20', float('nan')):>8.1%}")
        filas.append({"modelo": etiqueta, "n_features": len(feats), "auc": auc,
                      "delta_vs_base": delta, **caps})

    OUT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"validacion": "temporal_2023_2024", "poblacion": "sistema", "auc": auc_sis},
        {"validacion": "temporal_2023_2024", "poblacion": "universidades", "auc": auc_u},
    ]).to_parquet(OUT / "piso_preingreso_temporal.parquet", index=False)

    doc = REPO_ROOT / "documentacion" / "datos"
    doc.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(filas).to_csv(doc / "ablacion_piso.csv", index=False, encoding="utf-8")

    # Cobertura y descriptivas de las fuentes nuevas, para el informe.
    desc = []
    for c in FEATS_ASISTENCIA + ["quintil_ingreso", "decil_dfe"]:
        s = te[c]
        desc.append({"variable": c, "cobertura": round(float(s.notna().mean()), 4),
                     "media": round(float(s.mean()), 4) if s.notna().any() else None,
                     "sd": round(float(s.std()), 4) if s.notna().any() else None,
                     "p25": round(float(s.quantile(0.25)), 4) if s.notna().any() else None,
                     "p50": round(float(s.quantile(0.50)), 4) if s.notna().any() else None,
                     "p75": round(float(s.quantile(0.75)), 4) if s.notna().any() else None})
    for c in ("prioritario", "preferente", "gratuidad", "fscu", "beca_arancel"):
        desc.append({"variable": c, "cobertura": 1.0,
                     "media": round(float(te[c].mean()), 4),
                     "sd": None, "p25": None, "p50": None, "p75": None})
    pd.DataFrame(desc).to_csv(doc / "descriptivas_trayectoria_escolar.csv",
                              index=False, encoding="utf-8")

    print("\n  " + "-" * 74)
    print("  Ninguna persona ni ningun anio aparece en train y test a la vez.")
    print("  El delta de cada bloque es su valor incremental sobre la ficha de")
    print("  admision, que es la prueba que corresponde antes de incorporar")
    print("  variables socioeconomicas a un modelo que prioriza estudiantes.")


def main() -> int:
    # De 2025 solo se necesita saber quien sigue y en que carrera.
    m25 = load_matricula(2025, ["mrun", "cod_inst", "cod_carrera"])
    m24 = load_matricula(2024)
    retencion_ua(m24, m25)
    piso_predictivo(m24, m25)
    del m24
    piso_temporal(m25)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
