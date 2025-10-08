import os, re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --- Percorsi attesi (fallback: qualunque statistics-val-*.csv in /mnt/data)
ROOT = ... # set this to your MedMe repo path
files = [
    f"{ROOT}results/statistics-val-Fitzpatrick+SKINCON_CBM_joint_lambda-1-image.csv",
    f"{ROOT}results/statistics-val-Fitzpatrick+SKINCON_CBM_joint_lambda-0.1-image.csv",
    f"{ROOT}results/statistics-val-Fitzpatrick+SKINCON_CBM_joint_lambda-0.01-image.csv",
    f"{ROOT}results/statistics-val-Fitzpatrick+SKINCON_CBM_joint_lambda-0.001-image.csv",
    f"{ROOT}results/statistics-val-Fitzpatrick+SKINCON_CBM_independent_1-image.csv",  # concetti (x)
    f"{ROOT}results/statistics-val-Fitzpatrick+SKINCON_CBM_independent_2-image.csv",  # task (y)
]
if not any(os.path.exists(p) for p in files):
    raise FileNotFoundError("Nessun file di risultati trovato")


def read_csv_any(p):
    try:    return pd.read_csv(p)
    except: return pd.read_csv(p, sep=";")

def first_numeric(series):
    for v in series:
        try:
            f = float(v)
            if not (np.isnan(f) or np.isinf(f)): return f
        except Exception:
            continue
    return np.nan

def norm_err(v):
    if pd.isna(v): return np.nan
    if 1.0 < v <= 100.0: return v/100.0  # percentuale -> [0,1]
    return v

def _pick_global_min(df: pd.DataFrame, metric_name: str) -> float:
    """
    Estrae la 7a colonna (Global min) della riga con 'metric_name' nella 1a colonna.
    Ritorna np.nan se la metrica non esiste.
    """
    # normalizza testi nella 1a colonna
    first_col = df.iloc[:, 0].astype(str).str.strip()
    mask = first_col.str.lower() == metric_name.lower()
    if not mask.any():
        return np.nan
    # 7a colonna = indice 6
    val = df.loc[mask, df.columns[7]].iloc[0]
    try:
        return float(val)
    except Exception:
        return np.nan

# --- Inferenza robusta ma deterministica basata sui nomi-tipo che mi hai dato ---

def infer_concept_error(df: pd.DataFrame) -> float:
    """
    Joint: usa 'test_SKINCON_concept_loss'
    Independent (_1): usa 'test_SKINCON_loss'
    """
    # prova prima il caso Joint (metrica dedicata ai concetti)
    v = _pick_global_min(df, "test_SKINCON_concept_auroc")
    if not np.isnan(v):
        return 1-v
    # fallback: formato Independent
    return 1-_pick_global_min(df, "test_SKINCON_auroc")

def infer_task_error(df: pd.DataFrame) -> float:
    """
    Joint: usa 'test_SKINCON_final_loss'
    Independent (_2): usa 'test_SKINCON_loss'
    """
    # prova prima il caso Joint (metrica dedicata al task)
    v = _pick_global_min(df, "test_SKINCON_final_acc")
    if not np.isnan(v):
        return 1-v
    # fallback: formato Independent
    return 1-_pick_global_min(df, "test_SKINCON_acc")

def label_from_path(p):
    b = os.path.basename(p)
    if "joint_lambda-" in b:
        m = re.search(r"joint_lambda-([0-9.]+)", b)
        return f"Joint, λ = {m.group(1)}" if m else "Joint"
    if "standard" in b.lower(): return "Standard"
    return os.path.splitext(b)[0]

rows, handled = [], set()

# --- Caso speciale: Independent = (file _1 -> concetti) + (file _2 -> task)
ind1 = [p for p in files if "independent_1" in os.path.basename(p)]
ind2 = [p for p in files if "independent_2" in os.path.basename(p)]
if ind1 and ind2:
    df1, df2 = read_csv_any(ind1[0]), read_csv_any(ind2[0])
    rows.append({
        "label": "Independent",
        "concept_error": infer_concept_error(df1),
        "task_error":    infer_task_error(df2),
    })
    handled.update([ind1[0], ind2[0]])

# --- Altri file (Joint, Standard, ecc.)
for p in files:
    if p in handled: 
        continue
    df = read_csv_any(p)
    rows.append({
        "label": label_from_path(p),
        "concept_error": infer_concept_error(df),
        "task_error":    infer_task_error(df),
    })

res = pd.DataFrame(rows).sort_values("label").reset_index(drop=True)
print(res)

# --- Grafico
plt.figure(figsize=(6,4), dpi=160)
ax = plt.gca()
for _, r in res.iterrows():
    ax.scatter(r["concept_error"], r["task_error"], marker="s")
    ax.annotate(r["label"], (r["concept_error"], r["task_error"]),
                xytext=(4,4), textcoords="offset points", fontsize=8)
ax.set_xlabel("Concept (c) error")
ax.set_ylabel("Task (y) error")
ax.set_title("CBM Trade-off")
ax.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig(f"{ROOT}results/cbm_tradeoff_independent_combined.png", bbox_inches="tight")
plt.show()