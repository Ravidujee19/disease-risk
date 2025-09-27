import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import base64, datetime as dt
import argparse

# Config 
RAW_DIR   = Path(__file__).resolve().parent.parent / "data" / "raw"

# PNGs saving path
IMG_ROOT  = Path(__file__).resolve().parent.parent / "reports" / "figures" / "before_preprocessed"

# HTML + dataset_profile.csv saving path
META_DIR  = Path(__file__).resolve().parent.parent / "reports" / "before_preprocessed"

TARGET = "target"            
FIG_LIMIT_CAT = 20          
BOXPLOT_NUM_LIMIT = 12       
BOXPLOT_MAX_CLASSES = 5      
NUMERIC_THRESHOLD = 0.60     

# Helpers
def save_fig(fig, path: Path, tight=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()  # ensure layout computed
    if tight:
        fig.savefig(path, dpi=140, bbox_inches="tight")
    else:
        fig.savefig(path, dpi=140)
    plt.close(fig)

def img_to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")

def terminal_preview(df: pd.DataFrame):
    print("== RAW CSV PREVIEW ==")
    print(f"- Shape: {df.shape}")
    print("- Head:")
    print(df.head(5).to_string(index=False))
    print("- Missing (%) top 10:")
    miss = (df.isna().mean()*100).sort_values(ascending=False)
    print(miss.head(10).round(2).to_string())

def coerce_numeric_cols(df: pd.DataFrame, threshold: float = NUMERIC_THRESHOLD) -> list[str]:
    """Return columns that are numeric or mostly-numeric (after coercion)."""
    numeric_cols = []
    for c in df.columns:
        # already numeric
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
            continue
        # try coercion
        s = pd.to_numeric(df[c], errors="coerce")
        valid_ratio = s.notna().mean()
        if valid_ratio >= threshold:
            numeric_cols.append(c)
    return numeric_cols

# Artifact builders
def make_classic_outputs(df: pd.DataFrame) -> tuple[Path, list[tuple[str, Path]], str]:
    """Create PNGs under IMG_ROOT + dataset_profile.csv under META_DIR.
       Return (profile_csv, [(title, path)], balance_html)."""
    figs: list[tuple[str, Path]] = []

    # Profile CSV
    rows = []
    for c in df.columns:
        s = df[c]
        rows.append({
            "column": c,
            "dtype": str(s.dtype),
            "missing_%": round(s.isna().mean()*100, 2),
            "unique_vals": int(s.nunique(dropna=True)),
            "examples": ", ".join(map(str, s.dropna().astype(str).unique()[:5].tolist()))
        })
    profile_df = pd.DataFrame(rows)
    META_DIR.mkdir(parents=True, exist_ok=True)
    profile_csv = META_DIR / "dataset_profile.csv"
    profile_df.to_csv(profile_csv, index=False)

    # Missingness
    miss = df.isna().mean() * 100
    fig = plt.figure(figsize=(8, max(4, 0.22*len(miss))))
    miss.sort_values().plot(kind="barh")
    plt.title("Missing % by column")
    p = IMG_ROOT / "missingness.png"; save_fig(fig, p)
    figs.append(("Missing % by column", p))

    # Cardinality
    card = df.nunique(dropna=True)
    fig = plt.figure(figsize=(8, max(4, 0.22*len(card))))
    card.sort_values().plot(kind="barh")
    plt.title("Unique values per column")
    p = IMG_ROOT / "cardinality.png"; save_fig(fig, p)
    figs.append(("Unique values per column", p))

    # Detect numeric robustly 
    numeric_cols = coerce_numeric_cols(df)
    df_num = {}
    for c in numeric_cols:
        df_num[c] = pd.to_numeric(df[c], errors="coerce")

    # Numeric histograms
    for col in numeric_cols:
        s = df_num[col].dropna()
        if s.empty:
            print(f"(!) Skip hist: {col} had no numeric values after coercion.")
            continue
        fig = plt.figure(figsize=(6,4))
        s.hist(bins=30)
        plt.title(f"Histogram • {col}")
        p = IMG_ROOT / f"hist_{col}.png"; save_fig(fig, p)
        figs.append((f"Histogram • {col}", p))

    # Boxplots vs target (manual & robust)
    if TARGET in df.columns:
        tgt = df[TARGET].astype("string")
        # top classes by count to avoid 100s of boxes
        top_classes = tgt.value_counts().index[:BOXPLOT_MAX_CLASSES].tolist()
        if len(top_classes) < 2:
            print(f"(!) Skip all boxplots: TARGET '{TARGET}' has <2 classes among top groups.")
        else:
            for col in numeric_cols[:BOXPLOT_NUM_LIMIT]:
                vals = df_num[col]
                data = pd.DataFrame({col: vals, TARGET: tgt})
                data = data[data[TARGET].isin(top_classes)].dropna(subset=[col, TARGET])

                groups = []
                labels = []
                for cls in top_classes:
                    g = data.loc[data[TARGET] == cls, col].dropna().values
                    if len(g) >= 5:
                        groups.append(g)
                        labels.append(str(cls))

                if len(groups) < 2:
                    print(f"(!) Skip boxplot {col}: not enough groups with data (need ≥2 with ≥5 samples).")
                    continue

                fig, ax = plt.subplots(figsize=(7,4))
                ax.boxplot(groups, labels=labels, showfliers=False)
                ax.set_title(f"{col} vs {TARGET} (top classes)")
                ax.set_xlabel(TARGET); ax.set_ylabel(col)
                p = IMG_ROOT / f"box_{col}_vs_{TARGET}.png"
                save_fig(fig, p, tight=False)
                figs.append((f"Boxplot • {col} vs {TARGET}", p))

    # Categorical bars (non-numeric)
    cat_cols = [c for c in df.columns if c not in numeric_cols]
    for col in cat_cols:
        vc = df[col].astype("string").fillna("NA").value_counts().head(FIG_LIMIT_CAT)
        if vc.empty:
            print(f"(!) Skip bar: {col} has no values.")
            continue
        fig = plt.figure(figsize=(6,4))
        vc.plot(kind="bar")
        plt.title(f"Distribution • {col}")
        p = IMG_ROOT / f"bar_{col}.png"; save_fig(fig, p)
        figs.append((f"Bar • {col}", p))

    # Correlation heatmap
    if len(numeric_cols) > 1:
        corr_df = pd.DataFrame({c: df_num[c] for c in numeric_cols})
        corr = corr_df.corr(numeric_only=True)
        fig = plt.figure(figsize=(min(12, 0.6*len(numeric_cols)+4), min(10, 0.6*len(numeric_cols)+4)))
        plt.imshow(corr, aspect="auto")
        plt.colorbar()
        plt.xticks(range(len(numeric_cols)), numeric_cols, rotation=90)
        plt.yticks(range(len(numeric_cols)), numeric_cols)
        plt.title("Correlation Heatmap")
        p = IMG_ROOT / "corr_heatmap.png"; save_fig(fig, p, tight=False)
        figs.append(("Correlation Heatmap", p))

    # Target balance
    balance_html = ""
    if TARGET in df.columns:
        vc = df[TARGET].astype("string").fillna("NA").value_counts()
        fig = plt.figure(figsize=(6,4))
        vc.plot(kind="bar")
        plt.title(f"Target Balance • {TARGET}")
        p = IMG_ROOT / f"target_balance.png"; save_fig(fig, p)
        figs.append((f"Target Balance • {TARGET}", p))
        balance_html = vc.to_frame(name="count").to_html(border=0)

    return profile_csv, figs, balance_html

def make_html_report(csv_path: Path, figs: list[tuple[str, Path]], balance_html: str, input_file: Path) -> Path:
    profile_df = pd.read_csv(csv_path)
    preview = profile_df[["column", "dtype", "missing_%", "unique_vals", "examples"]]
    preview_html = preview.to_html(index=False, border=0)

    img_cards = []
    for title, path in figs:
        b64 = img_to_b64(path)
        img_cards.append(f"""
          <div class="card">
            <h3>{title}</h3>
            <img src="data:image/png;base64,{b64}" alt="{title}" />
          </div>
        """)

    generated = dt.datetime.now().strftime("%Y-%m-%d %H:%M")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"/>
<title>EDA Report (Before Preprocessing)</title>
<style>
 body {{ font-family: Arial, sans-serif; background:#0b0f17; color:#e5e7eb; margin:0; }}
 header {{ padding:24px 28px; background:#111827; position:sticky; top:0; }}
 h1 {{ margin:0 0 6px; font-size:24px; }}
 .meta {{ color:#9ca3af; font-size:14px; }}
 main {{ padding:24px; max-width:1200px; margin:auto; }}
 section {{ margin-bottom:28px; }}
 .grid {{ display:grid; grid-template-columns: repeat(auto-fill, minmax(340px,1fr)); gap:18px; }}
 .card {{ background:#111827; border:1px solid #1f2937; border-radius:14px; padding:14px; }}
 .card img {{ width:100%; height:auto; border-radius:10px; display:block; }}
 table {{ width:100%; border-collapse: collapse; }}
 th, td {{ border-bottom: 1px solid #1f2937; padding:8px 10px; text-align:left; }}
 th {{ background:#0f172a; }}
 .small {{ color:#9ca3af; font-size:13px; }}
</style></head>
<body>
<header>
  <h1>Exploratory Data Analysis — Before Preprocessing</h1>
  <div class="meta">
    Source: <b>{input_file.name}</b> · Path: <code>{input_file}</code><br/>Generated: {generated}
  </div>
</header>
<main>
  <section>
    <h2>Dataset Profile</h2>
    <p class="small">Saved CSV: <code>{csv_path}</code></p>
    {preview_html}
  </section>
  {"<section><h2>Target Balance</h2>"+balance_html+"</section>" if balance_html else ""}
  <section>
    <h2>Visuals</h2>
    <div class="grid">
      {"".join(img_cards)}
    </div>
  </section>
</main>
</body></html>"""

    report_path = META_DIR / "EDA_Report.html"
    META_DIR.mkdir(parents=True, exist_ok=True)
    report_path.write_text(html, encoding="utf-8")
    return report_path

# Main method
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["all","plots","html"], default="all",
                    help="all = PNGs + CSV + HTML (default). plots = PNGs + CSV only. html = only HTML (reuses existing artifacts).")
    args = ap.parse_args()

    # Locate CSV
    csv_files = sorted(RAW_DIR.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {RAW_DIR}")
    input_file = csv_files[0]

    # Ensure dirs
    IMG_ROOT.mkdir(parents=True, exist_ok=True)
    META_DIR.mkdir(parents=True, exist_ok=True)

    # Load
    df = pd.read_csv(input_file)
    terminal_preview(df)

    # Build artifacts
    if args.mode in ("all", "plots"):
        profile_csv, figs, balance_html = make_classic_outputs(df)
    else:
        profile_csv = META_DIR / "dataset_profile.csv"
        if not profile_csv.exists():
            raise FileNotFoundError("dataset_profile.csv not found. Run with --mode plots or all first.")
        figs = [(p.stem.replace("_", " ").title(), p) for p in sorted(IMG_ROOT.glob("*.png"))]
        balance_html = ""

    if args.mode in ("all", "html"):
        report = make_html_report(profile_csv, figs, balance_html, input_file)
        print(f"📑 HTML report: {report}")

    print(f"Figures dir: {IMG_ROOT}")
    print(f"Profile CSV: {META_DIR/'dataset_profile.csv'}")
    print("EDA done.")

if __name__ == "__main__":
    main()
