"""
Sound Life Cohort — Bayesian Network Foundation Model Dashboard
=================================================================
Interactive Streamlit app for exploring Bayesian-network structure-learning
results across the Sound Life Cohort multi-omics layers (L1a ... L6, L_all).

Run with:  streamlit run app.py
"""

import re
import pathlib

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_agraph import Config, Edge, Node, agraph

# --------------------------------------------------------------------------
# Paths & constants
# --------------------------------------------------------------------------
ROOT = pathlib.Path(__file__).parent
EDGE_DIR = ROOT / "data" / "edges"
IMG_NET_DIR = ROOT / "data" / "images" / "network"
IMG_HIST_DIR = ROOT / "data" / "images" / "hist"
REPORT_DIR = ROOT / "data" / "reports"

SCORECARD_PATH = REPORT_DIR / "all_networks_scorecard.csv"
LIT_REVIEW_PATH = REPORT_DIR / "lit_review_module_dashboard_inputs.csv"

ROOT_NODES = {"age_group", "sex", "cmv"}

# --------------------------------------------------------------------------
# Two modeling approaches, same layers, same candidate node/arc sets:
#   cv        — structure learning validated via k-fold cross-validation,
#               averaged network thresholded at an algorithm-chosen optimum.
#   bootstrap — structure learning validated via bootstrap resampling,
#               averaged network thresholded at a conservative arc-strength
#               cutoff (with the 0.50 "optimal-fit" line shown for reference).
# --------------------------------------------------------------------------
MODELS = {
    "cv": {
        "label": "Cross-validation (CV)",
        "short": "CV",
        "edge_dir": EDGE_DIR / "cv",
        "net_dir": IMG_NET_DIR / "cv",
        "hist_dir": IMG_HIST_DIR / "cv",
        "avg_suffix": "avg",
        "color": "#2F6F62",
    },
    "bootstrap": {
        "label": "Bootstrap",
        "short": "Bootstrap",
        "edge_dir": EDGE_DIR / "bootstrap",
        "net_dir": IMG_NET_DIR / "bootstrap",
        "hist_dir": IMG_HIST_DIR / "bootstrap",
        "avg_suffix": "avg_opt",
        "color": "#C1793B",
    },
}
MODEL_ORDER = ["cv", "bootstrap"]

LAYER_META = {
    "L1a": {
        "label": "L1a — Clinical (full panel)",
        "modality": "Clinical labs & metadata",
        "description": (
            "Full curated clinical-labs panel: anthropometrics, CBC with "
            "differential, CMP, lipid profile, and inflammatory markers, "
            "plus the three demographic root nodes."
        ),
    },
    "L1b": {
        "label": "L1b — Clinical (curated subset)",
        "modality": "Clinical labs & metadata",
        "description": (
            "A trimmed, curated subset of the L1a clinical panel intended "
            "to test whether a smaller, more targeted node set improves "
            "structure-learning stability at this sample size."
        ),
    },
    "L2": {
        "label": "L2 — Olink proteomics",
        "modality": "Plasma proteomics (Olink)",
        "description": (
            "Plasma protein nodes selected from published immune-aging / "
            "inflammation biology (Olink Explore panel), plus the three "
            "root nodes."
        ),
    },
    "L3a": {
        "label": "L3a — Clinical (full) + Olink",
        "modality": "Clinical + Proteomics (integrated)",
        "description": (
            "Integrated network combining the full clinical panel (L1a) "
            "with the Olink protein set (L2), to look for cross-modality "
            "structure."
        ),
    },
    "L3b": {
        "label": "L3b — Clinical (curated) + Olink",
        "modality": "Clinical + Proteomics (integrated)",
        "description": (
            "Integrated network combining the curated clinical subset "
            "(L1b) with the Olink protein set (L2)."
        ),
    },
    "L4": {
        "label": "L4 — Whole-blood Hallmark pathways",
        "modality": "Whole-blood RNA-seq (unstimulated)",
        "description": (
            "Sample-level Hallmark pathway activity scores derived from "
            "bulk, unstimulated whole-blood RNA-seq, reflecting baseline "
            "in vivo transcriptional state."
        ),
    },
    "L5": {
        "label": "L5 — Immune-cell composition",
        "modality": "scRNA-seq cell-type frequencies",
        "description": (
            "CLR/ALR-transformed cell-type proportions across AIFI L1/L2/L3 "
            "resolution levels, describing baseline immune-cell composition."
        ),
    },
    "L6": {
        "label": "L6 — Pseudobulk pathway signaling",
        "modality": "Pseudobulk scRNA-seq (cell type × pathway)",
        "description": (
            "Cell-type-by-Hallmark-pathway pseudobulk activity scores — "
            "the highest-dimensional node set, capturing pathway signaling "
            "within specific immune cell subsets."
        ),
    },
    "L_all": {
        "label": "L_all — Combined network",
        "modality": "All modalities (integrated)",
        "description": (
            "The full, integrated network spanning clinical, proteomic, "
            "cell-composition, and pathway-signaling nodes together."
        ),
    },
}

LAYER_ORDER = ["L1a", "L1b", "L2", "L3a", "L3b", "L4", "L5", "L6", "L_all"]
LAYER_CASE_MAP = {k.upper(): k for k in LAYER_ORDER}

COLOR_ROOT = "#e15759"       # red   — demographic / serostatus root nodes
COLOR_PROTEIN = "#af7aa1"    # purple — Olink protein (ALL CAPS gene symbol)
COLOR_CELLFREQ = "#59a14f"   # green — cell-type frequency (l1_/l2_/l3_ ..._alr)
COLOR_PATHWAY = "#f28e2b"    # orange — Hallmark pathway / pb signaling score
COLOR_CLINICAL = "#4e79a7"   # blue  — clinical lab / metadata

PATHWAY_LAYERS = {"L4", "L6"}
CELLFREQ_LAYERS = {"L5"}

# Validation-page palette — kept distinct from the layer/node colors above
# so validation results are never visually confused with structure-learning
# node categories.
COLOR_SUPPORTED = "#2F6F62"     # teal  — agrees with expectation / literature
COLOR_REVERSED = "#C1793B"      # amber — association found, direction disagrees
COLOR_NODIR = "#8C9BAF"         # slate — related, no directional claim
COLOR_CONFLICT = "#e15759"      # red   — actively conflicting
COLOR_NOMATCH = "#D8D3C4"       # sand  — no literature match found (candidate novel)


# --------------------------------------------------------------------------
# Visual identity — injected once per page
# --------------------------------------------------------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,500;8..60,600;8..60,700&family=Inter:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

        :root {
          --ink: #16233B;
          --paper: #FBFAF6;
          --panel: #FFFFFF;
          --teal: #2F6F62;
          --teal-soft: #E7EFEC;
          --amber: #C1793B;
          --amber-soft: #FBF1E6;
          --line: #E4DED0;
          --muted: #68707D;
        }

        html, body, [class^="css"], [class*=" css"] {
          font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        [data-testid="stAppViewContainer"] { background: var(--paper); }
        [data-testid="stHeader"] { background: transparent; }

        [data-testid="stSidebar"] {
          background: var(--ink);
        }
        [data-testid="stSidebar"] * { color: #E7E9EF !important; }
        [data-testid="stSidebar"] [data-baseweb="select"] * { color: #16233B !important; }
        [data-testid="stSidebar"] hr { border-color: #2C3B58; }
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #9FAAC0 !important; }

        h1, h2, h3 {
          font-family: 'Source Serif 4', Georgia, serif !important;
          color: var(--ink);
          letter-spacing: -0.01em;
        }
        h1 { font-weight: 700 !important; }
        h2, h3 { font-weight: 600 !important; }

        p, li, div, span { color: var(--ink); }
        .stCaption, [data-testid="stCaptionContainer"] { color: var(--muted) !important; }

        [data-testid="stMetric"] {
          background: var(--panel);
          border: 1px solid var(--line);
          border-radius: 10px;
          padding: 0.95rem 1.1rem 0.75rem 1.1rem;
        }
        [data-testid="stMetricValue"] {
          font-family: 'IBM Plex Mono', monospace;
          color: var(--ink);
        }
        [data-testid="stMetricLabel"] {
          color: var(--muted) !important;
          text-transform: uppercase;
          letter-spacing: 0.06em;
          font-size: 0.72rem;
        }

        .eyebrow {
          font-family: 'IBM Plex Mono', monospace;
          text-transform: uppercase;
          letter-spacing: 0.16em;
          font-size: 0.72rem;
          color: var(--teal);
          margin-bottom: 0.5rem;
        }
        .hero-title {
          font-family: 'Source Serif 4', serif;
          font-weight: 700;
          font-size: 2.35rem;
          line-height: 1.18;
          color: var(--ink);
          margin: 0 0 0.7rem 0;
        }
        .hero-sub {
          color: var(--muted);
          font-size: 1.03rem;
          max-width: 68ch;
          line-height: 1.6;
          margin-bottom: 0.4rem;
        }
        .hr-line {
          border: none;
          border-top: 1px solid var(--line);
          margin: 1.5rem 0 1.25rem 0;
        }
        .section-label {
          font-family: 'IBM Plex Mono', monospace;
          text-transform: uppercase;
          font-size: 0.72rem;
          letter-spacing: 0.12em;
          color: var(--muted);
          margin: 0.2rem 0 0.55rem 0;
        }
        .finding-card {
          background: var(--amber-soft);
          border-left: 3px solid var(--amber);
          border-radius: 3px;
          padding: 0.8rem 1.05rem;
          margin-bottom: 0.5rem;
          font-size: 0.93rem;
          line-height: 1.55;
          color: var(--ink);
        }
        .finding-card b { color: var(--ink); }
        .insight-card {
          background: var(--teal-soft);
          border-left: 3px solid var(--teal);
          border-radius: 3px;
          padding: 0.8rem 1.05rem;
          margin-bottom: 0.5rem;
          font-size: 0.93rem;
          line-height: 1.55;
          color: var(--ink);
        }
        .layer-chip {
          display: inline-block;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          padding: 0.15rem 0.6rem;
          border-radius: 999px;
          background: var(--teal-soft);
          color: var(--teal);
          border: 1px solid #C9DCD6;
          margin-right: 0.4rem;
        }
        .modality-tag {
          display: inline-block;
          font-family: 'IBM Plex Mono', monospace;
          font-size: 0.72rem;
          letter-spacing: 0.02em;
          padding: 0.2rem 0.65rem;
          border-radius: 4px;
          background: var(--panel);
          border: 1px solid var(--line);
          color: var(--muted);
          margin-bottom: 0.6rem;
        }
        [data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 8px; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_cards(bullets: list[str], kind: str = "finding") -> None:
    css_class = "finding-card" if kind == "finding" else "insight-card"
    html = "".join(f'<div class="{css_class}">{b}</div>' for b in bullets)
    st.markdown(html, unsafe_allow_html=True)


def section_label(text: str) -> None:
    st.markdown(f'<div class="section-label">{text}</div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------
# Data loading — every loader takes a `model` key ("cv" or "bootstrap") so
# the exact same code path serves both modeling approaches.
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_edges(layer_key: str, model: str = "cv") -> pd.DataFrame:
    path = MODELS[model]["edge_dir"] / f"{layer_key}.csv"
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["in_final_dag"] = df["in_final_dag"].astype(str).str.upper() == "TRUE"
    return df


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame:
    path = REPORT_DIR / "model_comparison.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_bootstrap_thresholds() -> pd.DataFrame:
    path = REPORT_DIR / "bootstrap_thresholds.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_scorecard() -> pd.DataFrame:
    """Held-out flu-vaccine validation scores per network, per validation
    method (cv / bootstrap) — the project's first evaluation approach
    (see About / Methods: 'Held-out benchmark'). Each `wvote_*` column runs
    from -1 (the network's learned root -> feature -> flu-outcome path
    contradicts literature-established expectation) to +1 (fully agrees),
    weighted by edge strength x |effect size|; NaN means no root edge
    existed at that layer to score for that root. `n_root_edges` matters:
    a high score built on very few edges (e.g. <= 3) rests on too little
    structure to trust on its own."""
    if not SCORECARD_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(SCORECARD_PATH)
    df["network"] = df["network"].map(LAYER_CASE_MAP).fillna(df["network"])
    return df


@st.cache_data(show_spinner=False)
def load_lit_review() -> pd.DataFrame:
    """Per-network, per-model rollup of the project's second evaluation
    approach (see About / Methods: 'Edge-level literature corroboration').
    Every edge in a network's final DAG was searched against INDRA and
    PubMed and classified as: direction Supported by what was found,
    Reversed (an association is documented but the literature's direction
    disagrees with the network's), related with No directional info, or
    actively Conflicting — each further split by which source(s) surfaced
    it. Edges with zero hits in either source ('No literature match found')
    are the project's candidate-novel-finding pool."""
    if not LIT_REVIEW_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(LIT_REVIEW_PATH)

    def parse_network(n: str):
        n = str(n).lower()
        if n.endswith("_cv"):
            return n[:-3], "cv"
        if n.endswith("_boot"):
            return n[:-5], "bootstrap"
        return n, None

    parsed = df["network"].apply(parse_network)
    df["layer"] = parsed.apply(lambda t: t[0].upper()).map(LAYER_CASE_MAP)
    df["model"] = parsed.apply(lambda t: t[1])

    class_groups = {
        "Supported": ["support_dir_indra_only", "support_dir_pm_only",
                      "support_dir_both", "support_dir_neither"],
        "Reversed": ["reverse_indra_only", "reverse_pm_only",
                     "reverse_both", "reverse_neither"],
        "No directional info": ["no_dir_indra_only", "no_dir_pm_only",
                                 "no_dir_both", "no_dir_neither"],
        "Conflicting": ["conflicting_indra_only", "conflicting_pm_only",
                         "conflicting_both", "conflicting_neither"],
    }
    for label, cols in class_groups.items():
        df[label] = df[cols].sum(axis=1)
    df["No literature match found"] = df["edges"] - df["edges_with_search_results"]
    return df


@st.cache_data(show_spinner=False)
def final_dag_edges(layer_key: str, model: str = "cv") -> pd.DataFrame:
    df = load_edges(layer_key, model)
    return df[df["in_final_dag"]].sort_values("strength", ascending=False)


@st.cache_data(show_spinner=False)
def deduped_directed_edges(layer_key: str, model: str = "cv") -> pd.DataFrame:
    """Collapse the two direction-rows per undirected pair into a single
    best-direction directed edge (mirrors, approximately, what bnlearn's
    averaged-network construction does before thresholding)."""
    df = load_edges(layer_key, model).copy()
    df["pair_key"] = df.apply(lambda r: tuple(sorted([r["from"], r["to"]])), axis=1)
    idx = df.groupby("pair_key")["direction"].idxmax()
    out = df.loc[idx].drop(columns="pair_key").sort_values("strength", ascending=False)
    return out.reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compare_layer_models(layer_key: str) -> dict:
    """Head-to-head comparison of the CV and Bootstrap final DAGs for one
    layer: which arcs both models agree on (same pair, same direction),
    which they agree exist but disagree on direction, and which are unique
    to one model."""
    cv_fd = final_dag_edges(layer_key, "cv")
    bt_fd = final_dag_edges(layer_key, "bootstrap")

    def pair_map(df: pd.DataFrame) -> dict:
        return {tuple(sorted([r["from"], r["to"]])): (r["from"], r["to"], r["strength"])
                for _, r in df.iterrows()}

    cv_map = pair_map(cv_fd)
    bt_map = pair_map(bt_fd)

    cv_pairs, bt_pairs = set(cv_map), set(bt_map)
    shared_pairs = cv_pairs & bt_pairs
    cv_only_pairs = cv_pairs - bt_pairs
    bt_only_pairs = bt_pairs - cv_pairs

    same_direction, reversed_direction = [], []
    for p in shared_pairs:
        cv_from, cv_to, cv_s = cv_map[p]
        bt_from, bt_to, bt_s = bt_map[p]
        record = {"from_cv": cv_from, "to_cv": cv_to, "strength_cv": cv_s,
                  "from_boot": bt_from, "to_boot": bt_to, "strength_boot": bt_s}
        if (cv_from, cv_to) == (bt_from, bt_to):
            same_direction.append(record)
        else:
            reversed_direction.append(record)

    union_n = len(cv_pairs | bt_pairs)
    jaccard = (len(shared_pairs) / union_n) if union_n else 0.0

    return {
        "cv_final": cv_fd,
        "bt_final": bt_fd,
        "cv_pairs": cv_pairs,
        "bt_pairs": bt_pairs,
        "shared_pairs": shared_pairs,
        "cv_only_pairs": cv_only_pairs,
        "bt_only_pairs": bt_only_pairs,
        "same_direction": same_direction,
        "reversed_direction": reversed_direction,
        "jaccard": jaccard,
        "cv_map": cv_map,
        "bt_map": bt_map,
    }


def classify_node(node: str, layer_key: str) -> str:
    if node in ROOT_NODES:
        return COLOR_ROOT
    if re.fullmatch(r"[A-Z0-9]+", node):
        return COLOR_PROTEIN
    if layer_key in CELLFREQ_LAYERS and re.match(r"^l[123]_", node):
        return COLOR_CELLFREQ
    if layer_key in PATHWAY_LAYERS:
        return COLOR_PATHWAY
    return COLOR_CLINICAL


def node_category_label(color: str) -> str:
    return {
        COLOR_ROOT: "Root / demographic node",
        COLOR_PROTEIN: "Olink protein",
        COLOR_CELLFREQ: "Cell-type frequency",
        COLOR_PATHWAY: "Pathway / signaling score",
        COLOR_CLINICAL: "Clinical / other",
    }[color]


# --------------------------------------------------------------------------
# Data-driven "findings" text — every number below is computed live from
# the edge tables / model-comparison report, not hand-written.
# --------------------------------------------------------------------------
def node_degree(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=int)
    return pd.concat([df["from"], df["to"]]).value_counts()


def root_touching_edges(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    mask = df["from"].isin(ROOT_NODES) | df["to"].isin(ROOT_NODES)
    return df[mask]


@st.cache_data(show_spinner=False)
def layer_findings(layer_key: str, model: str = "cv") -> list[str]:
    fd = final_dag_edges(layer_key, model)
    full = load_edges(layer_key, model)
    n_pairs = len(full) // 2
    retention = (len(fd) / n_pairs) if n_pairs else 0.0
    model_label = MODELS[model]["short"]

    bullets = [
        f"The {model_label}-averaged network keeps <b>{len(fd):,} of {n_pairs:,}</b> "
        f"candidate arcs evaluated at this layer — a <b>{retention:.0%}</b> retention rate."
    ]

    deg = node_degree(fd)
    if len(deg):
        hub, hub_n = deg.index[0], int(deg.iloc[0])
        bullets.append(
            f"<b>{hub}</b> is the most-connected node in the final DAG, "
            f"with <b>{hub_n}</b> edges touching it."
        )

    if len(fd):
        top = fd.iloc[0]
        bullets.append(
            f"Strongest-supported edge: <b>{top['from']} → {top['to']}</b> "
            f"(bootstrap strength {top['strength']:.2f}, direction confidence {top['direction']:.2f})."
        )

    r_edges = root_touching_edges(fd)
    if len(r_edges):
        touched = sorted(
            (set(r_edges["from"]) | set(r_edges["to"])) & ROOT_NODES
        )
        bullets.append(
            f"<b>{len(r_edges)}</b> edges in the final DAG connect directly to a root "
            f"demographic node ({', '.join(touched)})."
        )
    else:
        bullets.append(
            "No edges in the final DAG connect directly to a root demographic node "
            "at this layer — any age/sex/CMV effects here would be mediated indirectly."
        )

    return bullets


@st.cache_data(show_spinner=False)
def overview_findings(model: str = "cv") -> list[str]:
    retentions = {}
    for k in LAYER_ORDER:
        full = load_edges(k, model)
        fd = final_dag_edges(k, model)
        n_pairs = len(full) // 2
        retentions[k] = (len(fd) / n_pairs) if n_pairs else 0.0

    densest = max(retentions, key=retentions.get)
    sparsest = min(retentions, key=retentions.get)

    bullets = [
        f"<b>{densest}</b> retains the largest share of candidate arcs in its final "
        f"DAG ({retentions[densest]:.0%}); <b>{sparsest}</b> retains the smallest "
        f"({retentions[sparsest]:.0%})."
    ]

    if model == "cv":
        comp = load_model_comparison()
        if not comp.empty:
            best = comp.loc[comp["cv_loss_mean"].idxmin()]
            worst = comp.loc[comp["cv_loss_mean"].idxmax()]
            bullets.append(
                f"Among the layers in the four-algorithm comparison report, "
                f"<b>{best['network']}</b> has the lowest mean held-out CV loss "
                f"({best['cv_loss_mean']:.2f}); <b>{worst['network']}</b> has the highest "
                f"({worst['cv_loss_mean']:.2f}). Loss scales with node count and modality, "
                f"so this is more meaningful within a layer than across layers."
            )
    else:
        thr = load_bootstrap_thresholds()
        if not thr.empty:
            widest = thr.loc[(thr["boot_cons"] - thr["boot_opt"]).idxmax()]
            bullets.append(
                f"Bootstrap conservative thresholds range from "
                f"<b>{thr['boot_cons'].min():.2f}</b> to <b>{thr['boot_cons'].max():.2f}</b> "
                f"arc-strength support across layers (vs. a fixed 0.50 optimal-fit line); "
                f"<b>{widest['layer']}</b> has the widest gap between the two "
                f"(cons={widest['boot_cons']:.2f})."
            )

    fd_all = final_dag_edges("L_all", model)
    r_all = root_touching_edges(fd_all)
    if len(r_all):
        counts = pd.concat(
            [
                r_all.loc[r_all["from"].isin(ROOT_NODES), "from"],
                r_all.loc[r_all["to"].isin(ROOT_NODES), "to"],
            ]
        ).value_counts()
        top_root = counts.index[0]
        bullets.append(
            f"In the combined <b>L_all</b> network, <b>{top_root}</b> is the root node "
            f"with the most direct connections ({int(counts.iloc[0])}) into the "
            f"multi-omics layers."
        )

    return bullets


@st.cache_data(show_spinner=False)
def cv_vs_bootstrap_overview_findings() -> list[str]:
    """Cross-model summary bullets computed across all nine layers at once."""
    rows = []
    for k in LAYER_ORDER:
        cmp = compare_layer_models(k)
        rows.append({"layer": k, **cmp})

    total_cv = sum(len(r["cv_pairs"]) for r in rows)
    total_bt = sum(len(r["bt_pairs"]) for r in rows)
    total_shared = sum(len(r["shared_pairs"]) for r in rows)
    total_reversed = sum(len(r["reversed_direction"]) for r in rows)
    mean_jaccard = sum(r["jaccard"] for r in rows) / len(rows)

    most_agree = max(rows, key=lambda r: r["jaccard"])
    least_agree = min(rows, key=lambda r: r["jaccard"])

    bullets = [
        f"Across all nine layers, the CV final DAGs total <b>{total_cv:,}</b> arcs and the "
        f"Bootstrap final DAGs total <b>{total_bt:,}</b> arcs; <b>{total_shared:,}</b> "
        f"undirected connections are recovered by <b>both</b> approaches "
        f"(mean layer-level agreement, Jaccard = <b>{mean_jaccard:.0%}</b>)."
    ]
    bullets.append(
        f"<b>{most_agree['layer']}</b> shows the strongest agreement between models "
        f"(Jaccard = {most_agree['jaccard']:.0%}); <b>{least_agree['layer']}</b> shows the "
        f"weakest (Jaccard = {least_agree['jaccard']:.0%})."
    )
    if total_reversed:
        bullets.append(
            f"Of the connections both models recover, <b>{total_reversed:,}</b> are "
            f"flagged with <b>opposite edge direction</b> between CV and Bootstrap — "
            f"worth treating as lower-confidence causal calls even though both "
            f"approaches agree an association exists."
        )
    return bullets


@st.cache_data(show_spinner=False)
def diagnostics_findings(layer_key: str, model: str = "cv") -> list[str]:
    bullets = []

    if model == "cv":
        comp = load_model_comparison()
        if not comp.empty:
            match = comp[comp["network"].str.startswith(f"{layer_key}_")]
            if not match.empty:
                row = match.iloc[0]
                bullets.append(
                    f"Hill-climbing and tabu search searched the same "
                    f"<b>{int(row['n_nodes'])}</b>-node space and landed on similarly sized "
                    f"graphs ({int(row['edges_hc'])} vs {int(row['edges_tabu'])} edges); "
                    f"MM-HC was the most conservative, at <b>{int(row['edges_mmhc'])}</b> edges."
                )
                bullets.append(
                    f"The averaged network at its optimal threshold keeps "
                    f"<b>{int(row['edges_avg_opt'])}</b> edges — "
                    f"{int(row['edges_avg_opt']) - int(row['edges_avg_conservative'])} more than "
                    f"the conservative threshold, and "
                    f"{int(row['edges_avg_50']) - int(row['edges_avg_opt'])} fewer than pooling "
                    f"everything at strength ≥ 0.5."
                )
                bullets.append(
                    f"Mean held-out CV loss for this layer is "
                    f"<b>{row['cv_loss_mean']:.2f} ± {row['cv_loss_sd']:.2f}</b> across folds."
                )
            else:
                bullets.append(
                    f"{layer_key} wasn't part of the original four-algorithm comparison "
                    "report — see the Model Comparison page for the layers that were."
                )
    else:
        thr = load_bootstrap_thresholds()
        fd = final_dag_edges(layer_key, "bootstrap")
        full = load_edges(layer_key, "bootstrap")
        n_pairs = len(full) // 2
        if not thr.empty:
            row = thr[thr["layer"] == layer_key]
            if not row.empty:
                cons = row.iloc[0]["boot_cons"]
                bullets.append(
                    f"This layer's bootstrap network is thresholded at a conservative "
                    f"arc-strength cutoff of <b>{cons:.2f}</b>, well above the "
                    f"<b>0.50</b> optimal-fit reference line shown on the histogram."
                )
        bullets.append(
            f"At that threshold, <b>{len(fd):,} of {n_pairs:,}</b> candidate arcs "
            f"survive into the final bootstrap-averaged DAG "
            f"(retention {len(fd) / n_pairs:.1%})." if n_pairs else
            "No candidate arcs were evaluated for this layer."
        )
    return bullets


@st.cache_data(show_spinner=False)
def model_comparison_findings() -> list[str]:
    comp = load_model_comparison()
    if comp.empty:
        return []
    bullets = []

    gap = comp["edges_tabu"] - comp["edges_mmhc"]
    row = comp.loc[gap.idxmax()]
    bullets.append(
        f"MM-HC is consistently the most conservative algorithm here; its widest gap "
        f"to tabu search is at <b>{row['network']}</b>, where it finds "
        f"<b>{int(row['edges_mmhc'])}</b> edges versus <b>{int(row['edges_tabu'])}</b> "
        f"for tabu search."
    )

    best = comp.loc[comp["cv_loss_mean"].idxmin()]
    worst = comp.loc[comp["cv_loss_mean"].idxmax()]
    bullets.append(
        f"<b>{best['network']}</b> has the lowest mean held-out CV loss "
        f"({best['cv_loss_mean']:.2f} ± {best['cv_loss_sd']:.2f}); "
        f"<b>{worst['network']}</b> has the highest "
        f"({worst['cv_loss_mean']:.2f} ± {worst['cv_loss_sd']:.2f})."
    )

    density = comp["edges_avg_opt"] / comp["n_nodes"]
    row2 = comp.loc[density.idxmax()]
    bullets.append(
        f"By edges-per-node, <b>{row2['network']}</b> produces the densest averaged "
        f"network ({density.loc[density.idxmax()]:.2f} edges per node)."
    )

    return bullets


@st.cache_data(show_spinner=False)
def scorecard_findings() -> list[str]:
    """Data-driven summary of the held-out flu-vaccine benchmark, computed
    live from all_networks_scorecard.csv (CV rows)."""
    sc = load_scorecard()
    if sc.empty:
        return []
    cv = sc[sc["source"] == "cv"]
    bullets = []

    age = cv.dropna(subset=["wvote_age"]).sort_values("wvote_age", ascending=False)
    if len(age):
        well_supported = age[age["n_root_edges"] > 3]
        thin = age[age["n_root_edges"] <= 3]
        if len(well_supported):
            top = well_supported.iloc[0]
            bullets.append(
                f"On <b>age</b>, the strongest well-supported result is "
                f"<b>{top['network']}</b> ({top['wvote_age']:.2f}, built on "
                f"{int(top['n_root_edges'])} root edges)."
            )
        if len(thin):
            highest_thin = thin.sort_values("wvote_age", ascending=False).iloc[0]
            if len(well_supported) == 0 or highest_thin["wvote_age"] > well_supported.iloc[0]["wvote_age"]:
                bullets.append(
                    f"<b>{highest_thin['network']}</b> scores higher still "
                    f"({highest_thin['wvote_age']:.2f}) but on only "
                    f"{int(highest_thin['n_root_edges'])} root edges — too little "
                    f"structure for that score to be trustworthy on its own."
                )

    sex = cv.dropna(subset=["wvote_sex"])
    if len(sex):
        n_negative = int((sex["wvote_sex"] < 0).sum())
        bullets.append(
            f"<b>{n_negative} of {len(sex)}</b> networks score negative on "
            f"<b>sex</b> under cross-validation — consistent with a measurement "
            f"artifact in this cohort (men show weaker IgG fold-change but not "
            f"weaker peak HAI, so paths tied to each endpoint can disagree by "
            f"construction), not necessarily a modeling failure."
        )
        pos = sex[sex["wvote_sex"] > 0]
        if len(pos):
            bullets.append(
                f"<b>{', '.join(pos['network'])}</b> is the exception, scoring "
                f"positive on sex ({pos.iloc[0]['wvote_sex']:.2f})."
            )

    cmv = cv.dropna(subset=["wvote_cmv"])
    bullets.append(
        f"<b>CMV</b> has the sparsest validation coverage of the three roots: "
        f"only <b>{len(cmv)} of {len(cv)}</b> networks had a scoreable CMV root "
        f"edge under cross-validation."
    )

    return bullets


@st.cache_data(show_spinner=False)
def lit_review_findings() -> list[str]:
    """Data-driven summary of the automated literature-corroboration
    results, computed live from lit_review_module_dashboard_inputs.csv
    (CV rows)."""
    lr = load_lit_review()
    if lr.empty:
        return []
    cv = lr[lr["model"] == "cv"]
    bullets = []

    total_edges = int(cv["edges"].sum())
    total_results = int(cv["edges_with_search_results"].sum())
    bullets.append(
        f"Across the CV networks' final DAGs, <b>{total_results:,} of "
        f"{total_edges:,}</b> edges (<b>{total_results / total_edges:.0%}</b>) "
        f"returned at least one literature hit from the automated search."
    )

    total_support = int(cv["Supported"].sum())
    total_reverse = int(cv["Reversed"].sum())
    total_nodir = int(cv["No directional info"].sum())
    total_conflict = int(cv["Conflicting"].sum())
    bullets.append(
        f"Of edges with a literature match: <b>{total_support:,}</b> agree in "
        f"direction, <b>{total_nodir:,}</b> describe a related but non-directional "
        f"association, <b>{total_reverse:,}</b> match an association but disagree "
        f"on direction, and <b>{total_conflict:,}</b> are flagged as actively "
        f"conflicting."
    )

    cv2 = cv.copy()
    cv2["novel_rate"] = (cv2["No literature match found"] / cv2["edges"]).fillna(0)
    top_novel = cv2.loc[cv2["novel_rate"].idxmax()]
    bullets.append(
        f"<b>{top_novel['layer']}</b> has the highest share of edges with no "
        f"literature match at all (<b>{top_novel['novel_rate']:.0%}</b> of its "
        f"{int(top_novel['edges'])} edges) — the largest pool of candidate novel "
        f"findings, pending closer review."
    )

    return bullets


# --------------------------------------------------------------------------
# Page setup
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Sound Life BN Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

st.sidebar.markdown(
    '<div class="eyebrow" style="color:#8FB8AC;">COHORT REPORT</div>',
    unsafe_allow_html=True,
)
st.sidebar.title("Sound Life Cohort")
st.sidebar.caption("Bayesian-network foundation model — results dashboard")
page = st.sidebar.radio(
    "Navigate",
    ["Overview", "Network Explorer", "Diagnostics", "CV vs Bootstrap",
     "Model Comparison", "Validation", "About / Methods"],
)

MODEL_PAGES = {"Overview", "Network Explorer", "Diagnostics"}
if page in MODEL_PAGES:
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Modeling approach")
    model_label = st.sidebar.radio(
        "Which validated network to explore",
        [MODELS[m]["label"] for m in MODEL_ORDER],
        help=(
            "Cross-validation (CV): k-fold cross-validated structure "
            "learning, averaged network at an algorithm-chosen optimal "
            "threshold. Bootstrap: bootstrap-resampled structure learning, "
            "averaged network at a conservative arc-strength threshold. "
            "Same layers, same candidate node/arc sets, two different "
            "validation strategies — see 'CV vs Bootstrap' for a direct "
            "comparison."
        ),
    )
    model = "cv" if model_label == MODELS["cv"]["label"] else "bootstrap"
else:
    model = "cv"

st.sidebar.markdown("---")
st.sidebar.markdown(
    "[Sound Life raw-data dashboard ↗](https://soundlife-dashboard-ik8gg8mk5yrfogaeveepzh.streamlit.app/)"
)


# --------------------------------------------------------------------------
# OVERVIEW
# --------------------------------------------------------------------------
if page == "Overview":
    st.markdown('<div class="eyebrow">STRUCTURE LEARNING · MULTI-OMICS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-title">A Bayesian-Network Foundation Model<br>for the Sound Life Cohort</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        """
        <div class="hero-sub">Multi-omics cohorts like Sound Life carry thousands of features per
        subject but only on the order of 100 baseline subjects. Rather than a black-box predictive
        model, this project learns Bayesian networks (BNs) over biologically curated node sets,
        producing a directed, queryable, auditable joint probability distribution — a structure
        that can serve as the inferential core of a foundation model and can be interrogated for
        both established and novel relationships between immune aging, clinical phenotype, and
        molecular biology.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="hr-line">', unsafe_allow_html=True)

    st.markdown(
        f'<span class="modality-tag">Showing: {MODELS[model]["label"]} model '
        f'&nbsp;·&nbsp; switch in the sidebar, or see "CV vs Bootstrap" for a '
        f'direct comparison</span>',
        unsafe_allow_html=True,
    )

    n_layers = len(LAYER_META)
    total_final_edges = sum(len(final_dag_edges(k, model)) for k in LAYER_ORDER)
    total_nodes = sum(
        len(set(final_dag_edges(k, model)["from"]) | set(final_dag_edges(k, model)["to"]))
        for k in LAYER_ORDER
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Network layers", n_layers)
    c2.metric(f"Total edges in final DAGs ({MODELS[model]['short']})", f"{total_final_edges:,}")
    c3.metric("Total nodes across layers", f"{total_nodes:,}")

    section_label("Key findings")
    render_cards(overview_findings(model), kind="finding")

    st.markdown("### Layers at a glance")
    rows = []
    for k in LAYER_ORDER:
        fd = final_dag_edges(k, model)
        full = load_edges(k, model)
        nodes = set(fd["from"]) | set(fd["to"])
        n_pairs = len(full) // 2
        rows.append(
            {
                "Layer": k,
                "Description": LAYER_META[k]["label"],
                "Modality": LAYER_META[k]["modality"],
                "Nodes (final DAG)": len(nodes),
                "Edges (final DAG)": len(fd),
                "Candidate arcs evaluated": n_pairs,
                "Retention": (len(fd) / n_pairs) if n_pairs else 0.0,
            }
        )
    layers_df = pd.DataFrame(rows)
    styled = (
        layers_df.style
        .format({"Retention": "{:.0%}"})
        .background_gradient(subset=["Retention"], cmap="Greens", vmin=0, vmax=layers_df["Retention"].max())
        .bar(subset=["Edges (final DAG)"], color="#C9DCD6")
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.markdown("### Success criteria")
    st.markdown(
        """
1. **SC1** — the data-preparation pipeline produces both continuous
   (CLG-ready) and discretized representations of the baseline multi-omics
   matrix as fixed, agent-independent inputs, ingested into `bnlearn` for
   structure learning under both representations.
2. **SC2** — the learned network functions as a *queryable* foundation
   model: conditional-probability queries over clinical, proteomic,
   transcriptomic, and cell-composition nodes return results usable for
   hypothesis generation, not merely a fitted object with no inferential use.
"""
    )
    st.info(
        "Use **Network Explorer** in the sidebar to interactively browse, "
        "filter, and query any of the nine learned networks below, "
        "**CV vs Bootstrap** to see exactly where the two validation "
        "approaches agree and disagree, or **Validation** to see how well "
        "each network holds up against held-out flu outcomes and the "
        "published literature.",
        icon="🕸️",
    )


# --------------------------------------------------------------------------
# NETWORK EXPLORER
# --------------------------------------------------------------------------
elif page == "Network Explorer":
    st.markdown('<div class="eyebrow">INTERACTIVE GRAPH</div>', unsafe_allow_html=True)
    st.title("Network Explorer")

    layer_key = st.sidebar.selectbox(
        "Layer",
        LAYER_ORDER,
        format_func=lambda k: LAYER_META[k]["label"],
    )
    meta = LAYER_META[layer_key]
    st.markdown(
        f'<span class="modality-tag">{meta["modality"]}</span> '
        f'<span class="modality-tag">Model: {MODELS[model]["label"]}</span>',
        unsafe_allow_html=True,
    )
    st.caption(meta["description"])

    section_label(f"Findings for {layer_key} ({MODELS[model]['short']})")
    render_cards(layer_findings(layer_key, model), kind="finding")

    st.sidebar.markdown("#### Filters")
    final_only = st.sidebar.checkbox(
        "Show official averaged final DAG only",
        value=True,
        help=(
            "When checked, shows exactly the edges retained in this "
            f"model's averaged network at its threshold "
            "(in_final_dag = TRUE). Uncheck to explore the full "
            "arc-strength table at a threshold you choose."
        ),
    )

    full_df = load_edges(layer_key, model)
    max_display = 500

    if final_only:
        show_df = final_dag_edges(layer_key, model)
        threshold_note = "official averaged-network threshold"
    else:
        threshold = st.sidebar.slider(
            "Minimum arc strength (bootstrap support)",
            0.0, 1.0, 0.5, 0.01,
        )
        dedup = deduped_directed_edges(layer_key, model)
        show_df = dedup[dedup["strength"] >= threshold]
        threshold_note = f"strength ≥ {threshold:.2f}"
        if len(show_df) > max_display:
            st.sidebar.warning(
                f"{len(show_df):,} candidate arcs meet this threshold — "
                f"showing the top {max_display} by strength for "
                "renderability. Raise the threshold to narrow this down."
            )
            show_df = show_df.head(max_display)

    st.sidebar.markdown("#### Display")
    physics_on = st.sidebar.checkbox("Physics simulation", value=True)
    hierarchical = st.sidebar.checkbox("Hierarchical layout", value=False)

    nodes_in_view = sorted(set(show_df["from"]) | set(show_df["to"]))
    st.markdown('<hr class="hr-line">', unsafe_allow_html=True)
    st.caption(
        f"Showing **{len(show_df)}** edges across **{len(nodes_in_view)}** "
        f"nodes ({threshold_note})."
    )

    # ---- build graph ----
    if len(show_df) == 0:
        st.warning("No edges meet the current filter.")
    else:
        degree = {}
        for _, r in show_df.iterrows():
            degree[r["from"]] = degree.get(r["from"], 0) + 1
            degree[r["to"]] = degree.get(r["to"], 0) + 1

        agraph_nodes = [
            Node(
                id=n,
                label=n,
                size=14 + 3 * degree.get(n, 1),
                color=classify_node(n, layer_key),
                shape="dot",
            )
            for n in nodes_in_view
        ]
        agraph_edges = [
            Edge(
                source=r["from"],
                target=r["to"],
                label="",
                width=1 + 3 * r["strength"],
            )
            for _, r in show_df.iterrows()
        ]
        config = Config(
            width=1000,
            height=650,
            directed=True,
            physics=physics_on,
            hierarchical=hierarchical,
            nodeHighlightBehavior=True,
            highlightColor="#F7A7A6",
            collapsible=False,
            node={"labelProperty": "label"},
        )

        col_graph, col_info = st.columns([2.4, 1])
        with col_graph:
            clicked = agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)
            legend_html = "  ".join(
                f"<span style='color:{c}'>●</span> {node_category_label(c)}"
                for c in [COLOR_ROOT, COLOR_PROTEIN, COLOR_CELLFREQ, COLOR_PATHWAY, COLOR_CLINICAL]
            )
            st.markdown(legend_html, unsafe_allow_html=True)

        with col_info:
            st.markdown("#### Inspect a node")
            manual_pick = st.selectbox(
                "Search / select node",
                [""] + nodes_in_view,
                index=0,
            )
            selected_node = manual_pick or clicked

            if selected_node:
                st.markdown(f"**{selected_node}**")
                if selected_node in ROOT_NODES:
                    st.caption("Root / demographic node (CMV, age group, or sex).")
                parents = full_df[
                    (full_df["to"] == selected_node) & (full_df["in_final_dag"])
                ][["from", "strength", "direction"]].sort_values("strength", ascending=False)
                children = full_df[
                    (full_df["from"] == selected_node) & (full_df["in_final_dag"])
                ][["to", "strength", "direction"]].sort_values("strength", ascending=False)

                st.markdown(f"Parents in final DAG ({len(parents)})")
                st.dataframe(parents, hide_index=True, use_container_width=True)
                st.markdown(f"Children in final DAG ({len(children)})")
                st.dataframe(children, hide_index=True, use_container_width=True)

                with st.expander("All candidate arcs touching this node (strength ≥ 0.1)"):
                    cand = full_df[
                        ((full_df["from"] == selected_node) | (full_df["to"] == selected_node))
                        & (full_df["strength"] >= 0.1)
                    ][["from", "to", "strength", "direction", "in_final_dag"]].sort_values(
                        "strength", ascending=False
                    )
                    st.dataframe(cand, hide_index=True, use_container_width=True)
            else:
                st.caption("Click a node in the graph, or pick one from the search box, to see its parents/children.")

        st.download_button(
            "Download filtered edge list (CSV)",
            show_df.to_csv(index=False).encode("utf-8"),
            file_name=f"{layer_key}_{model}_filtered_edges.csv",
            mime="text/csv",
        )

    net_dir = MODELS[model]["net_dir"]
    avg_suffix = MODELS[model]["avg_suffix"]
    with st.expander("Static reference layout (as originally rendered)"):
        if model == "bootstrap":
            img_col1, img_col2, img_col3 = st.columns(3)
            cons_path = net_dir / f"{layer_key}_avg_conservative.png"
            opt_path = net_dir / f"{layer_key}_{avg_suffix}.png"
            tabu_path = net_dir / f"{layer_key}_tabu.png"
            if opt_path.exists():
                img_col1.image(str(opt_path), caption=f"{layer_key} — Bootstrap Averaged (opt=0.50)", use_container_width=True)
            if cons_path.exists():
                img_col2.image(str(cons_path), caption=f"{layer_key} — Bootstrap Averaged (conservative)", use_container_width=True)
            if tabu_path.exists():
                img_col3.image(str(tabu_path), caption=f"{layer_key} — Tabu search DAG", use_container_width=True)
        else:
            img_col1, img_col2 = st.columns(2)
            avg_path = net_dir / f"{layer_key}_{avg_suffix}.png"
            tabu_path = net_dir / f"{layer_key}_tabu.png"
            if avg_path.exists():
                img_col1.image(str(avg_path), caption=f"{layer_key} — CV Averaged", use_container_width=True)
            if tabu_path.exists():
                img_col2.image(str(tabu_path), caption=f"{layer_key} — Tabu search DAG", use_container_width=True)


# --------------------------------------------------------------------------
# DIAGNOSTICS
# --------------------------------------------------------------------------
elif page == "Diagnostics":
    st.markdown('<div class="eyebrow">MODEL FIT & STABILITY</div>', unsafe_allow_html=True)
    st.title("Diagnostics")
    layer_key = st.sidebar.selectbox(
        "Layer", LAYER_ORDER, format_func=lambda k: LAYER_META[k]["label"]
    )

    st.markdown(
        f'<span class="modality-tag">Model: {MODELS[model]["label"]}</span>',
        unsafe_allow_html=True,
    )

    df = load_edges(layer_key, model)

    dfindings = diagnostics_findings(layer_key, model)
    if dfindings:
        section_label(f"Findings for {layer_key} ({MODELS[model]['short']})")
        render_cards(dfindings, kind="insight")

    st.markdown(f"### {MODELS[model]['short']} arc-strength distribution (interactive)")
    fig = px.histogram(
        df, x="strength", nbins=40,
        title=f"{layer_key} — {MODELS[model]['short']} Edge Strength Distribution",
        labels={"strength": "Edge strength (bootstrap support)"},
    )
    fig.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(family="Inter, sans-serif", color="#16233B"),
        title_font=dict(family="Source Serif 4, serif", size=18),
        bargap=0.05,
    )
    fig.update_traces(marker_color=MODELS[model]["color"])
    fig.add_vline(x=0.5, line_color="#2F6F62", line_dash="solid",
                   annotation_text="opt = 0.50")
    if model == "bootstrap":
        thr = load_bootstrap_thresholds()
        row = thr[thr["layer"] == layer_key]
        if not row.empty:
            cons = float(row.iloc[0]["boot_cons"])
            fig.add_vline(x=cons, line_color="#C1793B", line_dash="dash",
                           annotation_text=f"cons = {cons:.2f}")
    else:
        fig.add_vline(x=0.85, line_color="#C1793B", line_dash="dash",
                       annotation_text="0.85")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "This recreates the arc-strength histogram directly from the "
        "arc table (both direction-rows per candidate pair included, "
        "matching the original static figure)."
    )

    if model == "cv":
        st.markdown("### Held-out CV loss distribution")
        loss_path = MODELS["cv"]["hist_dir"] / f"{layer_key}_loss.png"
        if loss_path.exists():
            st.image(str(loss_path), use_container_width=False)
            st.caption(
                "Per-fold held-out log-likelihood loss (lower = better fit). "
                "Rendered from the original report — raw per-fold values were "
                "not exported alongside the edge table."
            )
        else:
            st.info("No loss-distribution image found for this layer.")
    else:
        st.markdown("### Static reference histogram")
        hist_path = MODELS["bootstrap"]["hist_dir"] / f"{layer_key}_edge_strength.png"
        if hist_path.exists():
            st.image(str(hist_path), use_container_width=False)
            st.caption(
                "Original bootstrap edge-strength histogram with the "
                "optimal-fit (0.50) and conservative reference thresholds "
                "marked. Bootstrap resampling has no held-out CV loss — "
                "see the Model Comparison page for the CV loss benchmark."
            )
        else:
            st.info("No histogram image found for this layer.")

    st.markdown("### Edge-count summary")
    n_final = df["in_final_dag"].sum()
    n_total_pairs = len(df) // 2
    c1, c2, c3 = st.columns(3)
    c1.metric("Candidate pairs evaluated", f"{n_total_pairs:,}")
    c2.metric("Edges in final averaged DAG", f"{n_final:,}")
    c3.metric("Retention rate", f"{n_final / n_total_pairs:.1%}")


# --------------------------------------------------------------------------
# CV VS BOOTSTRAP
# --------------------------------------------------------------------------
elif page == "CV vs Bootstrap":
    st.markdown('<div class="eyebrow">TWO VALIDATION STRATEGIES · SAME LAYERS</div>', unsafe_allow_html=True)
    st.title("CV vs Bootstrap")
    st.markdown(
        """
        <div class="hero-sub">Every layer in this project was learned twice: once with
        <b>k-fold cross-validation</b> (averaged network at an algorithm-chosen optimal
        threshold) and once with <b>bootstrap resampling</b> (averaged network at a
        conservative arc-strength threshold, with a 0.50 optimal-fit line shown for
        reference). Both runs start from the identical candidate node/arc set per layer,
        so any difference in the final DAG reflects the validation strategy, not the
        input data.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="hr-line">', unsafe_allow_html=True)

    section_label("Key findings, across all layers")
    render_cards(cv_vs_bootstrap_overview_findings(), kind="finding")

    st.markdown("### Agreement summary by layer")
    summary_rows = []
    for k in LAYER_ORDER:
        cmp = compare_layer_models(k)
        summary_rows.append(
            {
                "Layer": k,
                "CV edges (final DAG)": len(cmp["cv_pairs"]),
                "Bootstrap edges (final DAG)": len(cmp["bt_pairs"]),
                "Shared, same direction": len(cmp["same_direction"]),
                "Shared, reversed direction": len(cmp["reversed_direction"]),
                "CV-only": len(cmp["cv_only_pairs"]),
                "Bootstrap-only": len(cmp["bt_only_pairs"]),
                "Agreement (Jaccard)": cmp["jaccard"],
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    styled_summary = (
        summary_df.style
        .format({"Agreement (Jaccard)": "{:.0%}"})
        .background_gradient(subset=["Agreement (Jaccard)"], cmap="Greens", vmin=0, vmax=1)
        .bar(subset=["CV edges (final DAG)", "Bootstrap edges (final DAG)"], color="#C9DCD6")
    )
    st.dataframe(styled_summary, use_container_width=True, hide_index=True)

    col_a, col_b = st.columns(2)
    with col_a:
        melt = summary_df.melt(
            id_vars="Layer",
            value_vars=["CV edges (final DAG)", "Bootstrap edges (final DAG)"],
            var_name="Model", value_name="Edges",
        )
        fig_edges = px.bar(
            melt, x="Layer", y="Edges", color="Model", barmode="group",
            title="Final-DAG edge count: CV vs Bootstrap",
            color_discrete_map={
                "CV edges (final DAG)": MODELS["cv"]["color"],
                "Bootstrap edges (final DAG)": MODELS["bootstrap"]["color"],
            },
        )
        fig_edges.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", color="#16233B"),
            title_font=dict(family="Source Serif 4, serif", size=16),
        )
        st.plotly_chart(fig_edges, use_container_width=True)
    with col_b:
        fig_jac = px.bar(
            summary_df, x="Layer", y="Agreement (Jaccard)",
            title="Layer-level agreement between models (Jaccard index)",
            color_discrete_sequence=["#2F6F62"],
        )
        fig_jac.update_layout(
            yaxis_tickformat=".0%",
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", color="#16233B"),
            title_font=dict(family="Source Serif 4, serif", size=16),
        )
        st.plotly_chart(fig_jac, use_container_width=True)

    st.markdown('<hr class="hr-line">', unsafe_allow_html=True)
    st.markdown("### Layer deep-dive")
    layer_key = st.selectbox(
        "Layer", LAYER_ORDER, format_func=lambda k: LAYER_META[k]["label"], key="cvb_layer"
    )
    meta = LAYER_META[layer_key]
    st.markdown(f'<span class="modality-tag">{meta["modality"]}</span>', unsafe_allow_html=True)
    st.caption(meta["description"])

    cmp = compare_layer_models(layer_key)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("CV final-DAG edges", len(cmp["cv_pairs"]))
    c2.metric("Bootstrap final-DAG edges", len(cmp["bt_pairs"]))
    c3.metric("Shared connections", len(cmp["shared_pairs"]))
    c4.metric("Agreement (Jaccard)", f"{cmp['jaccard']:.0%}")

    st.markdown("#### Reference networks, side by side")
    img_col1, img_col2 = st.columns(2)
    cv_avg = MODELS["cv"]["net_dir"] / f"{layer_key}_avg.png"
    bt_avg = MODELS["bootstrap"]["net_dir"] / f"{layer_key}_avg_opt.png"
    if cv_avg.exists():
        img_col1.image(str(cv_avg), caption=f"{layer_key} — CV Averaged", use_container_width=True)
    if bt_avg.exists():
        img_col2.image(str(bt_avg), caption=f"{layer_key} — Bootstrap Averaged (opt)", use_container_width=True)

    st.markdown("#### Arc-strength distributions, side by side")
    hist_col1, hist_col2 = st.columns(2)
    cv_hist = MODELS["cv"]["hist_dir"] / f"{layer_key}_edge_strength.png"
    bt_hist = MODELS["bootstrap"]["hist_dir"] / f"{layer_key}_edge_strength.png"
    if cv_hist.exists():
        hist_col1.image(str(cv_hist), caption=f"{layer_key} — CV edge strength", use_container_width=True)
    if bt_hist.exists():
        hist_col2.image(str(bt_hist), caption=f"{layer_key} — Bootstrap edge strength", use_container_width=True)

    st.markdown("#### Where the two models differ")
    tab_shared, tab_reversed, tab_cvonly, tab_btonly = st.tabs(
        [
            f"Agree, same direction ({len(cmp['same_direction'])})",
            f"Agree, opposite direction ({len(cmp['reversed_direction'])})",
            f"CV only ({len(cmp['cv_only_pairs'])})",
            f"Bootstrap only ({len(cmp['bt_only_pairs'])})",
        ]
    )
    with tab_shared:
        if cmp["same_direction"]:
            same_df = pd.DataFrame(cmp["same_direction"]).sort_values(
                "strength_cv", ascending=False
            )
            st.dataframe(same_df, use_container_width=True, hide_index=True)
        else:
            st.caption("No connections agree on both existence and direction at this layer.")
    with tab_reversed:
        if cmp["reversed_direction"]:
            rev_df = pd.DataFrame(cmp["reversed_direction"]).sort_values(
                "strength_cv", ascending=False
            )
            st.dataframe(rev_df, use_container_width=True, hide_index=True)
            st.caption(
                "Both models find a connection between these node pairs, but disagree "
                "on which node is the cause and which is the effect — treat these as "
                "lower-confidence causal calls."
            )
        else:
            st.caption("No shared connections have conflicting direction at this layer.")
    with tab_cvonly:
        if cmp["cv_only_pairs"]:
            cv_only_df = cmp["cv_final"][
                cmp["cv_final"].apply(
                    lambda r: tuple(sorted([r["from"], r["to"]])) in cmp["cv_only_pairs"], axis=1
                )
            ][["from", "to", "strength", "direction"]]
            st.dataframe(cv_only_df, use_container_width=True, hide_index=True)
            st.caption(
                "These arcs survive cross-validation's threshold but not the "
                "bootstrap's more conservative one."
            )
        else:
            st.caption("Every CV final-DAG connection is also found by Bootstrap at this layer.")
    with tab_btonly:
        if cmp["bt_only_pairs"]:
            bt_only_df = cmp["bt_final"][
                cmp["bt_final"].apply(
                    lambda r: tuple(sorted([r["from"], r["to"]])) in cmp["bt_only_pairs"], axis=1
                )
            ][["from", "to", "strength", "direction"]]
            st.dataframe(bt_only_df, use_container_width=True, hide_index=True)
            st.caption(
                "These arcs survive the bootstrap's conservative threshold but not "
                "cross-validation's optimal-fit threshold."
            )
        else:
            st.caption("Every Bootstrap final-DAG connection is also found by CV at this layer.")


# --------------------------------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------------------------------
elif page == "Model Comparison":
    st.markdown('<div class="eyebrow">ALGORITHM BENCHMARK</div>', unsafe_allow_html=True)
    st.title("Model Comparison")
    st.info(
        "This page benchmarks structure-learning **algorithms** (hill-climbing, "
        "tabu search, MM-HC) within the CV pipeline. For a head-to-head "
        "comparison of the **CV vs Bootstrap validation strategies**, see "
        "**CV vs Bootstrap** in the sidebar.",
        icon="ℹ️",
    )
    comp = load_model_comparison()

    if comp.empty:
        st.warning("No model-comparison data available.")
    else:
        st.caption(
            "Algorithm comparison (hill-climbing, tabu search, MM-HC, and "
            "bootstrap-averaged networks at three thresholds) for the "
            "layers included in the original comparison report. L3a, L3b, "
            "and L_all were not part of that report and are shown "
            "elsewhere in this dashboard using their own edge tables."
        )

        section_label("Findings")
        render_cards(model_comparison_findings(), kind="insight")

        st.markdown("### Raw comparison table")
        st.dataframe(comp, use_container_width=True, hide_index=True)

        edge_cols = ["edges_hc", "edges_tabu", "edges_mmhc", "edges_avg_opt",
                     "edges_avg_conservative", "edges_avg_50"]
        melt = comp.melt(id_vars="network", value_vars=edge_cols,
                          var_name="method", value_name="n_edges")
        palette = ["#4e79a7", "#2F6F62", "#af7aa1", "#f28e2b", "#C1793B", "#e15759"]
        fig1 = px.bar(
            melt, x="network", y="n_edges", color="method", barmode="group",
            title="Edge count by structure-learning method",
            color_discrete_sequence=palette,
        )
        fig1.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", color="#16233B"),
            title_font=dict(family="Source Serif 4, serif", size=18),
        )
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            x=comp["network"], y=comp["cv_loss_mean"],
            error_y=dict(type="data", array=comp["cv_loss_sd"]),
            name="CV held-out loss (mean ± SD)",
            marker_color="#2F6F62",
        ))
        fig2.update_layout(
            title="Cross-validated held-out loss by layer",
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", color="#16233B"),
            title_font=dict(family="Source Serif 4, serif", size=18),
        )
        st.plotly_chart(fig2, use_container_width=True)

        fig3 = px.bar(
            comp, x="network", y="bic_tabu",
            title="BIC of tabu-search network by layer",
            color_discrete_sequence=["#C1793B"],
        )
        fig3.update_layout(
            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            font=dict(family="Inter, sans-serif", color="#16233B"),
            title_font=dict(family="Source Serif 4, serif", size=18),
        )
        st.plotly_chart(fig3, use_container_width=True)


# --------------------------------------------------------------------------
# VALIDATION  (new — held-out benchmark + literature corroboration)
# --------------------------------------------------------------------------
elif page == "Validation":
    st.markdown('<div class="eyebrow">DOES THE LEARNED STRUCTURE REFLECT REAL BIOLOGY?</div>', unsafe_allow_html=True)
    st.title("Validation")
    st.markdown(
        """
        <div class="hero-sub">Since this project has no single prediction target, every
        network is checked two independent ways: a <b>held-out benchmark</b> against real
        flu-vaccine outcomes never seen during training, and an automated <b>literature
        corroboration</b> pass over every recovered edge. Both are shown here for both
        validation strategies (CV and Bootstrap) side by side.</div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<hr class="hr-line">', unsafe_allow_html=True)

    tab_benchmark, tab_litreview = st.tabs(["Held-out Benchmark (Flu Response)", "Literature Review"])

    # ---------------- Held-out benchmark tab ----------------
    with tab_benchmark:
        sc = load_scorecard()
        if sc.empty:
            st.warning(
                "No scorecard data found. Expected at "
                f"`{SCORECARD_PATH.relative_to(ROOT)}`."
            )
        else:
            st.markdown(
                "Each root variable (age, sex, CMV) is scored from **-1** "
                "(every learned root → feature → flu-outcome path the network "
                "learned contradicts literature-established expectation) to "
                "**+1** (every path agrees), weighted by edge strength × "
                "|effect size|. `NaN` means the network had no root edge to "
                "score for that root."
            )
            section_label("Key findings")
            render_cards(scorecard_findings(), kind="finding")

            root_choice = st.radio(
                "Root variable", ["Age", "Sex", "CMV"], horizontal=True, key="val_root"
            )
            wvote_col = {"Age": "wvote_age", "Sex": "wvote_sex", "CMV": "wvote_cmv"}[root_choice]

            plot_df = sc.dropna(subset=[wvote_col]).copy()
            order = (
                plot_df[plot_df["source"] == "cv"]
                .sort_values(wvote_col, ascending=False)["network"]
                .tolist()
            )
            remaining = [n for n in plot_df["network"].unique() if n not in order]
            order = order + remaining

            plot_df["source_label"] = plot_df["source"].map(
                {"cv": MODELS["cv"]["label"], "bootstrap": MODELS["bootstrap"]["label"]}
            )
            fig = px.bar(
                plot_df, x="network", y=wvote_col, color="source_label",
                barmode="group", category_orders={"network": order},
                title=f"{root_choice} validation score by network",
                labels={wvote_col: f"wvote_{root_choice.lower()} (-1 to +1)", "network": "Network"},
                color_discrete_map={
                    MODELS["cv"]["label"]: MODELS["cv"]["color"],
                    MODELS["bootstrap"]["label"]: MODELS["bootstrap"]["color"],
                },
                hover_data=["n_root_edges", "n_subjects"],
            )
            fig.update_layout(
                yaxis_range=[-1.05, 1.05],
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(family="Inter, sans-serif", color="#16233B"),
                title_font=dict(family="Source Serif 4, serif", size=16),
                legend_title_text="",
            )
            fig.add_hline(y=0, line_color="#8C9BAF", line_dash="dot",
                           annotation_text="coin flip")
            st.plotly_chart(fig, use_container_width=True)
            st.caption(
                "Hover a bar to see how many root edges and subjects it's based "
                "on — a high score built on very few edges (roughly ≤ 3) rests "
                "on too little structure to trust on its own."
            )

            st.markdown("### Raw scorecard")
            display_cols = ["network", "source", "label", "n_subjects", "n_root_edges",
                             "n_flu_relevant", "agree_all", "agree_relevant",
                             "wvote_age", "wvote_sex", "wvote_cmv"]
            st.dataframe(
                sc[display_cols].sort_values(["network", "source"]),
                use_container_width=True, hide_index=True,
            )
            st.download_button(
                "Download full scorecard (CSV)",
                sc.to_csv(index=False).encode("utf-8"),
                file_name="all_networks_scorecard.csv",
                mime="text/csv",
            )

    # ---------------- Literature review tab ----------------
    with tab_litreview:
        lr = load_lit_review()
        if lr.empty:
            st.warning(
                "No literature-review data found. Expected at "
                f"`{LIT_REVIEW_PATH.relative_to(ROOT)}`."
            )
        else:
            st.markdown(
                "Every edge in a network's final DAG was automatically searched "
                "against **INDRA** and **PubMed** and classified by whether the "
                "literature agrees on both the association and its direction."
            )
            section_label("Key findings")
            render_cards(lit_review_findings(), kind="finding")

            lr_model_label = st.radio(
                "Validation strategy", [MODELS["cv"]["label"], MODELS["bootstrap"]["label"]],
                horizontal=True, key="val_lr_model",
            )
            lr_model = "cv" if lr_model_label == MODELS["cv"]["label"] else "bootstrap"
            lr_view = lr[lr["model"] == lr_model].sort_values("edges", ascending=False)

            class_cols = ["Supported", "No directional info", "Reversed",
                          "Conflicting", "No literature match found"]
            melt = lr_view.melt(
                id_vars="layer", value_vars=class_cols,
                var_name="Classification", value_name="Edges",
            )
            fig_lr = px.bar(
                melt, x="layer", y="Edges", color="Classification", barmode="stack",
                title=f"Edge literature-classification breakdown — {MODELS[lr_model]['label']}",
                category_orders={"layer": lr_view["layer"].tolist()},
                color_discrete_map={
                    "Supported": COLOR_SUPPORTED,
                    "No directional info": COLOR_NODIR,
                    "Reversed": COLOR_REVERSED,
                    "Conflicting": COLOR_CONFLICT,
                    "No literature match found": COLOR_NOMATCH,
                },
            )
            fig_lr.update_layout(
                plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                font=dict(family="Inter, sans-serif", color="#16233B"),
                title_font=dict(family="Source Serif 4, serif", size=16),
                legend_title_text="",
            )
            st.plotly_chart(fig_lr, use_container_width=True)

            st.markdown("### Raw literature-review rollup")
            display_cols_lr = ["layer", "model", "edges", "edges_with_search_results"] + class_cols
            st.dataframe(
                lr[display_cols_lr].sort_values(["layer", "model"]),
                use_container_width=True, hide_index=True,
            )
            st.caption(
                "`No literature match found` = edges with zero hits in either "
                "source — the project's candidate-novel-finding pool, pending "
                "closer manual review rather than automated classification. "
                "Note: for L3a, the four classification totals (Supported + "
                "No directional info + Reversed + Conflicting) don't sum "
                "exactly to `edges_with_search_results` (off by 1 under CV, "
                "3 under Bootstrap) — a small discrepancy in the source data "
                "worth checking against the underlying classification "
                "pipeline before treating L3a's breakdown as fully reconciled."
            )
            st.download_button(
                "Download full literature-review rollup (CSV)",
                lr.to_csv(index=False).encode("utf-8"),
                file_name="lit_review_module_dashboard_inputs.csv",
                mime="text/csv",
            )


# --------------------------------------------------------------------------
# ABOUT / METHODS
# --------------------------------------------------------------------------
else:
    st.markdown('<div class="eyebrow">METHODOLOGY</div>', unsafe_allow_html=True)
    st.title("About this project")
    st.markdown(
        """
### Purpose

Use open-source multi-omics data from the **Sound Life Cohort** to learn
Bayesian networks (BNs) capable of serving as the basis for a foundation
model, producing an interpretable joint probability distribution over
clinical, proteomic, transcriptomic, and cell-composition variables. Edges
of the learned DAGs are intended to frame causal-inference queries. Building
analytic applications on top of the resulting model is explicitly **out of
scope**.

### Data

All data derive from **Gong et al. 2025**, a longitudinal study of healthy
adults in two age groups (25–35 and 55–65), profiled across two consecutive
influenza seasons. Six modalities are used: clinical labs & metadata,
influenza serology/HAI (held out as a validation target), plasma proteomics
(Olink Explore 1536), single-cell cell-type frequencies (AIFI L1/L2/L3),
pseudobulk scRNA-seq, and unstimulated whole-blood bulk RNA-seq. Every
modality is reduced to one row per subject at Flu Year 1 / Day 0 baseline
before entering the training matrix, since bnlearn's structure-learning
algorithms require independent observations.

### Node selection

Nodes are chosen from **prior biological literature**, not from variance
or correlation computed within this dataset, to preserve a non-circular
structural prior. Three fixed subject-level characteristics — **CMV
serostatus, age group, and biological sex** — are treated as root nodes.

### Modeling

Structure learning uses hill-climbing, tabu search, and MM-HC algorithms
scored by BIC/BDe. Every layer is learned under **two independent
validation strategies**, from the identical candidate node/arc set:

- **Cross-validation (CV).** K-fold cross-validated structure learning;
  the averaged network is thresholded at an algorithm-chosen optimal
  arc-strength cutoff per layer.
- **Bootstrap.** Bootstrap-resampled structure learning; the averaged
  network is thresholded at a conservative arc-strength cutoff per layer
  (well above the shared 0.50 "optimal-fit" reference line), trading some
  recall for extra stability.

Continuous nodes are evaluated under a conditional linear Gaussian (CLG)
representation against a fully discretized alternative. See **CV vs
Bootstrap** in the sidebar for a layer-by-layer comparison of where the
two strategies agree and disagree on structure.

### Evaluation

Two complementary validation approaches are used, since the project has
no single prediction target — both are shown in full on the **Validation**
page:

1. **Held-out benchmark (influenza vaccine response).** Serology and HAI
   titers are excluded from training (structured missingness) and instead
   used to check whether the fitted network's conditional-probability
   queries agree in direction and approximate magnitude with
   literature-established relationships (e.g., older age and CMV
   seropositivity suppressing vaccine response). Scored per root variable
   (age, sex, CMV) and per network as a weighted vote from -1 to +1.
2. **Edge-level literature corroboration.** Each recovered edge is
   evaluated by an automated search against INDRA and PubMed and
   classified as direction-supported, direction-reversed, related with no
   directional claim, actively conflicting, or unmatched (a candidate
   novel finding), per network and per validation strategy.

### Key assumptions & limitations

- The baseline sample size (~100 subjects) is small relative to the
  80–120 target node count per layer, motivating literature-driven node
  selection and bootstrap thresholding rather than data-driven feature
  selection.
- Literature-based node selection is necessary to preserve a valid,
  non-circular prior on network structure.
- Influenza vaccine response, though held out of training, is assumed to
  be an appropriate benchmark for validating whether the learned network
  reflects established immunology.
- CV and Bootstrap use different threshold philosophies (algorithm-chosen
  optimum vs a fixed conservative cutoff), so a lower edge count from one
  strategy does not by itself mean it is "wrong" — see **CV vs
  Bootstrap** for where the two agree, and treat direction-reversed
  connections between the two as lower-confidence causal calls.
- A high held-out benchmark score built on very few root edges (roughly
  ≤ 3) rests on too little structure to be trustworthy on its own — check
  `n_root_edges` on the **Validation** page before treating a score at
  face value.
- The literature-corroboration pipeline reports what automated search
  found, not a definitive verdict — "no literature match found" is a
  candidate-novel-finding pool pending closer manual review, not a
  confirmed discovery.

### Links

- [Sound Life raw-data exploration dashboard](https://soundlife-dashboard-ik8gg8mk5yrfogaeveepzh.streamlit.app/)
"""
    )