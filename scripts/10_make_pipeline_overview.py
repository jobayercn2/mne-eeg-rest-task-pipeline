from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# Output
# ============================================================

PROJECT_ROOT = Path(".")
FIGURES_DIR = PROJECT_ROOT / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

OUT_FILE = FIGURES_DIR / "pipeline_overview.png"


# ============================================================
# Style
# ============================================================

COLORS = {
    "rest": "#EAF2FF",
    "task": "#EAF8EC",
    "merge": "#FFF4D8",
    "analysis": "#F3E9FF",
    "neutral": "#F7F7F7",
    "docs": "#FCEEEE",
    "results": "#EAF7F5",
    "availability": "#F8F2E8",
    "edge": "#2F3A4A",
    "title": "#111827",
    "text": "#374151",
}

TITLE_FS = 24
SUBTITLE_FS = 12
SECTION_FS = 12
BOX_TITLE_FS = 10
BOX_TEXT_FS = 8
FOOTER_FS = 9


# ============================================================
# Helpers
# ============================================================

def add_box(ax, x, y, w, h, title, text="", facecolor="white",
            title_size=BOX_TITLE_FS, text_size=BOX_TEXT_FS):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.03,rounding_size=0.08",
        linewidth=1.2,
        edgecolor=COLORS["edge"],
        facecolor=facecolor
    )
    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h * 0.68,
        title,
        ha="center",
        va="center",
        fontsize=title_size,
        fontweight="bold",
        color=COLORS["title"],
        wrap=True
    )

    if text:
        ax.text(
            x + w / 2,
            y + h * 0.34,
            text,
            ha="center",
            va="center",
            fontsize=text_size,
            color=COLORS["text"],
            wrap=True
        )


def add_arrow(ax, x1, y1, x2, y2, rad=0.0):
    patch = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle="-|>",
        mutation_scale=13,
        linewidth=1.2,
        color=COLORS["edge"],
        connectionstyle=f"arc3,rad={rad}"
    )
    ax.add_patch(patch)


def add_section(ax, x, y, label):
    ax.text(
        x, y, label,
        ha="left", va="center",
        fontsize=SECTION_FS,
        fontweight="bold",
        color=COLORS["title"]
    )


# ============================================================
# Figure canvas
# ============================================================

fig, ax = plt.subplots(figsize=(18, 11))
ax.set_xlim(0, 18)
ax.set_ylim(0, 11)
ax.axis("off")


# ============================================================
# Title
# ============================================================

ax.text(
    9, 10.6,
    "Computational EEG Rest–Task Analysis Pipeline",
    ha="center",
    va="center",
    fontsize=TITLE_FS,
    fontweight="bold",
    color=COLORS["title"]
)

ax.text(
    9, 10.2,
    "Repository structure and workflow overview for a reproducible MNE-Python EEG project",
    ha="center",
    va="center",
    fontsize=SUBTITLE_FS,
    color=COLORS["text"]
)


# ============================================================
# Section labels
# ============================================================

add_section(ax, 0.6, 9.4, "Resting-state workflow")
add_section(ax, 0.6, 7.5, "RDK task workflow")
add_section(ax, 0.6, 5.4, "Inputs and intermediate outputs")
add_section(ax, 0.6, 4.1, "Repository organization")
add_section(ax, 9.2, 4.1, "Public outputs")


# ============================================================
# Row 1: Resting-state workflow
# ============================================================

y_rest = 8.55
w = 2.55
h = 0.9

add_box(ax, 0.6, y_rest, w, h,
        "01_preprocess_resting.py",
        "Filter, notch,\nreference, epoch",
        COLORS["rest"])

add_box(ax, 3.45, y_rest, w, h,
        "02_qc_resting.py",
        "Epoch retention\nPSD checks",
        COLORS["rest"])

add_box(ax, 6.30, y_rest, w, h,
        "03_extract_resting_features.py",
        "Band power +\nspecparam features",
        COLORS["rest"])

add_arrow(ax, 3.15, y_rest + h/2, 3.45, y_rest + h/2)
add_arrow(ax, 6.00, y_rest + h/2, 6.30, y_rest + h/2)


# ============================================================
# Row 2: Task workflow
# ============================================================

y_task = 6.65

add_box(ax, 0.6, y_task, w, h,
        "04_inspect_rdk_events.py",
        "Status channel\ntrigger codes",
        COLORS["task"])

add_box(ax, 3.45, y_task, w, h,
        "05_preprocess_rdk_task.py",
        "Event epoching\nbaseline correction",
        COLORS["task"])

add_box(ax, 6.30, y_task, w, h,
        "06_extract_task_features.py",
        "Condition features\ncoherent − random",
        COLORS["task"])

add_arrow(ax, 3.15, y_task + h/2, 3.45, y_task + h/2)
add_arrow(ax, 6.00, y_task + h/2, 6.30, y_task + h/2)


# ============================================================
# Row 3: Merge / analysis / ML
# ============================================================

y_merge = 7.55

add_box(ax, 9.30, y_merge, 2.35, 1.0,
        "07_merge_rest_task.py",
        "Merge resting and\ntask features",
        COLORS["merge"])

add_box(ax, 12.00, y_merge, 2.50, 1.0,
        "08_exploratory_analysis.py",
        "Correlations,\nscatterplots,\nheatmap",
        COLORS["analysis"])

add_box(ax, 14.95, y_merge, 2.00, 1.0,
        "09_ml_demo_loocv.py",
        "LOOCV regression\nML demo",
        COLORS["analysis"])

# arrows from branches into merge
add_arrow(ax, 8.85, y_rest + h/2, 9.30, y_merge + 0.68, rad=0.08)
add_arrow(ax, 8.85, y_task + h/2, 9.30, y_merge + 0.32, rad=-0.08)

# arrows between merge -> analysis -> ml
add_arrow(ax, 11.65, y_merge + 0.5, 12.00, y_merge + 0.5)
add_arrow(ax, 14.50, y_merge + 0.5, 14.95, y_merge + 0.5)


# ============================================================
# Inputs and intermediate outputs
# ============================================================

y_mid = 5.55

add_box(ax, 0.8, y_mid, 2.8, 0.95,
        "Input data",
        "data/raw/resting_state\nand data/raw/task",
        "white")

add_box(ax, 4.0, y_mid, 2.8, 0.95,
        "Processed data",
        "data/processed/resting_state\nand data/processed/task",
        "white")

add_box(ax, 7.2, y_mid, 2.8, 0.95,
        "Feature tables",
        "resting_state_features\nand task_features",
        "white")

# NEW: separate explanation boxes for merge / analysis / ml
add_box(ax, 10.45, y_mid, 2.2, 0.95,
        "Merged outputs",
        "merged_rest_&\n Task _features",
        "white")

add_box(ax, 13.10, y_mid, 2.2, 0.95,
        "Analysis outputs",
        "Statistical_analysis",
        "white")

add_box(ax, 15.75, y_mid, 1.9, 0.95,
        "ML outputs",
        "basic_ml tables\nand figures",
        "white")

# arrows upward from explanations
add_arrow(ax, 2.2, y_mid + 0.95, 1.9, y_task + 0.15)
add_arrow(ax, 5.4, y_mid + 0.95, 4.8, y_task + 0.15)
add_arrow(ax, 8.6, y_mid + 0.95, 7.6, y_task + 0.15)

# merged outputs -> merge box
add_arrow(ax, 11.55, y_mid + 0.95, 10.50, y_merge + 0.02)

# analysis outputs -> exploratory analysis box
add_arrow(ax, 14.20, y_mid + 0.95, 13.25, y_merge + 0.02)

# ml outputs -> ml box
add_arrow(ax, 16.70, y_mid + 0.95, 15.95, y_merge + 0.02)


# ============================================================
# Bottom: Repository structure
# ============================================================

repo_text = (
    "computational-eeg-rest-task-analysis-pipeline/\n"
    "├── README.md\n"
    "├── requirements.txt\n"
    "├── .gitignore\n"
    "├── scripts/\n"
    "├── docs/\n"
    "├── results/\n"
    "│   ├── tables/\n"
    "│   └── figures/\n"
    "└── figures/\n"
    "    └── pipeline_overview.png"
)

add_box(ax, 0.8, 1.45, 6.5, 3.0,
        "Repository structure",
        repo_text,
        COLORS["neutral"],
        title_size=11,
        text_size=8)

docs_text = (
    "docs/\n"
    "• project_summary.md\n"
    "• methodology.md\n"
    "• qc_decisions.md\n"
    "• interpretation_notes.md"
)

add_box(ax, 6.2, 1.95, 3.4, 1.6,
        "Documentation",
        docs_text,
        COLORS["docs"],
        title_size=11,
        text_size=8)

results_text = (
    "results/\n"
    "• descriptive_statistics.csv\n"
    "• rest_task_correlations.csv\n"
    "• ml_model_performance.csv\n"
    "• ridge_coefficients.csv\n"
    "• correlation_heatmap.png\n"
    "• selected scatterplots\n"
    "• selected ML figures"
)

add_box(ax, 10.1, 1.65, 4.2, 1.9,
        "Selected public outputs",
        results_text,
        COLORS["results"],
        title_size=11,
        text_size=8)

availability_text = (
    "No raw EEG\n"
    "No processed EEG\n"
    "No participant metadata"
)

add_box(ax, 14.8, 1.75, 2.9, 1.8,
        "Data availability",
        availability_text,
        COLORS["availability"],
        title_size=11,
        text_size=8)


# ============================================================
# Footer
# ============================================================

footer = (
    "Public repository excludes raw and processed EEG data. "
    "Focus: reproducible workflow, responsible QC, interpretable features, and cautious exploratory analysis."
)

ax.text(
    9, 0.65,
    footer,
    ha="center",
    va="center",
    fontsize=FOOTER_FS,
    color=COLORS["text"],
    style="italic"
)

plt.tight_layout()
plt.savefig(OUT_FILE, dpi=300, bbox_inches="tight")
plt.close()

print(f"Saved pipeline overview to: {OUT_FILE.resolve()}")