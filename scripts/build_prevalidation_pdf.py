from pathlib import Path
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "reports" / "dataset_prevalidation.md"
output = ROOT / "reports" / "dataset_prevalidation.pdf"

raw = source.read_text(encoding="utf-8")
paragraphs = []
for block in raw.split("\n\n"):
    clean = re.sub(r"[*`]", "", block).strip()
    clean = re.sub(r"^#+\s*", "", clean)
    if clean:
        paragraphs.append(clean.replace("\n", " "))

with PdfPages(output) as pdf:
    fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
    y = .95
    for i, paragraph in enumerate(paragraphs):
        heading = i == 0 or paragraph in {
            "Proposed client question", "Why the dataset fits the project",
            "Evaluation design", "Key validity guardrail",
        }
        width = 76 if heading else 103
        lines = textwrap.wrap(paragraph, width=width)
        height = .027 * len(lines) + (.016 if heading else .01)
        if y - height < .06:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
            fig = plt.figure(figsize=(8.27, 11.69), facecolor="white")
            y = .95
        fig.text(
            .08, y, "\n".join(lines),
            fontsize=18 if i == 0 else (12 if heading else 10.5),
            weight="bold" if heading else "normal",
            color="#0A1F44" if heading else "#222222",
            va="top", linespacing=1.35,
        )
        y -= height
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)

print(output)

