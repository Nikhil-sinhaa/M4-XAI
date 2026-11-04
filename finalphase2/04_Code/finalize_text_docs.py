"""
Update the Phase 2 documents once the TEXT re-scoring results exist (run by scripts/finish_text_local.sh).

Reads phase2_results/text_rescore/summary/TEXT_FINDINGS.json (written by phase2_report/build_report.py, every number in
it is computed from the CSVs) and edits README.md, research_gap.md, the two video scripts and AI_USAGE.md.
Idempotent: a document that already mentions rescore_text_baselines.py / the text findings is left alone.

    python scripts/finalize_text_docs.py
"""

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
F = json.loads((ROOT / "phase2_results" / "text_rescore" / "summary" / "TEXT_FINDINGS.json").read_text())
N, CH, CASES = F["n_models"], F["n_changed"], F["cases"]
names = ", ".join(v["name"] for v in F["models"].values())
docs = ", ".join(f"{v['name']} {v['docs']}" for v in F["models"].values())
rev = "; ".join(f"{a} under {b}" for a, b in F["reversed"]) or "none"
bss = F["bss_family"]
FACT = (f"The [PAD] ranking changes under [MASK] or [UNK] in {CH} of {N} text models; complete reversals: {rev}. "
        f"ABPC is lower under [MASK] than under [PAD] in {F['mask_lower']} of {CASES} (model, method) cases "
        f"(mean {F['mean_pad_minus_mask']:+.3f}) and lower under [UNK] in {F['unk_lower']} of {CASES} (mean {F['mean_pad_minus_unk']:+.3f}). "
        f"BSS: IG/SG/LIME {bss['ig_sg_lime'][0]:.3f} on average, attention-based {bss['attention'][0]:.3f}.")


def edit(path, fn):
    p = ROOT / path
    old = p.read_text()
    new = fn(old)
    if new != old:
        p.write_text(new)
        print("updated", path)
    else:
        print("unchanged", path)


def readme(s):
    if "rescore_text_baselines.py" in s:
        return s
    s = s.replace("| MobileNet (sample-size analysis) |",
                  f"| {names} (text) | IG, SG, LIME + best GA / BT-head / BT-token start layer | first 200 MovieReview validation documents that align with the saved maps (N per model in the report) | 3 masking tokens: [PAD] (M4), [MASK], [UNK] |\n| MobileNet (sample-size analysis) |", 1)
    s = s.replace("**Not done:** text models ([PAD]/[MASK]/[UNK]), ViT-L, PScore/SynScore under other baselines. Conclusions cover these 8 vision models only.",
                  "**Not done:** ViT-L, PScore/SynScore under other baselines. Conclusions cover these 8 vision models and 5 text models only.")
    s = s.replace("| `analyze_baselines.py` |", "| `rescore_text_baselines.py` | Text counterpart of `rescore_baselines.py`: re-scores the saved token maps (`results/m4/text/<model>/<variant>/maps`) with [PAD], [MASK] and [UNK]. Documents whose saved map no longer matches the tokenization are left out (`docs.txt`). |\n| `analyze_baselines.py` |", 1)
    s = s.replace("# 2. analysis (CPU, seconds).", "# 1b. text models (200 documents, ~30 runs; bf16 forward passes)\n.venv/bin/python rescore_text_baselines.py --models all --limit 200 --amp\n.venv/bin/python analyze_baselines.py --rescore_root results/m4/text_rescore --ref pad --vision_root \"\"\n# 2. analysis (CPU, seconds).", 1)
    s = s.replace("\n### Earlier Phase 2 scripts", f"- **Text models ({N}):** {FACT}\n\n### Earlier Phase 2 scripts", 1)
    return s


def gap(s):
    if "rescore_text_baselines.py" in s:
        return s
    s = s.replace("- **Sample-size analysis**:", f"- **Text models**: {names} (IG, SG, LIME and the best GA / BT-head / BT-token start layer; the 200 MovieReview validation documents whose saved map matches the tokenization: {docs}) re-scored with [PAD] (M4), [MASK] and [UNK] by `rescore_text_baselines.py`.\n- **Sample-size analysis**:", 1)
    s = s.replace("**Not done (stated limitations):** ViT-L, all text models and text baselines, PScore/SynScore", "**Not done (stated limitations):** ViT-L, PScore/SynScore", 1)
    s = s.replace("Results therefore cover 8 vision models and small image subsets,", "Results therefore cover 8 vision models, 5 text models and small subsets,", 1)
    s = s.replace("conclusions about M4's text models, or about ViT-L, cannot be drawn from this work.", "conclusions about ViT-L cannot be drawn from this work.", 1)
    return s


def notes(s):
    if "Text models" in s:
        return s
    s = s.replace("- Sample size: a random 500-image subset", f"- **Text models ({N}, 200 documents):** {FACT} (report section 7.6)\n- Sample size: a random 500-image subset", 1)
    s = s.replace("vision only (no text, no ViT-L)", "no ViT-L", 1)
    s = s.replace("Future work: text baselines ([PAD]/[MASK]/[UNK]), ViT-L,", "Future work: ViT-L,", 1)
    s = s.replace("it says nothing about the text models.", "text models were tested only for [PAD]/[MASK]/[UNK].", 1)
    return s


def spoken(s):
    if "section 7.6 of the report" in s:
        return s
    para = (f"\n\nWe did the same for text. On {N} language models, we replaced the [PAD] token that M4 uses by [MASK] and [UNK]. "
            f"The ranking changed in {CH} of the {N} models, and ABPC was lower under [MASK] than under [PAD] in {F['mask_lower']} of {CASES} cases. This is in section 7.6 of the report.")
    s = s.replace("\n\n[page 9] Finally, sample size.", para + "\n\n[page 9] Finally, sample size.", 1)
    s = s.replace("vision models only, on small subsets. We did not run the text experiments or ViT-Large.", "vision and text models only, on small subsets. We did not run ViT-Large.", 1)
    s = s.replace("Future work is the text baselines, ViT-Large,", "Future work is ViT-Large,", 1)
    return s


def usage(s):
    if "rescore_text_baselines" in s:
        return s
    add = ("\n3. **Prompt**: (same session) \"full complete text model for 200 for phase 1 and 2\", \"do all on l40\"\n"
           "   - **Output**: `rescore_text_baselines.py` (text re-scoring under [PAD]/[MASK]/[UNK]), a parallel job queue on the GPU server, text support in `analyze_baselines.py`, a text section in the report.\n"
           "   - **Modification / checks**: verified that [PAD] re-scoring reproduces M4's stored ABPC for GA (within 0.0003); found that some saved Phase 1 maps are one token longer than the current tokenization and excluded those documents instead of guessing; checked bf16 vs fp32 (at most 0.003 ABPC).\n")
    return s.replace("\n---\n\n## How AI Output Was Modified", add + "\n---\n\n## How AI Output Was Modified", 1)


edit("README.md", readme)
edit("research_gap.md", gap)
edit("phase2_report/VIDEO_SCRIPT.md", notes)
edit("phase2_report/VIDEO_SCRIPT_FULL.md", spoken)
edit("AI_USAGE.md", usage)
