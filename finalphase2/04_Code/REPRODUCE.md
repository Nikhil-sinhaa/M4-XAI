# Reproduce Phase 2 (from this folder)

Analysis only, no GPU, seconds (uses the committed curves in `../05_Results_Data`):

    pip install numpy pandas matplotlib scipy        # or: uv sync --frozen  (uses pyproject.toml / uv.lock)
    python analyze_baselines.py --rescore_root ../05_Results_Data/rescore --vision_root ../05_Results_Data/vision
    python analyze_baselines.py --rescore_root ../05_Results_Data/text_rescore --ref pad --vision_root ""

Re-scoring (needs a GPU, the M4 attribution maps from Phase 1 in `results/m4/vision` and `results/m4/text`, and the model weights):

    python rescore_baselines.py --models resnet50 --limit 500 --amp          # vision, 5 baselines
    python rescore_text_baselines.py --models all --limit 200 --amp          # text, [PAD]/[MASK]/[UNK]

Report: `python build_report.py` then convert `report.html` to PDF. See `README.md` for the full project.
