# Amazon ML Challenge 2026: Business Entity Resolution

For every Source 1 business record, find all Source 2 / Source 3 records that refer to the same real-world business.
Scored by macro-averaged F0.5 per Source 1 entity.

## Files

| File | Purpose |
|---|---|
| `kaggle_entity_resolution.ipynb` | Kaggle notebook: the full pipeline |
| `kaggle_entity_resolution.py` | Same code as a script with `# %%` cells (edit this one) |
| `build_notebook.py` | Rebuilds the `.ipynb` from the `.py`: `python build_notebook.py` |

## Pipeline

1. **Normalise:** transliterate/strip accents, expand abbreviations, drop legal suffixes, and turn website names into plain names.
   Also extract house number, postcode and number tokens from the address.
2. **Block (per country):** char-trigram TF-IDF with exact top-K cosine on the GPU, using four blockers:
   name, address, name+address, and consonant-skeleton name (for transliterations).
3. **Features:** rapidfuzz name/address similarities, digit agreement, missing-field flags, and each candidate's gap to the best candidate for the same S1.
4. **Model:** LightGBM binary classifier, validated on a split by S1 entity.
5. **Decide:** one-to-one assignment (each S2/S3 record goes to at most one S1), then a probability threshold tuned for macro F0.5.
6. **Output:** `output/matching_results.tsv` and `output/candidate_pairs.tsv`, then run the official validator.

## Run on Kaggle

1. Upload `student_resource` as a private Kaggle Dataset and add it as notebook input.
2. Accelerator **GPU P100** (or **T4 x2** if PyTorch rejects sm_60), Internet on, Persistence "Files only".
3. Smoke test with `TRAIN_S1=50_000`, then run with the full config via *Save Version, Save & Run All*.

The dataset is not included in this repository.

## Challenge material

* `challenge/`: problem statement, guidelines, step-by-step guide PDF, documentation template and the official `utils/validate_submission.py`.
* **Dataset:** the full `student_resource` zip (1.09 GB: train/test TSVs) is attached to the
  [`dataset` release](../../releases/tag/dataset), because GitHub rejects files over 100 MB in the repo.
  Download it and upload it to Kaggle as a private dataset.
