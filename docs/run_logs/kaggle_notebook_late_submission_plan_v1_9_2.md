# Kaggle Notebook Late Submission Plan — SpineScoutX v1.9.2

> Research-only · non-commercial · not diagnostic · not clinically validated.

## What v1.9.1 got wrong

v1.9.1 used `kaggle competitions submit -f submission.csv`, which is the **direct CSV
file-upload path**. This competition (`rsna-2024-lumbar-spine-degenerative-classification`)
is a **code competition (Kaggle Notebook)** and does not accept plain CSV uploads via API.
The correct late-submission path is:

```bash
kaggle competitions submit \
  -c rsna-2024-lumbar-spine-degenerative-classification \
  -k arashsajjadi/<NOTEBOOK_SLUG> \
  -v <VERSION> \
  -f submission.csv \
  -m "SpineScoutX v1.9 notebook late submission"
```

Where:
- `-k` = the kernel (notebook) user/slug reference
- `-v` = the kernel version number (integer)
- `-f submission.csv` = the NAME of the output file produced by the kernel,
  NOT a local file to upload

## What this sprint does

1. Build a Kaggle Dataset containing model weights + SpineScoutX wheel.
2. Create a Kaggle Script (Python kernel) that:
   - Reads competition test images from `/kaggle/input/rsna-.../test_images/`
   - Loads the SpineScoutX wheel and model tarball from the dataset
   - Runs full 5-route inference (canal/foraminal/subarticular)
   - Writes `/kaggle/working/submission.csv`
   - Validates it
3. Push the kernel to Kaggle and run it.
4. Submit the kernel output using `-k` and `-v`.
5. Report the score honestly.

## What this sprint is NOT

- Not a training sprint (no model is retrained).
- Not an accuracy sprint (internal metrics unchanged).
- Not an official competition entry (late submission).
- Not leaderboard tuning (one clean submission).

## Safety constraints

| Rule | Action |
|---|---|
| No Kaggle token printed/committed | Token read from `api_kaggle.txt`, never printed |
| No competition DICOMs committed | All data in `data/raw/rsna/` is gitignored |
| No model weights committed | Weights are in gitignored `outputs/real/` |
| No Kaggle dataset metadata with secrets | `dataset-metadata.json` has no tokens |
| One submission only | No repeated leaderboard tuning |
| Honest score reporting | If rejected, document the exact reason |

## Expected phases

| Phase | Description |
|---|---|
| 0 | Branch, plan (this file) |
| 1 | Auth check + kernel/dataset capability |
| 2 | Create Kaggle Dataset with model assets |
| 3 | Build and push Kaggle notebook/script |
| 4 | Wait for notebook run |
| 5 | Submit with -k and -v |
| 6 | Score comparison |
| 7 | README update |
| 8 | Gates, PR, merge, tag |
