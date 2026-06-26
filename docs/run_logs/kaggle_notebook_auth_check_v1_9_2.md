# Kaggle Auth + Notebook Capability Check — v1.9.2

> Research-only · non-commercial · not diagnostic.

## Auth status

Source: `~/.kaggle/kaggle.json` (pre-existing; chmod 600)
Username: `arash` (Kaggle: arashsajjadi)
Token: NOT printed/committed — verified present and working

```
$ kaggle competitions submit --help
```

Confirmed flags: `-k/--kernel`, `-v/--version`, `-f/--file`, `-m/--message`

## Existing kernels

```
$ kaggle kernels list --mine
```

| Ref | Last Run | Status |
|---|---|---|
| arashsajjadi/ndr-negative-results-pitfalls | 2026-06-25 | COMPLETE |
| arashsajjadi/simple-fine-tuning-baseline | 2026-06-25 | COMPLETE |
| arashsajjadi/chat-gpt-sentiment-analysis | 2023 | — |
| arashsajjadi/analysis | 2021 | — |

None of the existing kernels are competition-specific. A new kernel will be created
for this submission.

## Notebook submission capability

`kaggle competitions submit` with `-k KERNEL -v VERSION -f OUTPUT_FILE` is the
correct late-submission path for this code competition. The `-f` argument is the
OUTPUT FILE NAME from the kernel (e.g. `submission.csv`), NOT a local file upload.

The competition slug is: `rsna-2024-lumbar-spine-degenerative-classification`

## Conclusion

Auth: ✅
Kernel push capability: ✅
Code-competition submit path: ✅

Proceeding to Phase 2 (Kaggle Dataset with model assets).
