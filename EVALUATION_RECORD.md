# Evaluation Record

Date: 2026-07-15

## Commands Run

```bash
python evaluate.py --checkpoint baseline_ckpt.pt --text_file ..\data\dev_eval.txt
```

Run from `llm_handout/starter`.

Result:

```json
{"bpb": 2.3718, "n_params": 1339840, "steps": 2000, "tokens_in_eval": 159225, "tokens_scored": 159224}
```

```bash
python evaluate.py --checkpoint ckpt.pt --text_file data\dev_eval.txt
```

Run from `plivo-ml-24CH10033`.

Result:

```json
{"bpb": 2.026, "n_params": 1935104, "steps": 2000, "tokens_in_eval": 58894, "tokens_scored": 58893}
```

## Candidate Decision

| Candidate | Status | Result |
|---|---|---|
| `codex_plivo_speedrun_submission/ckpt.pt` | Rejected | Failed to load with its current `model.py`; state dict architecture mismatch. |
| `plivo-ml-24CH10033/ckpt_run1.pt` | Rejected | Incompatible with current source and worse in temporary reconstruction testing: 3.5448 BPB with RoPE, 3.7891 without RoPE. |
| `plivo-ml-24CH10033/ckpt.pt` | Selected | Valid scorer run: 2.026 BPB, 1,935,104 parameters, 2,000 steps. |

## Final Submission

The GitHub push uses `plivo-ml-24CH10033/ckpt.pt` with the matching `model.py`, `tokenizer.py`, `train.py`, `evaluate.py`, `RUNLOG.md`, `NOTES.md`, and `SUMMARY.html`.
