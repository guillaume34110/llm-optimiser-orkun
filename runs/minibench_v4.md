# Orkun minibench report

- checkpoint: `kaggle/_out_v4_burdok/sft_procedural/ckpts/best.safetensors`
- config: `configs/orkun_sft_procedural_v4.yaml`
- temperature: 0.45  ·  pass@4
- generated: 2026-06-14 14:58:49

## Capability — GoalFamily pass rates

**greedy 0/9**  ·  **sampled(pass@4) 1/9 families**

| family | greedy | greedy graded | sampled pass@k |
|---|---|---|---|
| write_file | · | 0.5 | 0/4 |
| fix_token | · | 0.0 | 0/4 |
| compute | · | 0.5 | 0/4 |
| json_transform | · | 0.5 | 0/4 |
| arith | · | 0.5 | 0/4 |
| sequence | · | 0.5 | 0/4 |
| bool_sat | · | 0.0 | 0/4 |
| pipeline | · | 0.75 | 2/4 |
| echo | · | 0.5 | 0/4 |

## Induction — teacher-forced copy fidelity

`copy_acc` = fraction of payload tokens predicted correctly under forced decoding. `prefix_len` = correct leading tokens before first miss. `exact_rate` = whole-payload reproductions. A trained induction head holds `copy_acc≈1.0` flat across lengths; decay here = pre-induction backbone.

| payload len | trials | copy_acc | prefix_len | exact_rate |
|---|---|---|---|---|
| 2 | 8 | 0.812 | 1.38 | 0.75 |
| 4 | 8 | 0.604 | 1.0 | 0.25 |
| 8 | 8 | 0.576 | 1.25 | 0.0 |
| 12 | 8 | 0.541 | 1.0 | 0.0 |
