# Student-test paired comparison — preliminary

Code commit: `2856389243d92372461a9f2d07e8d7cb7be47ae8`. Artifact SHA-256: `89fb2ad254614e0269fbdd4a45538692535fdbd0a506e6be33556de49e54e5e5`. Dataset ledger SHA-256: `f267ab7a04611834e01dc65d0de9c6cd1eb161243347bf579b49ecda28bf3e5c`.

Configurations and file hashes were saved in `plan.json` before this comparison. The 10 student-test trips were not used to fit this artifact. Trip 00000052 had been inspected in earlier routing experiments, so this partition is not fully untouched. All metrics below are agreement with approximate inferred labels, not verified accuracy. Candidate misses count as wrong in the confident-label denominator.

| Scorer | Decoder | Agreements / confident labels | Proxy agreement | Unmatched | Reset events |
|---|---|---:|---:|---:|---:|
| Distance | classifier_only | 4,972 / 9,118 | 54.53% | 0 | 0 |
| Distance | midpoint | 4,584 / 9,118 | 50.27% | 0 | 753 |
| Distance | projected | 7,797 / 9,118 | 85.51% | 0 | 126 |
| Heading | classifier_only | 7,601 / 9,118 | 83.36% | 0 | 0 |
| Heading | midpoint | 6,867 / 9,118 | 75.31% | 0 | 753 |
| Heading | projected | 7,790 / 9,118 | 85.44% | 0 | 126 |

Of 19,160 observations, 9,118 have confident proxy labels, 8,529 are ambiguous and 1,513 unavailable; 16 confident labels are missing from candidates. No-candidate observations: 0. Trip 00000076 has no usable labels due to a reference discontinuity.

| Trip | Observations | Confident | Ambiguous | Unavailable | Misses | Distance + projected agreements | Heading + projected agreements |
|---|---:|---:|---:|---:|---:|---:|---:|
| 00000000 | 2503 | 1970 | 533 | 0 | 2 | 1921 | 1908 |
| 00000007 | 1354 | 1036 | 318 | 0 | 0 | 934 | 934 |
| 00000009 | 1049 | 171 | 878 | 0 | 12 | 122 | 122 |
| 00000040 | 2995 | 1969 | 1026 | 0 | 0 | 1883 | 1879 |
| 00000052 | 262 | 115 | 147 | 0 | 0 | 115 | 115 |
| 00000056 | 3958 | 1323 | 2608 | 27 | 0 | 1229 | 1224 |
| 00000065 | 768 | 234 | 534 | 0 | 1 | 225 | 224 |
| 00000076 | 1486 | 0 | 0 | 1486 | 0 | 0 | 0 |
| 00000077 | 3911 | 1985 | 1926 | 0 | 1 | 1120 | 1128 |
| 00000091 | 874 | 315 | 559 | 0 | 0 | 248 | 256 |

Distance+projected and heading+projected differ by 7 agreements overall; do not claim a meaningful advantage from this small difference. The same projected continuity policy is used for the audit and decoder, so its continuity counts are not independent evidence of route correctness. The labeling method uses distance and excludes uncertain points, introducing selection bias. Comparisons are provisional and do not establish the organizer’s official 95% metric. If these results are used to revise model/decoder settings, student-test becomes development data; reserve another untouched set for final assessment.

Detailed per-trip timings, projected and midpoint continuity audits, events and examples are in `comparison/*.json`. The separately generated per-observation ledgers are in `ledger/`. Both are local and Git-ignored.

Reproduction:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m src.models.heldout_evaluation '../Student folder/student_test' --plan outputs/heldout_v1/plan.json --artifact outputs/model_work/heading_comparison_v1/provisional_heading_model.joblib --output-dir outputs/heldout_v1/comparison_reproduction
```

Ask organizers for the official pointwise or route metric, denominator, and whether observation-level labels exist. No message was sent.
