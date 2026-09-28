# LiPF6 adaptive search plan

Approved design: `../specs/2026-09-23-lipf6-gam-search-design.md`.

- [ ] Extend evaluator JSON with optional validated smoothing settings and the agreed error measures; verify defaults and propagation with one focused test.
- [ ] Record the baseline, then choose and record each next hypothesis before fitting; at most 24 candidates on unchanged development splits.
- [ ] Save candidate scores, residual diagnostics, chronological decisions, and the selected configuration; verify prediction IDs exclude the holdout.
- [ ] Summarize improvements, variation, tail errors, publication differences, and limitations in a report.
