# First competition GAM implementation plan

Approved spec: `../specs/2026-09-23-competition-gam-v1-design.md`.

- [x] Implement one standalone command for the eight inputs, fixed models, three checks, diagnostics and submission.
- [x] Verify mixture weighting, log-target metrics and unseen-salt encoding with focused tests; run existing tests (11 passed).
- [x] Run once on the supplied files and independently verify predictions, metrics, splits and submission (5,959 rows; all nine RMSE values recomputed).
- [x] Inspect diagnostic plots, report limitations, and document the command.

Results: `runs/competition-gam-01/report.md`. Descriptor GAM RMSE: supplied 0.62522, DMC 0.23757, EMC 0.26350. No fit warnings. Diagnostic plots show a large error in the extreme low-conductivity tail and structured residuals; these remain limitations of the fixed first model.
