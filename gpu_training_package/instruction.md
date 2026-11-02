Train and evaluate an NYC taxi route activity forecasting model with the bundled
NYC TLC yellow taxi records in `/app/data`. Use a real PyTorch Geometric TGN on
NVIDIA CUDA. All timestamps and splits must be chronological.

1. Verify input integrity, document the dataset and define non-overlapping
   training, validation, and held-out test periods. Preserve an embargo before
   the train and validation boundaries so trips used as labels have completed.
2. Train on the training period, validate after each epoch, and save the
   checkpoint with the best validation loss. Record finite losses and GPU details.
3. Evaluate the saved model and the recent and seasonal historical baselines on
   the same full held-out test period and candidate routes. Save the predictions
   and metrics so the verifier can recompute sampled route MRR independently.
4. Report the comparison honestly. The metric gate passes only if the TGN's MRR
   exceeds both historical baselines; lower fare error or higher business profit
   is not implied.

Write training, split, checkpoint, assessment and evaluation files to
`/app/artifacts`, and a report to `/app/reports/report.md`. The default solution
entrypoint is `bash /app/solution/solve.sh`.
