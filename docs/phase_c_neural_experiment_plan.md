# Phase C Neural Experiment Plan

## Purpose

Phase C extends the rolling historical simulation with PyTorch sequence
models: MLP, RNN, GRU, and LSTM. The first GPU stage is a screening sweep, not
the final neural experiment. It compares a compact set of architectures and
hyperparameters cheaply enough to inspect before spending GPU time on monthly
rolling refits.

## Safeguards Added Before Screening

- Each sequence window must contain consecutive calendar months.
- Feature scaling and target scaling are fitted only on the training portion
  of each historical as-of window.
- Early stopping and `ReduceLROnPlateau` scheduling are retained.
- Completed model configurations are checkpointed as chunk artifacts.
- Rerunning with `resume: true` loads finished chunks and trains only missing
  configurations.
- Chunk identifiers include a runner-contract version so preprocessing changes
  do not silently reuse artifacts created under older semantics.
- Complexity profiling estimates neural parameter burden from input width,
  hidden width, layers, recurrent gate count, and sequence length where
  applicable.
- Neural runs can branch across training-window-safe feature policies:
  `none`, `variance_pruned`, `corr_pruned`, and `mutual_info_top_30`.
- Neural runs can branch across `sequence_raw`, `sequence_pca_20`, and
  `sequence_pca_95` representations. PCA is fitted after feature selection
  inside each historical training window.

## Screening Funnel

The reviewable screening config is:

```text
experiment_configs/phase_c_neural_screening.yaml
```

It currently contains:

- `4` model builds: MLP, RNN, GRU, and LSTM
- `4` curated parameter sets per build
- `3` representative feature families
- raw and residual modes
- `40` quarterly as-of dates from January 2016 through December 2025
- `96` model configurations and `3,840` estimated fits

The first GPU screening config deliberately uses `feature_policy = none` and
`representation_policy = sequence_raw`. This isolates architecture, sequence
length, hidden width, layers, dropout, and learning rate. After shortlisting
promising neighborhoods, expand the selected candidates across the implemented
feature-policy and PCA branches.

The policy/PCA contract smoke test is:

```bash
python run_neural_models.py \
  --experiment-config experiment_configs/phase_c_neural_policy_smoke.yaml
```

Inspect the expanded count before launch:

```bash
python plan_neural_experiment.py \
  --config experiment_configs/phase_c_neural_screening.yaml
```

## Colab Workflow

After cloning or pulling the repository in Colab:

```bash
cd /content/transit-ml-pipeline
python -m pip install -r requirements-neural.txt
```

The local feature-store artifacts are intentionally ignored by git. Copy the
two required files into the cloned repository:

```bash
mkdir -p /content/transit-ml-pipeline/feature_store/income_interactions_h3_v1
cp /content/feature_store/income_interactions_h3_v1/feature_table.parquet \
  /content/transit-ml-pipeline/feature_store/income_interactions_h3_v1/
cp /content/feature_store/income_interactions_h3_v1/feature_families.json \
  /content/transit-ml-pipeline/feature_store/income_interactions_h3_v1/
```

For a disposable smoke test, local `/content` output is fine. Before the
screening sweep, point `outputs.results_base_uri`, the chunk directory, and the
checkpoint directory at a mounted Google Drive folder or periodically copy the
output folder to Drive. Colab's `/content` filesystem disappears when the
runtime is recycled.

Launch only after reviewing the planner output:

```bash
python run_neural_models.py \
  --experiment-config experiment_configs/phase_c_neural_screening.yaml
```

## Decisions After Screening

- Select finalist architectures and narrow their hyperparameter neighborhoods.
- Rerun finalists with monthly as-of dates for dashboard-comparable results.
- Run finalist seeds more than once to measure neural variance.
- Compare PCA and feature-policy branches for the shortlisted architectures,
  especially on the wider neural feature families.
- Add shortlisted-model attribution, such as permutation importance, rather
  than calculating expensive explanations for every screening configuration.
- Consider finer per-as-of checkpoints if finalist configurations become long
  enough that losing one partially completed configuration is costly.
