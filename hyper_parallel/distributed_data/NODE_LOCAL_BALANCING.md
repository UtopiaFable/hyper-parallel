# Node-local balancing

`enable_dp_balance=True` selects one fixed pipeline: default backbone FLOPs,
capacity-constrained LPT, node-local Gloo exchange, automatic one-step buffering,
and final H2D on a copy stream. No separate backend, buffer, or planner options
are needed. The default is false and retains the original data path.

```python
config = DistributedDatasetConfig(
    seq_len=seq_len,
    local_batch_size=microbatches_per_step,
    packing_budgets=packing_budgets,
    enable_dp_balance=True,
)
loader = build_distributed_dataloader(
    None, mesh, config,
    external_step_source=source,
    metadata_fn=metadata_fn,
    pack_fn=pack_fn,
    collate_fn=list,
    model_config=model_config,
    move_fn=move_microbatch_to_device,
    bin_stats_fn=summarize_bin,
    max_steps=train_steps,
)
try:
    for cpu_microbatches in loader:
        for index, cpu_microbatch in enumerate(cpu_microbatches):
            batch = (loader.take_device_microbatch(index)
                     if loader.prefetches_to_device else cpu_microbatch)
            train_microbatch(batch)
finally:
    loader.close()
```

- Each source yield contains `local_batch_size` non-empty raw-sample bins.
  Source sampling, worker count and `prefetch_factor` remain source concerns.
- Omit `cost_model` to construct `DefaultCostModel(model_config)` automatically.
  An explicit callback replaces it. Missing default-model dimensions are an
  error, never a fallback to metadata cost. Supply the actual `mlp_layer_types`.
- Default-model features are per-sample `P`, `D`, and conditional-image token
  runs for blockwise attention. `P + D == pack_tokens`. Packing budgets remain
  hard limits separate from predicted FLOPs.
- Accept a candidate only if its `(maximum cost, variance, range)` improves.
  Failed packing or an unchanged score retains the original sample layout.
- Only global rank zero logs its node's before/after packs, transfers and cost.
- H2D selects the current NPU/CUDA device automatically; `device` can specify it.
  `move_fn(microbatch, device)` preserves application-specific CPU metadata.
  CPU-only execution keeps host batches. Use compute-stream synchronization,
  not device-wide synchronization, to avoid draining the next step's copy.
- This opt-in requires pure DP, equal step counts and an external raw-step source.
  It does not support loader checkpoint/resume or native shared-metadata reads.
  Leave the switch off to use the original checkpointable native path.
