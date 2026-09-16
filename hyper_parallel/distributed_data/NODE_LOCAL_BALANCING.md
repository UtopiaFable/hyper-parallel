# Node-local balancing

`enable_dp_balance=True` selects one fixed pipeline: default backbone FLOPs,
capacity-constrained LPT, node-local Gloo exchange, automatic one-step buffering,
and final H2D on a copy stream. No separate backend, buffer, or planner options
are needed. The default is false. Existing native loading is unchanged; the
dataset facade described below preserves source bins and uses ordinary
consumer-stream device transfer when balancing is disabled.

## Dataset-based integration

Construct the config, bind the source's existing data contract once, and build
the loader. The application does not implement a balancing wrapper, a pack
adapter, logging, or a device-prefetch consumption hook.

```python
from hyper_parallel.distributed_data import (
    DistributedDatasetConfig,
    build_distributed_dataset,
    build_distributed_dataloader,
)

config = DistributedDatasetConfig(
    seq_len=seq_len,
    local_batch_size=microbatches_per_step,
    packing_budgets=packing_budgets,
    enable_dp_balance=True,
)
dataset = build_distributed_dataset(
    source,
    metadata="metadata",       # SampleMetadata already produced by the transform
    collate_fn=model_collator,  # Existing collator, called once per accepted bin
    cpu_fields=("cu_seqlens",),
    log_fields=("P", "D"),     # Optional additive metadata.features fields
)
with build_distributed_dataloader(
    dataset, mesh, config,
    model_config=model_config,
    device=device,
    max_steps=train_steps,
) as loader:
    for microbatches in loader:
        for batch in microbatches:
            train_microbatch(batch)
```

- `DistributedDataset` wraps a selected-step source, not a new file reader.
  It does not replace a model's processor, sampler or token-budget pack selector.
  Each source output is `[[sample, ...], ...]`, with one bin per microbatch.
  Use the original loader with final collation disabled to retain its selection
  behavior. A flat map-style dataset alone does not define those step boundaries.
- `metadata` accepts either an existing `sample -> SampleMetadata` callable,
  or a mapping field name containing precomputed `SampleMetadata`. A named
  metadata field is omitted from the dictionaries passed to the collator.
  The source is not consumed during dataset or loader construction.
- `cpu_fields` names top-level fields of the collated mapping. Their complete
  subtrees stay on CPU; other tensor leaves move recursively. Metadata and
  collation remain application semantics, not model names embedded in Hyper.
- `log_fields` selects numeric `SampleMetadata.features` to sum per bin.
  Generic sample/sequence/cost and send/receive logs need no extra callback.
  Configure the application's Python logging to include INFO messages.
- Iteration returns ready device microbatches. The loader waits on the copy
  event and records storage on the consumer stream internally. Do not call
  `take_device_microbatch()` or repeat application-side H2D on this facade.
  `loader.last_host_batch` retains the corresponding CPU microbatches for
  optional host-only metering without D2H. Moving previously CPU-only metering
  to the device outputs may introduce synchronization; use that host view if
  needed. A dataloader cannot remove device-wide waits from an existing trainer.

## Runtime contract

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
  CPU-only execution keeps host batches. Use compute-stream synchronization,
  not device-wide synchronization, to avoid draining the next step's copy.
- This opt-in requires pure DP, equal step counts and a raw-step source.
  It does not support loader checkpoint/resume or native shared-metadata reads.
  The dataset facade does not add checkpoint/resume, including when balancing
  is disabled. Use the original native entry for checkpointable loading.

## Existing integrations

The original `external_step_source`, `metadata_fn`, `pack_fn`, `move_fn` and
`bin_stats_fn` arguments remain supported for existing users. That entry still
yields CPU views and uses `take_device_microbatch()` for staged device inputs.
Do not mix those arguments with a `DistributedDataset`: the dataset owns all
data callbacks. Plain Dataset + native BatchSampler loading is unchanged.

A runnable CPU example is
`examples/torch/distributed_data/external_dataset.py`.
