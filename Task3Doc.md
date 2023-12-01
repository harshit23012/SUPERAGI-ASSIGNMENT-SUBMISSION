# Documentation for Training Loop Adaptation

## Single GPU Training Loop:

When executed on a machine with a single GPU, the main function calls the train function with rank 0 and world size 1. The model is simply moved to the GPU without DDP or FSDP wrapping.

## Distributed Data Parallel (DDP):

For multiple GPUs, `torch.multiprocessing.spawn` is used to initiate multiple processes, each running a copy of the model on its own GPU. In the train function, the model is wrapped with DDP which synchronizes gradients across different GPUs. The setup and cleanup functions are used to initialize and destroy the process group, which is necessary for DDP.

## Fully Sharded Data Parallel (FSDP):

FSDP is enabled by setting `fsdp` to `True` in the main function. In the train function, when `fsdp` is `True`, the model is wrapped with FSDP instead of DDP. FSDP shards the model parameters, gradients, and optimizer state across all GPUs, reducing the memory footprint on each GPU.

This training loop allows for seamless transitioning between single GPU, DDP, and FSDP setups, making it versatile for different hardware configurations and scaling needs. Users can switch between these modes by simply adjusting the `fsdp` flag and the number of available GPUs.
