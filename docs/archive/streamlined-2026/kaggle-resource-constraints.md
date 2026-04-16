# Kaggle Resource Constraints & Compute Strategy

## Hardware Available
1. **Standard Sessions (T4)**: 16 GB VRAM, 5-9h runtime limits, strict weekly quota.
2. **Special Allocation (G4 VMs)**: NVIDIA RTX PRO 6000 Blackwell GPUs available for ~30 weeks. Significantly faster with more VRAM, but subject to timeouts and disk limits.

## Compute Strategy
- **Maximize Signal per GPU-hour**: 
  - Real LoRA/GRPO training is feasible, especially on G4.
  - Must stay lightweight: rank ≤32, 4-bit/8-bit quantization.
  - Utilize Unsloth/Axolotl memory tricks.
- **Hybrid Approach**: 
  - Full NeMo stack (Docker-based) is powerful on G4 but risky/heavy on T4. Use a hybrid setup blending NeMo tools with lightweight alternatives depending on the VM assigned.

## Recommended Frameworks for Constraints
- **Axolotl**: YAML-driven, uses Unsloth backend. Extremely lightweight, reproducible, explicitly allowed.
- **Distilabel**: Faster/cheaper synthetic pipeline on T4 compared to full NeMo DataDesigner.
- **SGLang & TensorRT-LLM**: Essential for Notebook 4 (Inference) on G4 Blackwells to maximize tokens/sec under the 9-hour limit.
- **DSPy**: High accuracy lift via prompt optimization with almost zero extra GPU cost.
- **torch.compile + FlashAttention-2/3**: Free 10-30% speedup on G4.
