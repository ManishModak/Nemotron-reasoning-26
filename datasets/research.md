**Here’s my curated list of the *best* Opus 4.6 and 4.5 reasoning / thinking / CoT datasets** (the ones I deem highest-value for fine-tuning based on community usage, size/quality balance, cleanliness, SFT-readiness, and how often top models cite them).  

I prioritized:
- High downloads + frequent use in strong distilled models (e.g., Qwen3.5 variants)
- Filtered/cleaned data (no refusals, consistent format)
- Full “thinking” traces + problem + solution
- Diversity (math, logic, code, general reasoning)
- High-effort / adaptive reasoning where possible

All are **exclusively on Hugging Face** (no GitHub mirrors or raw files).

### Top Opus 4.6 Reasoning Datasets (Feb–Apr 2026 releases — still the gold standard)

| Rank | Dataset | Size | Why I deem it **good** | Best for | Link |
|------|---------|------|------------------------|----------|------|
| 1 | **nohurry/Opus-4.6-Reasoning-3000x-filtered** | ~3,000 (filtered) | Most popular by far (500+ likes, used in nearly every major Qwen3.5-Opus distill). Cleaned version of the original Crownelius set — removes refusals/unusable entries. Excellent problem → thinking → solution format. | All-rounder starting point | https://huggingface.co/datasets/nohurry/Opus-4.6-Reasoning-3000x-filtered |
| 2 | **Roman1111111/claude-opus-4.6-10000x** | 10,000 | Largest high-fidelity set. Strong focus on math (GSM8K/MATH), logic puzzles, and multi-step instructions with hidden CoT traces. Frequently mixed into v2 fine-tunes for scale. | Volume + math/logic depth | https://huggingface.co/datasets/Roman1111111/claude-opus-4.6-10000x |
| 3 | **Farseen0/opus-4.6-reasoning-sft-12k** (or **ykarout/Opus-4.6-reasoning-sft-12k**) | ~12,000 | Pre-cleaned & unified SFT-ready merge from 4+ Opus 4.6 sources (including Roman 10k + Crownelius). Ready to load and train immediately — conversational format. | Plug-and-play fine-tuning | https://huggingface.co/datasets/Farseen0/opus-4.6-reasoning-sft-12k |
| 4 | **TeichAI/Claude-Opus-4.6-Reasoning-887x** | 887 | Premium **high-reasoning-effort** set (Bullshit Bench, legal/life decisions, vague prompts). Very high quality per example — the “deep thinking” version. | Quality over quantity | https://huggingface.co/datasets/TeichAI/Claude-Opus-4.6-Reasoning-887x |
| 5 | **dalisoft/claude-opus-4.6-high-reasoning-700x** | 700 | Adaptive high-reasoning effort. Great complement to the main sets for extra depth on tough prompts. | Mixing with larger sets | https://huggingface.co/datasets/dalisoft/claude-opus-4.6-high-reasoning-700x |
| 6 | **Crownelius/Opus-4.6-Reasoning-2100x-formatted** | ~2,160 | TeichAI-style formatted version of Crownelius data. Clean and easy to merge. | Structural consistency | https://huggingface.co/datasets/Crownelius/Opus-4.6-Reasoning-2100x-formatted |

**My recommendation for 4.6**: Start with **nohurry 3k-filtered** (or the newer Crownelius original if you want the absolute latest filter) + mix in **Roman 10k** or the **12k SFT** for scale. Add **TeichAI 887x** for high-effort boost.

### Top Opus 4.5 Reasoning Datasets (still excellent, slightly older but very high quality)

| Dataset | Size | Why I deem it **good** | Best for | Link |
|---------|------|------------------------|----------|------|
| **TeichAI/claude-4.5-opus-high-reasoning-250x** | 250 | The go-to high-reasoning set for 4.5. Used in tons of distilled models. Deep, structured thinking traces. | Premium small-scale distillation | https://huggingface.co/datasets/TeichAI/claude-4.5-opus-high-reasoning-250x |
| **Crownelius/Opus-4.5-3000x** | ~3,000 | Direct 4.5 counterpart to the 4.6 3k set. Balanced across reasoning, math, and creative. | Larger volume for 4.5 | https://huggingface.co/datasets/Crownelius/Opus-4.5-3000x |

**Note**: 4.5 datasets are less actively used now (4.6 is superior), but the **TeichAI 250x** is still a favorite for its pure high-effort quality.

### Other Good Recent Reasoning/CoT Datasets (2026 releases, non-Opus but similar quality & style)

These are strong alternatives or complements if you want variety or massive scale:
- **ianncity/KIMI-K2.5-1000000x** → 1,000,000 reasoning traces (50% coding). Huge distillation dataset from KIMI-K2.5 high-reasoning. Great for scale.  
  Link: https://huggingface.co/datasets/ianncity/KIMI-K2.5-1000000x
- **opendatalab/ChartVerse-SFT-1.8M** → 1.8M chart-reasoning examples with rich CoT. Recently trending #1 on HF. Excellent for visual/logical reasoning.  
  Link: https://huggingface.co/datasets/opendatalab/ChartVerse-SFT-1.8M
- **voidful/gemini-3.1-opus-4.6-reasoning-merged** → Merged Gemini + Opus 4.6 style reasoning (used in several strong fine-tunes).

**Quick tip**: The **12k SFT-ready Opus 4.6 merges** are currently the sweet spot for most people doing fine-tunes right now — they combine the best of the above without you having to merge manually.
