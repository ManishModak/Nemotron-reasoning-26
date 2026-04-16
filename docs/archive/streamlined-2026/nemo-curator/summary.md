# NVIDIA NeMo Curator

**Overview:** A GPU-accelerated data curation toolkit designed to prepare high-quality datasets for LLMs, VLMs, and multimodal models.

*   **Supported Modalities:**
    *   **Text:** Deduplication (Exact/Fuzzy/Semantic), quality filtering (30+ filters), and language detection.
    *   **Image:** Aesthetic filtering, NSFW detection, and embedding generation.
    *   **Video:** Scene detection, clip extraction, and motion filtering.
    *   **Audio:** ASR transcription, quality assessment, and WER filtering.
*   **Performance:** Leverages NVIDIA RAPIDS and Ray for multi-node scaling. Benchmarks show 16x faster fuzzy deduplication and 40% lower TCO compared to CPU-based methods.
*   **Integration:** Part of the NVIDIA NeMo suite, designed to scale from laptops to multi-node H100 clusters.

**Source:** [NVIDIA-NeMo/Curator GitHub](https://github.com/NVIDIA-NeMo/Curator)