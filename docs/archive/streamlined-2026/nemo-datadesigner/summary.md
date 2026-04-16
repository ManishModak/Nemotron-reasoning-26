# NVIDIA NeMo DataDesigner

**Overview:** A framework for generating high-quality synthetic datasets from scratch or using seed data, moving beyond simple LLM prompting.

*   **Core Capabilities:**
    *   **Diverse Generation:** Uses statistical samplers, LLMs, or seed datasets.
    *   **Field Control:** Manages relationships and correlations between data fields.
    *   **Validation:** Built-in Python, SQL, and custom validators to ensure data quality.
    *   **Scoring:** Uses "LLM-as-a-judge" for automated quality assessment.
*   **Quick Start:** Installable via `pip install data-designer`; supports providers like NVIDIA Build API, OpenAI, and OpenRouter.
*   **Agent Integration:** Includes a specialized skill for coding agents (like Claude Code) to automate schema design and generation.

**Source:** [NVIDIA-NeMo/DataDesigner GitHub](https://github.com/NVIDIA-NeMo/DataDesigner)