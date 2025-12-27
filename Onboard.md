
---

### 🚀 **AIForge Developer Onboarding Cheatsheet**

**Framework:** AIForge v1.1.0 (Production Ready)
**Core Tools:** Python 3.9+, PyTorch, Pydantic, Loguru, `dependency-injector`, `pytest`, `mlflow`
**Principles:** SOLID, OCP, DIP, Test-Driven Development

---

#### **I. Quickstart: Get AIForge Running**

1.  **Clone / Setup Project:**
    ```bash
    git clone your_aiforge_repo
    cd your_aiforge_repo
    ```
2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Configure Environment (Optional):**
    *   Create `.env` file in project root (`aiforge/` folder) to override default settings (e.g., `AIFORGE_LOG_LEVEL=DEBUG`, `AIFORGE_DEVICE=cpu`).
4.  **Run a Sample Experiment (Single GPU/CPU):**
    ```bash
    python aiforge/main.py
    ```
    *   *Expected:* Logs to console and `aiforge_debug.log`. `mlruns/` directory created with experiment logs.
5.  **View MLflow Dashboard:**
    ```bash
    mlflow ui
    ```
    *   Open `http://localhost:5000` (or as indicated by `mlflow ui`) in your browser.

#### **II. Plugin Development: Your Contribution Workflow**

1.  **Generate New Plugin from Template:**
    ```bash
    cp -R aiforge/plugin_templates/template_plugin aiforge/plugins/my_new_plugin
    # Rename placeholders in new files (e.g., ###_MODEL_NAME_###)
    ```
2.  **Define Your Plugin's Config (`aiforge/config/model_configs.py`):**
    ```python
    # Example: aiforge/config/model_configs.py
    class MyNewModelConfig(ModelSpecificConfig):
        # Your Pydantic fields here, e.g.:
        my_param_int: int = Field(10, description="An integer parameter")
        my_param_str: str = Field("default", description="A string parameter")
    ```
3.  **Implement `model.py` (`aiforge/plugins/my_new_plugin/model.py`):**
    *   Inherit `aiforge.core.base_model.BaseModel`.
    *   Implement `__init__` (accepts `model_config: MyNewModelConfig`).
    *   Implement `forward`, `training_step`, `validation_step`, `configure_optimizer`, `configure_loss_fn`.
    *   Use `self.model_config` to access your Pydantic parameters.
4.  **Implement `dataset.py` (`aiforge/plugins/my_new_plugin/dataset.py`):**
    *   Inherit `aiforge.core.data_pipeline.BaseDataset`.
    *   Implement `__len__`, `__getitem__`, `load_data`.
    *   If `load_data` generates dummy data for tests, ensure it persists to `data_path` via `pickle`.
5.  **Implement `preprocessor.py` (Optional, `aiforge/plugins/my_new_plugin/preprocessor.py`):**
    *   Inherit `aiforge.core.data_pipeline.BasePreprocessor`.
    *   Implement `fit`, `transform`, `inverse_transform` (if applicable).
    *   Override `save`/`load` if custom serialization is needed (default pickle is fine for most).
6.  **Create a Test Case (`aiforge/tests/test_my_new_plugin.py`):**
    *   Write `pytest` integration tests to ensure your plugin works with `Container` and `Orchestrator`.
    *   Use `mock_mlflow` and `mock_distributed` fixtures to keep tests fast and isolated.

#### **III. Running Tests & Quality Assurance**

1.  **Run All Integration Tests:**
    ```bash
    pytest aiforge/tests/test_integration.py
    # or `pytest` from project root to run all tests in `aiforge/tests/`
    ```
2.  **Run with Multiple GPUs (if available & DDP configured):**
    ```bash
    torchrun --standalone --nproc_per_node=N aiforge/main.py
    # N = number of GPUs
    ```
3.  **Check CI Pipeline Status (e.g., GitHub Actions):**
    *   Ensure your Pull Requests automatically trigger the `aiforge_ci.yml` workflow.
    *   Verify all checks pass before merging.

#### **IV. Key Principles to Remember**

*   **SOLID Always:** Your code should be clean, modular, and adhere to Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion principles.
*   **Interface Over Implementation:** Always depend on `BaseModel`, `BaseDataset`, `BasePreprocessor` interfaces.
*   **Pydantic for Config:** Use Pydantic models for all plugin configurations. This provides type-safety, validation, and clear documentation.
*   **Loguru for Insights:** Use `logger.info()`, `logger.debug()`, `logger.error()` consistently for framework insights and debugging.
*   **Shared Utilities:** Before duplicating logic, check `aiforge/utils/` for reusable components. Propose new ones if applicable.
*   **Reproducibility:** Fix random seeds for data generation, model initialization, and DataLoader splits where appropriate, especially for tests.
*   **Distributed Awareness:** Design your `load_data` (and any other IO-heavy ops) to be aware of `rank` and `world_size` if DDP is used, ensuring data consistency across processes.

---

This cheatsheet should provide your development teams with a clear and concise roadmap for building robust, integrated, and high-quality AI plugins within the AIForge ecosystem.
