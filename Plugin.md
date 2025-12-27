
---

### 📚 **AIForge Plugin Development Guide**

**Version:** 1.0.0 (Enterprise Ready)
**Framework Core:** AIForge (PyTorch, Pydantic, Loguru, Dependency-Injector)
**Philosophy:** Open for Extension, Closed for Modification (OCP)
**Objective:** Enable seamless integration of new AI models, datasets, and preprocessors.

---

### **I. Understanding the AIForge Plugin Architecture**

AIForge is built with a strong emphasis on modularity. Its "Plugin Architecture" means:

1.  **Standardized Interfaces (Abstract Base Classes)**: The `core/` directory provides abstract base classes (`BaseModel`, `BaseDataset`, `BasePreprocessor`). Your plugins *must* implement these interfaces.
2.  **Dynamic Loading**: The `Container` and `Orchestrator` don't know your specific plugin classes beforehand. They dynamically load them at runtime based on the `model_plugin` and `dataset_plugin` names specified in your `ExperimentConfig`.
3.  **Configuration via Pydantic**: Your plugin's configuration should be defined using Pydantic models, inheriting from `ModelSpecificConfig`. This ensures rigorous type-checking and clear documentation for your plugin's parameters.
4.  **Self-Contained Structure**: Each plugin resides in its own sub-directory within `aiforge/plugins/`, making it easy to manage, version, and share.

This design ensures that:
*   You can add new functionality without touching the `aiforge/core/` modules (OCP).
*   Your components are interchangeable (LSP).
*   Dependencies are managed cleanly (DIP via `dependency-injector`).

---

### **II. Plugin Directory Structure (Your Plugin's Anatomy)**

Every plugin should follow a consistent structure. For a plugin named `my_awesome_plugin`:

```
aiforge/
└── plugins/
    └── my_awesome_plugin/    # Your plugin's root directory
        ├── __init__.py       # REQUIRED: Makes it a Python package
        ├── model.py          # REQUIRED: Your BaseModel implementation
        ├── dataset.py        # REQUIRED: Your BaseDataset implementation
        └── preprocessor.py   # OPTIONAL: Your BasePreprocessor implementation
        └── config.py         # OPTIONAL BUT RECOMMENDED: Your Pydantic config for this plugin
        └── utils.py          # OPTIONAL: Any helper functions specific to your plugin
```

**Key Files Explained:**
*   `__init__.py`: (Empty, or can contain version info) Tells Python this is a package.
*   `model.py`: Contains your concrete `Model` class, implementing `aiforge.core.base_model.BaseModel`.
*   `dataset.py`: Contains your concrete `Dataset` class, implementing `aiforge.core.data_pipeline.BaseDataset`.
*   `preprocessor.py`: (Optional) Contains your concrete `Preprocessor` class, implementing `aiforge.core.data_pipeline.BasePreprocessor`. If your dataset doesn't need a custom preprocessor, you can omit this file.

---

### **III. Step-by-Step Guide to Creating a New Plugin**

Let's create a plugin for a hypothetical **"Transformer Classifier"** model and a **"Text Data"** dataset.

**Plugin Name:** `transformer_text_plugin`

#### **Step 1: Define Plugin Configuration (`aiforge/config/model_configs.py`)**

First, define the Pydantic configuration for your model. Add this to `aiforge/config/model_configs.py`:

```python
# aiforge/config/model_configs.py (add this section)
# ... existing classes (TrainingConfig, ModelSpecificConfig, SimpleNNConfig) ...

class TransformerConfig(ModelSpecificConfig): # Inherit from ModelSpecificConfig
    """Configuration for the Transformer Classifier plugin."""
    vocab_size: int = Field(..., description="Size of the vocabulary")
    embedding_dim: int = Field(256, description="Dimension of token embeddings")
    num_heads: int = Field(8, description="Number of attention heads")
    num_layers: int = Field(3, description="Number of transformer encoder layers")
    ff_dim: int = Field(512, description="Dimension of the feed-forward layer")
    max_seq_len: int = Field(128, description="Maximum sequence length for input")
    output_dim: int = Field(..., description="Output dimension for classification")
    dropout: float = Field(0.1, ge=0.0, le=1.0, description="Dropout rate")
```

#### **Step 2: Create Plugin Directory and `__init__.py`**

Create the folder structure:
`aiforge/plugins/transformer_text_plugin/`
Inside this, create an empty file:
`aiforge/plugins/transformer_text_plugin/__init__.py`

#### **Step 3: Implement Your Model (`aiforge/plugins/transformer_text_plugin/model.py`)**

Your model must inherit from `aiforge.core.base_model.BaseModel` and implement all its abstract methods. Remember to handle your specific `ModelSpecificConfig`.

```python
# aiforge/plugins/transformer_text_plugin/model.py
import torch
import torch.nn as nn
import torch.optim as optim
from aiforge.core.base_model import BaseModel
from aiforge.config.model_configs import TransformerConfig # Import your specific config
from aiforge.core.logger import logger
from typing import Dict, Any

class TransformerEncoder(nn.Module):
    """A single Transformer Encoder Layer."""
    def __init__(self, embedding_dim: int, num_heads: int, ff_dim: int, dropout: float):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(embedding_dim, num_heads, dropout=dropout, batch_first=True)
        self.feed_forward = nn.Sequential(
            nn.Linear(embedding_dim, ff_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embedding_dim)
        )
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout1(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))
        return x

class Model(BaseModel): # Name your class 'Model'
    """
    A Transformer-based text classifier, implementing BaseModel.
    Expects batched sequences of token IDs as input.
    """
    def __init__(self, model_config: TransformerConfig):
        super().__init__(model_config) # Pass model_config to BaseModel constructor
        self.config = model_config # Store your specific config
        
        self.embedding = nn.Embedding(self.config.vocab_size, self.config.embedding_dim)
        self.pos_encoder = nn.Parameter(torch.randn(1, self.config.max_seq_len, self.config.embedding_dim)) # Learnable Positional Encoding

        self.transformer_layers = nn.ModuleList([
            TransformerEncoder(self.config.embedding_dim, self.config.num_heads, self.config.ff_dim, self.config.dropout)
            for _ in range(self.config.num_layers)
        ])
        
        # Classifier head
        self.classifier = nn.Linear(self.config.embedding_dim, self.config.output_dim)
        
        logger.debug(f"TransformerClassifier initialized with vocab_size={self.config.vocab_size}, "
                     f"embedding_dim={self.config.embedding_dim}, num_layers={self.config.num_layers}, "
                     f"output_dim={self.config.output_dim}.")

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Input x: (batch_size, sequence_length) of token IDs.
        Output: (batch_size, output_dim) of logits.
        """
        # Embed tokens and add positional encoding
        x = self.embedding(x) # (batch_size, sequence_length, embedding_dim)
        x = x + self.pos_encoder[:, :x.size(1), :] # Add positional encoding

        for layer in self.transformer_layers:
            x = layer(x, mask)
        
        # For classification, typically take the representation of the first token (like BERT's [CLS])
        # or average/pool over sequence. For simplicity, we'll average.
        pooled_output = torch.mean(x, dim=1) # (batch_size, embedding_dim)
        logits = self.classifier(pooled_output) # (batch_size, output_dim)
        return logits

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        inputs = batch['input_ids'] # Assuming batch contains 'input_ids'
        targets = batch['labels']
        
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        
        return {'loss': loss, 'predictions': logits}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        inputs = batch['input_ids']
        targets = batch['labels']
        
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        
        return {'loss': loss, 'predictions': logits}

    def configure_optimizer(self, optimizer_name: str, learning_rate: float) -> optim.Optimizer:
        optimizer_class = getattr(optim, optimizer_name)
        return optimizer_class(self.parameters(), lr=learning_rate)

    def configure_loss_fn(self, loss_fn_name: str) -> nn.Module:
        loss_fn_class = getattr(nn, loss_fn_name)
        return loss_fn_class()
```

#### **Step 4: Implement Your Dataset (`aiforge/plugins/transformer_text_plugin/dataset.py`)**

Your dataset must inherit from `aiforge.core.data_pipeline.BaseDataset` and implement its abstract methods. It will prepare tokenized text data.

```python
# aiforge/plugins/transformer_text_plugin/dataset.py
import torch
from torch.utils.data import Dataset
from aiforge.core.data_pipeline import BaseDataset, BasePreprocessor
from aiforge.core.logger import logger
from typing import Dict, Any, List, Optional
import os
import pickle

class Dataset(BaseDataset): # Name your class 'Dataset'
    """
    A dummy text dataset for transformer classification.
    Simulates tokenized input_ids and labels.
    """
    def __init__(self, preprocessor: Optional[BasePreprocessor] = None):
        super().__init__()
        self.data: List[Dict[str, torch.Tensor]] = []
        self.preprocessor = preprocessor
        logger.debug("DummyTextDataset initialized.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        if self.preprocessor:
            return self.preprocessor.transform(item)
        return item

    def load_data(self, data_path: str, num_samples: int = 1000, max_seq_len: int = 128, vocab_size: int = 5000, num_classes: int = 2,
                  rank: int = 0, world_size: int = 1, is_distributed: bool = False):
        """
        Generates dummy tokenized text data and persists/loads it from data_path.
        Each sample contains 'input_ids' (token IDs) and 'labels'.
        """
        if rank == 0:
            logger.info(f"Generating {num_samples} dummy text samples (max_seq_len={max_seq_len}, vocab_size={vocab_size}, num_classes={num_classes})...")
            generated_data = []
            torch.manual_seed(43) # Different seed for text data for distinction
            for _ in range(num_samples):
                input_ids = torch.randint(1, vocab_size, (max_seq_len,)) # Token IDs, 0 usually for padding
                labels = torch.randint(0, num_classes, (1,)).squeeze(0)
                generated_data.append({'input_ids': input_ids, 'labels': labels})
            
            with open(data_path, 'wb') as f:
                pickle.dump(generated_data, f)
            logger.info(f"Dummy text data generated and saved to {data_path}")
        
        if is_distributed:
            dist.barrier()
        
        with open(data_path, 'rb') as f:
            self.data = pickle.load(f)
        
        if rank == 0:
            logger.info(f"Dummy text data loaded by rank {rank}.")

```

#### **Step 5: (Optional) Implement Your Preprocessor (`aiforge/plugins/transformer_text_plugin/preprocessor.py`)**

For text data, a common preprocessor might handle tokenization, padding, or numericalization. For this guide, we'll implement a simple preprocessor that ensures `input_ids` are always padded to `max_seq_len`.

```python
# aiforge/plugins/transformer_text_plugin/preprocessor.py
import pickle
from aiforge.core.data_pipeline import BasePreprocessor
from aiforge.core.logger import logger
from typing import Any, List, Dict, Optional
import torch

class Preprocessor(BasePreprocessor): # Name your class 'Preprocessor'
    """
    A dummy text preprocessor that ensures sequences are padded/truncated to max_seq_len.
    """
    def __init__(self):
        self.max_seq_len: Optional[int] = None
        self.padding_token_id: int = 0 # Common padding token ID
        logger.debug("DummyTextPreprocessor initialized.")

    def fit(self, data: Any, max_seq_len: int, padding_token_id: int = 0, **kwargs):
        """
        Fits the preprocessor to the desired max_seq_len.
        'data' argument here is ignored, as we just need the max_seq_len for configuration.
        """
        self.max_seq_len = max_seq_len
        self.padding_token_id = padding_token_id
        logger.info(f"Preprocessor fitted with max_seq_len={max_seq_len}, padding_token_id={padding_token_id}.")

    def transform(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Applies padding/truncation to 'input_ids' in a single data item."""
        if self.max_seq_len is None:
            logger.warning("Preprocessor not fitted. Cannot transform. Returning original data item.")
            return data_item
        
        if 'input_ids' in data_item:
            current_seq_len = data_item['input_ids'].size(0)
            if current_seq_len > self.max_seq_len:
                data_item['input_ids'] = data_item['input_ids'][:self.max_seq_len]
            elif current_seq_len < self.max_seq_len:
                padding = torch.full(
                    (self.max_seq_len - current_seq_len,), 
                    self.padding_token_id, 
                    dtype=data_item['input_ids'].dtype
                )
                data_item['input_ids'] = torch.cat([data_item['input_ids'], padding])
        return data_item

    def inverse_transform(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Inverse transformation (e.g., remove padding). For this demo, it's a no-op."""
        logger.warning("Inverse transform for DummyTextPreprocessor is a no-op for this demo.")
        return data_item

    def save(self, path: str):
        with open(path, 'wb') as f:
            pickle.dump({'max_seq_len': self.max_seq_len, 'padding_token_id': self.padding_token_id}, f)
        logger.info(f"Preprocessor state saved to {path}")

    def load(self, path: str):
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.max_seq_len = state['max_seq_len']
            self.padding_token_id = state['padding_token_id']
        logger.info(f"Preprocessor state loaded from {path}")
```

#### **Step 6: Update `main.py` to Use Your New Plugin**

Modify `aiforge/main.py` to run an experiment with your `transformer_text_plugin`.

```python
# aiforge/main.py (Updated to run transformer_text_plugin)
import torch
import torch.distributed as dist
import os
import mlflow

from aiforge.core.orchestrator import Orchestrator
from aiforge.config.model_configs import ExperimentConfig, SimpleNNConfig, TrainingConfig, TransformerConfig # NEW: Import TransformerConfig
from aiforge.config.settings import settings
from aiforge.core.logger import logger
from aiforge.container import Container

def setup_distributed(rank, world_size, backend='nccl'):
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    logger.info(f"Distributed process group initialized for rank {rank} / {world_size}.")
    torch.cuda.set_device(rank)

def cleanup_distributed():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")

def run_experiment_main(config: ExperimentConfig, mlflow_run_id: Optional[str] = None):
    container = Container()
    container.experiment_config.override(config)
    container.mlflow_run_id.override(mlflow_run_id)

    orchestrator: Orchestrator = container.orchestrator()
    orchestrator.run_experiment()

def main():
    logger.level(settings.LOG_LEVEL)

    # --- Define experiment configurations ---
    # NEW: Experiment with Transformer Classifier
    vocab_size = 5000
    max_seq_len = 128
    num_classes = 2 # Binary classification for text

    text_data_file = settings.DATA_DIR / "dummy_text_data.pkl" 

    model_config_instance = TransformerConfig( # Use your new config
        vocab_size=vocab_size,
        embedding_dim=256,
        num_heads=4,
        num_layers=2,
        ff_dim=512,
        max_seq_len=max_seq_len,
        output_dim=num_classes,
        dropout=0.1
    )

    training_config_instance = TrainingConfig(
        epochs=3, # Fewer epochs for quick demo
        batch_size=8,
        learning_rate=0.0001,
        optimizer_name="Adam",
        loss_fn_name="CrossEntropyLoss",
        early_stopping_patience=1,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        
        use_distributed=False, 
        local_rank=0,
        world_size=1
    )

    experiment_config_instance = ExperimentConfig(
        experiment_name="transformer_text_classification_experiment", # New experiment name
        model_plugin="transformer_text_plugin",    # Your new plugin name
        dataset_plugin="transformer_text_plugin",  # Your new plugin name
        data_path=str(text_data_file),
        # Preprocessor args: max_seq_len is needed for fitting.
        preprocess_args={"max_seq_len": max_seq_len, "padding_token_id": 0}, 
        model_config=model_config_instance,
        training_config=training_config_instance
    )

    # 2. --- Handle Distributed Training Setup ---
    is_distributed_launch = int(os.environ.get('RANK', -1)) != -1
    if is_distributed_launch:
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        experiment_config_instance.training_config.use_distributed = True
        experiment_config_instance.training_config.local_rank = local_rank
        experiment_config_instance.training_config.world_size = world_size
        
        setup_distributed(global_rank, world_size)
        logger.info(f"Main: DDP launcher detected. Running as global_rank {global_rank} / {world_size}.")
    else:
        logger.info("Main: Running in single-process mode.")

    # 3. --- MLflow setup (only for rank 0) ---
    mlflow_run_id: Optional[str] = None
    current_rank = int(os.environ.get('RANK', 0))
    if current_rank == 0:
        mlflow.set_experiment(experiment_config_instance.experiment_name)
        with mlflow.start_run() as run:
            mlflow_run_id = run.info.run_id
            logger.info(f"Main: MLflow run initiated with ID: {mlflow_run_id}")
            mlflow.log_params(experiment_config_instance.model_dump())
    
    if is_distributed_launch:
        if current_rank == 0:
            run_id_int = int(mlflow_run_id, 16) if mlflow_run_id else 0
            run_id_tensor = torch.tensor([run_id_int]).to(f"cuda:{local_rank}")
        else:
            run_id_tensor = torch.tensor([0]).to(f"cuda:{local_rank}")
        
        dist.barrier()
        dist.broadcast(run_id_tensor, src=0)
        if current_rank != 0:
            mlflow_run_id = hex(run_id_tensor.item())[2:] if run_id_tensor.item() != 0 else None
            if mlflow_run_id:
                mlflow_run_id = f"{mlflow_run_id:0>32}"
            logger.info(f"Main: Rank {current_rank} received MLflow run ID: {mlflow_run_id}")

    # 4. --- Run the experiment ---
    try:
        run_experiment_main(experiment_config_instance, mlflow_run_id)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
    finally:
        cleanup_distributed()

if __name__ == "__main__":
    main()
```

---

### **IV. Testing Your New Plugin**

You can verify your plugin's integration using the existing integration tests in `aiforge/tests/test_integration.py`. The `Container`'s `wiring_config` already includes `aiforge.tests.test_integration`, so the DI should correctly load your new classes.

**To test the new plugin:**

1.  **Modify `test_integration.py`**: Temporarily update `sample_experiment_config` fixture to point to your new `transformer_text_plugin`.

    ```python
    # aiforge/tests/test_integration.py (modify sample_experiment_config fixture)
    # ... other imports and fixtures ...
    from aiforge.plugins.transformer_text_plugin.model import Model as TransformerTextModel # NEW
    from aiforge.plugins.transformer_text_plugin.dataset import Dataset as TransformerTextDataset # NEW
    from aiforge.plugins.transformer_text_plugin.preprocessor import Preprocessor as TransformerTextPreprocessor # NEW

    @pytest.fixture
    def sample_experiment_config():
        """Provides a sample ExperimentConfig for testing."""
        # OLD: test_data_file = settings.TEST_DATA_DIR / "test_dummy_data.pkl"
        test_data_file = settings.TEST_DATA_DIR / "test_dummy_text_data.pkl" # NEW: For text data

        # OLD: return ExperimentConfig( ... model_config=SimpleNNConfig(...) ... )
        return ExperimentConfig(
            experiment_name="test_transformer_text_exp",
            model_plugin="transformer_text_plugin", # NEW
            dataset_plugin="transformer_text_plugin", # NEW
            data_path=str(test_data_file),
            preprocess_args={"max_seq_len": 128, "padding_token_id": 0}, # NEW: Preprocessor args
            model_config=TransformerConfig(input_dim=10, output_dim=2, vocab_size=5000), # NEW: Use TransformerConfig
            training_config=TrainingConfig(
                epochs=1, # Even fewer for faster integration tests
                batch_size=4,
                learning_rate=0.0001,
                optimizer_name="Adam",
                loss_fn_name="CrossEntropyLoss",
                early_stopping_patience=1,
                early_stopping_monitor="val_loss",
                early_stopping_mode="min"
            )
        )

    # Update relevant assertions in test_container_wiring_and_orchestrator_init
    # and test_orchestrator_training_execution if necessary to reflect TransformerTextModel
    # For instance:
    # assert issubclass(model_class, TransformerTextModel)
    # assert issubclass(dataset_class, TransformerTextDataset)
    # assert issubclass(preprocessor_class, TransformerTextPreprocessor)
    # assert isinstance(orchestrator.model, TransformerTextModel)
    ```

2.  **Run Pytest**: `pytest aiforge/tests/test_integration.py`
    -   This will execute the integration tests, loading your new `transformer_text_plugin`. If everything is implemented correctly, the tests should pass, confirming that your plugin is correctly wired and operable within the AIForge ecosystem.

---

This guide, along with the provided code, enables any new team to develop and integrate their specialized AI models and data pipelines as plugins, ensuring maintainability, scalability, and adherence to robust engineering principles within the AIForge framework.
