



---

### 🧠 **AIForge: Modular & Object-Oriented AI Framework**

**Core Philosophy:** AIForge is designed for extensibility, maintainability, and testability. By separating concerns (SOLID), relying on clear interfaces, and centralizing configuration, it enables rapid prototyping and robust deployment of diverse AI models.

---

### **I. Directory Structure**

```
aiforge/
├── __init__.py               # Package marker
├── core/
│   ├── __init__.py           # Core package marker
│   ├── logger.py             # Loguru initialization
│   ├── base_model.py         # Abstract Base Class for all models
│   ├── data_pipeline.py      # Abstract Base Classes for data handling
│   └── orchestrator.py       # Central orchestrator for experiment lifecycle
├── config/
│   ├── __init__.py           # Config package marker
│   ├── settings.py           # Global application settings (Pydantic BaseSettings)
│   └── model_configs.py      # Pydantic models for experiment & model configurations
├── plugins/
│   ├── __init__.py           # Plugin discovery marker
│   └── simple_nn/            # Example plugin for a Simple Neural Network
│       ├── __init__.py
│       ├── model.py          # Concrete SimpleNN implementation (inherits BaseModel)
│       ├── dataset.py        # Concrete DummyDataset (inherits BaseDataset)
│       └── preprocessor.py   # Concrete DummyPreprocessor (inherits BasePreprocessor)
├── utils/
│   ├── __init__.py           # Utilities package marker
│   ├── metrics.py            # Common evaluation metrics (e.g., accuracy)
│   └── callbacks.py          # Training callbacks (e.g., EarlyStopping)
├── main.py                   # Entry point for running an experiment
└── requirements.txt          # Project dependencies
```

---

### **II. Core Components & Implementation**

#### **1. Logging (`aiforge/core/logger.py`)**
Initializes Loguru for comprehensive, configurable logging.

```python
# aiforge/core/logger.py
import sys
from loguru import logger

# Remove default handler to configure custom ones
logger.remove() 

# Add a console handler with INFO level
logger.add(sys.stderr, level="INFO") 

# Add a file handler for detailed DEBUG logs, rotates at 10 MB
logger.add(
    "aiforge_debug.log", 
    level="DEBUG", 
    rotation="10 MB", 
    compression="zip", 
    enqueue=True # Use a queue to prevent blocking main thread
)

# You can add context-aware logging, e.g., to filter by module:
# logger.add(
#     "aiforge_model_logs.log", 
#     filter=lambda record: "aiforge.plugins.simple_nn.model" in record["name"], 
#     level="DEBUG"
# )
```

#### **2. Abstract Model (`aiforge/core/base_model.py`)**
Defines the `BaseModel` abstract class, ensuring all concrete models adhere to a standard interface. Adheres to **Open/Closed Principle (OCP)** for extensibility and **Liskov Substitution Principle (LSP)** for interchangeability.

```python
# aiforge/core/base_model.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Type

class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all AI models.
    Adheres to:
    - OCP: Models extend BaseModel without modifying Orchestrator.
    - LSP: Concrete models are substitutable for BaseModel.
    """

    def __init__(self):
        super().__init__()
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        pass

    @abstractmethod
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Defines a single training step.
        Should return a dictionary containing 'loss' and any relevant metrics or predictions.
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Defines a single validation step.
        Should return a dictionary containing 'loss' and any relevant metrics or predictions.
        """
        pass

    @abstractmethod
    def configure_optimizer(self, optimizer_name: str, learning_rate: float) -> torch.optim.Optimizer:
        """
        Configures and returns the optimizer for the model's parameters.
        The model determines *how* its parameters are optimized, given the *choice* (optimizer_name)
        and hyperparameters (learning_rate) from the Orchestrator.
        """
        pass
    
    @abstractmethod
    def configure_loss_fn(self, loss_fn_name: str) -> nn.Module:
        """
        Configures and returns the loss function for the model.
        The model defines *what* loss function is instantiated, given the *choice* (loss_fn_name)
        from the Orchestrator.
        """
        pass

    def save_checkpoint(self, path: str):
        """Saves the model's state dictionary."""
        torch.save(self.state_dict(), path)
        logger.info(f"Model checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Loads the model's state dictionary."""
        self.load_state_dict(torch.load(path, map_location=self.device)) # Ensure loading to correct device
        logger.info(f"Model checkpoint loaded from {path}")
```

#### **3. Data Pipeline (`aiforge/core/data_pipeline.py`)**
Abstracts data loading and preprocessing, promoting modularity and reusability. Adheres to **OCP** and **LSP**.

```python
# aiforge/core/data_pipeline.py
from abc import ABC, abstractmethod
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Any, Dict, Optional, Tuple
import pickle

class BaseDataset(Dataset, ABC):
    """
    Abstract Base Class for datasets.
    Concrete datasets should inherit from this and implement data loading logic.
    """
    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        pass

    @abstractmethod
    def load_data(self, data_path: str, **kwargs):
        """
        Abstract method to load data into the dataset.
        Specific implementation will vary per dataset.
        """
        pass

class BasePreprocessor(ABC):
    """
    Abstract Base Class for data preprocessors.
    Concrete preprocessors should implement fit/transform logic.
    """
    @abstractmethod
    def fit(self, data: Any, **kwargs):
        """Fits the preprocessor to data (e.g., calculates means/stds for scaling)."""
        pass

    @abstractmethod
    def transform(self, data: Any) -> Any:
        """Applies transformation to a single data item or batch."""
        pass
    
    @abstractmethod
    def inverse_transform(self, data: Any) -> Any:
        """Applies inverse transformation to data (if applicable)."""
        pass

    def save(self, path: str):
        """Saves the preprocessor's internal state."""
        # Generic save (e.g., pickle), subclasses might override for specific formats
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
        logger.info(f"Preprocessor state saved to {path}")

    def load(self, path: str):
        """Loads the preprocessor's internal state."""
        with open(path, 'rb') as f:
            self.__dict__.update(pickle.load(f))
        logger.info(f"Preprocessor state loaded from {path}")
```

#### **4. Configuration (`aiforge/config/settings.py`, `aiforge/config/model_configs.py`)**
Uses Pydantic for robust, type-checked configuration management. Adheres to **Single Responsibility Principle (SRP)** by isolating configuration logic.

```python
# aiforge/config/settings.py
from pydantic import BaseSettings, Field
from pathlib import Path
import os
import torch
from aiforge.core.logger import logger

class AppSettings(BaseSettings):
    """Global application settings."""
    PROJECT_NAME: str = "AIForge"
    PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent # Points to 'aiforge/'
    LOG_LEVEL: str = Field("INFO", env="AIFORGE_LOG_LEVEL")
    DATA_DIR: Path = Field(PROJECT_ROOT / "data", env="AIFORGE_DATA_DIR")
    MODELS_DIR: Path = Field(PROJECT_ROOT / "models", env="AIFORGE_MODELS_DIR")
    RESULTS_DIR: Path = Field(PROJECT_ROOT / "results", env="AIFORGE_RESULTS_DIR")
    PLUGINS_DIR: Path = Field(PROJECT_ROOT / "plugins", env="AIFORGE_PLUGINS_DIR")
    
    DEVICE: str = Field("cuda" if torch.cuda.is_available() else "cpu", env="AIFORGE_DEVICE")

    class Config:
        env_file = ".env"
        env_file_encoding = 'utf-8'
        # Allows .env file to override these settings. E.g., AIFORGE_LOG_LEVEL=DEBUG

# Instantiate settings globally
settings = AppSettings()

# Ensure necessary directories exist
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
settings.PLUGINS_DIR.mkdir(parents=True, exist_ok=True)

logger.debug(f"Application settings loaded: {settings.json()}")

```

```python
# aiforge/config/model_configs.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional

class TrainingConfig(BaseModel):
    """Configuration for the training loop."""
    epochs: int = Field(10, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size for training and validation")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer")
    optimizer_name: Literal["Adam", "SGD", "RMSprop"] = Field("Adam", description="Name of the PyTorch optimizer")
    loss_fn_name: Literal["CrossEntropyLoss", "MSELoss", "BCELoss", "L1Loss"] = Field("CrossEntropyLoss", description="Name of the PyTorch loss function")
    gradient_clip_norm: Optional[float] = Field(None, description="Clip gradients to this max norm, if provided")
    early_stopping_patience: Optional[int] = Field(None, description="Number of epochs to wait for improvement before stopping")
    early_stopping_monitor: str = Field("val_loss", description="Metric to monitor for early stopping")
    early_stopping_mode: Literal["min", "max"] = Field("min", description="Mode for early stopping (min for loss, max for accuracy)")

class ModelSpecificConfig(BaseModel):
    """Base class for model-specific configurations. Plugins will extend this."""
    pass

class SimpleNNConfig(ModelSpecificConfig):
    """Configuration for the Simple Neural Network plugin."""
    input_dim: int = Field(..., description="Input dimension of the model")
    hidden_dim: int = Field(128, description="Dimension of the hidden layer")
    output_dim: int = Field(..., description="Output dimension of the model")
    dropout_rate: float = Field(0.2, ge=0.0, le=1.0, description="Dropout rate for regularization")
    activation_fn: Literal["ReLU", "Sigmoid", "Tanh", "GELU"] = Field("ReLU", description="Activation function to use in hidden layers")

class ExperimentConfig(BaseModel):
    """Overall experiment configuration."""
    experiment_name: str = Field("default_experiment", description="Unique name for the experiment results folder")
    model_plugin: str = Field(..., description="Name of the model plugin to use (e.g., 'simple_nn')")
    dataset_plugin: str = Field(..., description="Name of the dataset plugin to use (e.g., 'simple_nn')")
    
    data_path: str = Field(..., description="Path to the raw data file(s) for the dataset")
    preprocess_args: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the preprocessor's fit method")

    model_config: ModelSpecificConfig = Field(..., description="Model-specific configuration instance")
    training_config: TrainingConfig = Field(default_factory=TrainingConfig, description="Training loop configuration")

    class Config:
        # Pydantic v1 required `arbitrary_types_allowed = True` for polymorphic fields (ModelSpecificConfig).
        # Pydantic v2 handles this better.
        pass

# Example of how to combine configs for a specific experiment:
# simple_nn_exp_config = ExperimentConfig(
#     experiment_name="my_first_simple_nn",
#     model_plugin="simple_nn",
#     dataset_plugin="simple_nn",
#     data_path="/path/to/my/data.csv",
#     model_config=SimpleNNConfig(input_dim=10, output_dim=2),
#     training_config=TrainingConfig(epochs=50, learning_rate=0.01)
# )
```

#### **5. Orchestrator (`aiforge/core/orchestrator.py`)**
The central orchestrator manages the lifecycle of an experiment. Adheres to **SRP** (orchestration only) and **Dependency Inversion Principle (DIP)** (depends on abstractions, not concretions).

```python
# aiforge/core/orchestrator.py
import torch
from torch.utils.data import DataLoader
from aiforge.core.logger import logger
from aiforge.core.base_model import BaseModel
from aiforge.core.data_pipeline import BaseDataset, BasePreprocessor
from aiforge.config.model_configs import ExperimentConfig, TrainingConfig
from aiforge.config.settings import settings
from aiforge.utils.callbacks import EarlyStopping, Callback # Import EarlyStopping callback
from typing import Type, Dict, Any, Optional, List
import json
import importlib

class Orchestrator:
    """
    Manages the end-to-end AI experiment lifecycle:
    configuration, data loading, preprocessing, model training, and evaluation.
    Adheres to SOLID principles by depending on abstractions.
    """
    def __init__(self, experiment_config: ExperimentConfig):
        self.config = experiment_config
        self.device = torch.device(settings.DEVICE)
        
        self.model: Optional[BaseModel] = None
        self.dataset: Optional[BaseDataset] = None
        self.preprocessor: Optional[BasePreprocessor] = None
        
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        
        self._load_plugin_components()
        self._initialize_model()
        
        # Setup result directory
        self.experiment_results_dir = settings.RESULTS_DIR / self.config.experiment_name
        self.experiment_results_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Experiment results will be stored in: {self.experiment_results_dir}")

        self.callbacks: List[Callback] = []
        if self.config.training_config.early_stopping_patience:
            self.callbacks.append(
                EarlyStopping(
                    patience=self.config.training_config.early_stopping_patience,
                    monitor=self.config.training_config.early_stopping_monitor,
                    mode=self.config.training_config.early_stopping_mode
                )
            )

    def _load_plugin_components(self):
        """
        Dynamically loads model, dataset, and preprocessor classes from plugins.
        Adheres to DIP by loading classes based on configuration.
        """
        try:
            # Plugin discovery and import using importlib
            model_module_path = f"aiforge.plugins.{self.config.model_plugin}.model"
            dataset_module_path = f"aiforge.plugins.{self.config.dataset_plugin}.dataset"
            
            plugin_model_module = importlib.import_module(model_module_path)
            plugin_dataset_module = importlib.import_module(dataset_module_path)
            
            # Check for preprocessor in dataset plugin
            preprocessor_module = None
            try:
                preprocessor_module_path = f"aiforge.plugins.{self.config.dataset_plugin}.preprocessor"
                preprocessor_module = importlib.import_module(preprocessor_module_path)
            except ImportError:
                logger.warning(f"No preprocessor found for dataset plugin '{self.config.dataset_plugin}'. Skipping.")
            
            self.ModelClass: Type[BaseModel] = getattr(plugin_model_module, "Model")
            self.DatasetClass: Type[BaseDataset] = getattr(plugin_dataset_module, "Dataset")
            self.PreprocessorClass: Optional[Type[BasePreprocessor]] = getattr(preprocessor_module, "Preprocessor") if preprocessor_module else None

            logger.info(f"Loaded Model: {self.ModelClass.__name__} from '{self.config.model_plugin}' plugin.")
            logger.info(f"Loaded Dataset: {self.DatasetClass.__name__} from '{self.config.dataset_plugin}' plugin.")
            if self.PreprocessorClass:
                logger.info(f"Loaded Preprocessor: {self.PreprocessorClass.__name__} from '{self.config.dataset_plugin}' plugin.")

        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load plugin components. Ensure plugin names and class names ('Model', 'Dataset', 'Preprocessor') are correct: {e}")
            raise

    def _initialize_model(self):
        """
        Initializes the model, its optimizer, and loss function.
        Model-specific configuration is passed to the model's constructor.
        """
        if self.ModelClass is None:
            raise RuntimeError("ModelClass not loaded.")

        self.model = self.ModelClass(model_config=self.config.model_config).to(self.device)
        
        self.model.optimizer = self.model.configure_optimizer(
            self.config.training_config.optimizer_name,
            self.config.training_config.learning_rate
        )
        self.model.loss_fn = self.model.configure_loss_fn(
            self.config.training_config.loss_fn_name
        )
        logger.info(f"Model initialized: {self.model.__class__.__name__} on {self.device}")
        logger.debug(f"Model Architecture:\n{self.model}")

    def prepare_data(self):
        """
        Loads the dataset and optionally applies preprocessing.
        Assumes dataset's load_data method can handle arguments for preprocessing or
        that preprocessing is applied within the dataset itself for simplicity of this demo.
        """
        if self.DatasetClass is None:
            raise RuntimeError("DatasetClass not loaded.")

        logger.info(f"Loading data from {self.config.data_path} using {self.DatasetClass.__name__}...")
        self.dataset = self.DatasetClass()
        self.dataset.load_data(self.config.data_path) # Dataset specific loading logic

        if self.PreprocessorClass:
            self.preprocessor = self.PreprocessorClass()
            logger.info(f"Fitting preprocessor {self.preprocessor.__class__.__name__}...")
            # For this demo, assuming preprocessor fits on entire dataset's features.
            # In a real system, you might fit on a training split only.
            # We pass a sample of data, but the `transform` needs to be applied to each item
            # before DataLoader creation, or the DataLoader collate_fn needs to incorporate it.
            
            # To apply preprocessor: get all features, fit, then transform dataset items.
            all_features = [self.dataset[i]['features'] for i in range(len(self.dataset))]
            self.preprocessor.fit(all_features, **self.config.preprocess_args)
            logger.info(f"Preprocessor fitted and saved to {self.experiment_results_dir / 'preprocessor.pkl'}")
            self.preprocessor.save(self.experiment_results_dir / 'preprocessor.pkl')

            # Now apply transform to each item in the dataset
            # This is a key part that simplifies the dataset's role for this demo.
            # In production, BaseDataset might accept a preprocessor directly.
            self.dataset.preprocessor = self.preprocessor 
            logger.info("Preprocessor state attached to dataset for on-the-fly transformation.")
        
        # Split dataset for training and validation
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        # Use a fixed generator for reproducibility of the split
        generator = torch.Generator().manual_seed(42) 
        train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size], generator=generator)

        self.train_dataloader = DataLoader(
            train_dataset, batch_size=self.config.training_config.batch_size, shuffle=True
        )
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=self.config.training_config.batch_size, shuffle=False
        )
        logger.info(f"Data prepared: {len(train_dataset)} training samples, "
                    f"{len(val_dataset)} validation samples.")

    def train(self):
        """Executes the training loop."""
        if not self.model or not self.train_dataloader or not self.val_dataloader:
            raise RuntimeError("Model or data not prepared for training.")

        logger.info(f"Starting training for {self.config.training_config.epochs} epochs...")
        
        best_val_loss = float('inf')
        
        for epoch in range(self.config.training_config.epochs):
            self.model.train()
            total_train_loss = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.model.optimizer.zero_grad()
                
                # Move batch to device
                inputs = batch['features'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                step_outputs = self.model.training_step({'features': inputs, 'labels': targets}, batch_idx)
                loss = step_outputs['loss']
                loss.backward()

                if self.config.training_config.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training_config.gradient_clip_norm)

                self.model.optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            
            # Validation step
            avg_val_loss, val_metrics = self.evaluate(self.val_dataloader)
            
            logs = {
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                **val_metrics
            }
            logger.info(f"Epoch {epoch+1}/{self.config.training_config.epochs} | "
                        f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                        f"Val Metrics: {val_metrics}")

            # Callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch, logs)
                if isinstance(callback, EarlyStopping) and callback.stop_training:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                    self.model.load_checkpoint(self.experiment_results_dir / "best_model.pth")
                    return # Exit training loop

            # Update best model based on monitor metric
            monitor_value = logs.get(self.config.training_config.early_stopping_monitor)
            if monitor_value is not None:
                if (self.config.training_config.early_stopping_mode == 'min' and monitor_value < best_val_loss) or \
                   (self.config.training_config.early_stopping_mode == 'max' and monitor_value > best_val_loss):
                    best_val_loss = monitor_value
                    self.model.save_checkpoint(self.experiment_results_dir / "best_model.pth")
                    logger.debug(f"Saved best model to {self.experiment_results_dir / 'best_model.pth'}")

        logger.info("Training complete.")
        # If no early stopping, ensure the best model is loaded if it was ever saved.
        if (self.experiment_results_dir / "best_model.pth").exists():
             self.model.load_checkpoint(self.experiment_results_dir / "best_model.pth")


    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """Evaluates the model on a given dataloader."""
        if not self.model:
            raise RuntimeError("Model not initialized for evaluation.")

        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs = batch['features'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                step_outputs = self.model.validation_step({'features': inputs, 'labels': targets}, batch_idx)
                loss = step_outputs['loss']
                total_loss += loss.item()
                
                if 'predictions' in step_outputs:
                    all_preds.append(step_outputs['predictions'].cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / len(dataloader)
        
        metrics = {}
        if all_preds and all_targets:
            # Simple accuracy for classification
            if self.config.training_config.loss_fn_name == "CrossEntropyLoss":
                preds = torch.cat(all_preds)
                targets = torch.cat(all_targets)
                correct = (preds.argmax(dim=1) == targets).sum().item()
                accuracy = correct / len(targets)
                metrics['accuracy'] = accuracy
            # Add other metrics as needed based on the task

        return avg_loss, metrics

    def run_experiment(self):
        """Runs the entire experiment: data prep, train, final eval."""
        logger.info(f"--- Starting Experiment: {self.config.experiment_name} ---")
        
        # Save experiment config
        config_path = self.experiment_results_dir / "experiment_config.json"
        with open(config_path, "w") as f:
            f.write(self.config.json(indent=2))
        logger.info(f"Experiment configuration saved to {config_path}")

        self.prepare_data()
        self.train()
        
        logger.info("Running final evaluation on validation set...")
        final_val_loss, final_val_metrics = self.evaluate(self.val_dataloader)
        logger.info(f"Final Validation Loss: {final_val_loss:.4f} | Final Metrics: {final_val_metrics}")
        
        # Save final model
        final_model_path = self.experiment_results_dir / "final_model.pth"
        self.model.save_checkpoint(final_model_path)
        
        final_results = {
            "final_val_loss": final_val_loss,
            "final_val_metrics": final_val_metrics,
            "model_path": str(final_model_path),
            "config_path": str(config_path)
        }
        with open(self.experiment_results_dir / "final_results.json", "w") as f:
            json.dump(final_results, f, indent=2)

        logger.info(f"Experiment '{self.config.experiment_name}' finished.")
        logger.info(f"Results available in {self.experiment_results_dir}")

```

#### **6. Utilities (`aiforge/utils/metrics.py`, `aiforge/utils/callbacks.py`)**
Placeholder for common metrics and training callbacks.

```python
# aiforge/utils/metrics.py
import torch
from typing import Dict, Any
from aiforge.core.logger import logger

def accuracy(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    """Calculates classification accuracy."""
    if predictions.dim() > 1 and predictions.shape[1] > 1: # Assuming logits for multi-class
        predictions = predictions.argmax(dim=1)
    correct = (predictions == targets).sum().item()
    return correct / len(targets)

def f1_score(predictions: torch.Tensor, targets: torch.Tensor, average: str = 'macro') -> float:
    """
    Placeholder for F1-score. 
    In a real system, you'd use sklearn.metrics or a robust PyTorch implementation.
    """
    logger.warning("F1-score is a placeholder. Use sklearn.metrics for production.")
    # Example using actual data if needed, but keeping it simple for the demo
    # from sklearn.metrics import f1_score as sk_f1_score
    # if predictions.dim() > 1 and predictions.shape[1] > 1:
    #     predictions = predictions.argmax(dim=1)
    # return sk_f1_score(targets.cpu().numpy(), predictions.cpu().numpy(), average=average)
    return 0.85 # Dummy value
```

```python
# aiforge/utils/callbacks.py
from aiforge.core.logger import logger
from typing import Dict, Any, Optional

class Callback:
    """Base class for training callbacks."""
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Called at the end of each epoch."""
        pass

class EarlyStopping(Callback):
    """
    Callback for early stopping during training.
    Stops training if the monitored metric does not improve for a given patience.
    """
    def __init__(self, patience: int, min_delta: float = 0.001, monitor: str = 'val_loss', mode: str = 'min'):
        if mode not in ['min', 'max']:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'.")
        self.patience = patience
        self.min_delta = abs(min_delta) # delta should always be positive
        self.monitor = monitor
        self.mode = mode
        self.best_value: float = float('inf') if mode == 'min' else float('-inf')
        self.epochs_no_improve: int = 0
        self.stop_training: bool = False
        logger.info(f"EarlyStopping initialized: monitor='{monitor}', patience={patience}, min_delta={min_delta}, mode='{mode}'")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        current_value = logs.get(self.monitor)
        if current_value is None:
            logger.warning(f"EarlyStopping monitor '{self.monitor}' not found in logs for epoch {epoch}. Skipping check.")
            return

        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.epochs_no_improve = 0
                logger.debug(f"Epoch {epoch}: {self.monitor} improved to {current_value:.4f}.")
            else:
                self.epochs_no_improve += 1
                logger.debug(f"Epoch {epoch}: {self.monitor} did not improve ({current_value:.4f} vs best {self.best_value:.4f}). "
                             f"No improvement for {self.epochs_no_improve} epochs.")
        else: # mode == 'max'
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.epochs_no_improve = 0
                logger.debug(f"Epoch {epoch}: {self.monitor} improved to {current_value:.4f}.")
            else:
                self.epochs_no_improve += 1
                logger.debug(f"Epoch {epoch}: {self.monitor} did not improve ({current_value:.4f} vs best {self.best_value:.4f}). "
                             f"No improvement for {self.epochs_no_improve} epochs.")
        
        if self.epochs_no_improve >= self.patience:
            self.stop_training = True
            logger.info(f"Early stopping triggered for '{self.monitor}' after {epoch+1} epochs.")
```

---

### **III. Plugin Architecture: Simple Neural Network Example (`aiforge/plugins/simple_nn/`)**

This demonstrates how new algorithms are integrated into AIForge without modifying its core.

#### **1. `aiforge/plugins/simple_nn/__init__.py`**
(Empty file to mark `simple_nn` as a Python package.)

#### **2. `aiforge/plugins/simple_nn/model.py`**
A concrete implementation of `BaseModel` for a feedforward neural network.

```python
# aiforge/plugins/simple_nn/model.py
import torch
import torch.nn as nn
import torch.optim as optim
from aiforge.core.base_model import BaseModel
from aiforge.config.model_configs import SimpleNNConfig
from aiforge.core.logger import logger
from typing import Dict, Any, Type

class Model(BaseModel):
    """
    A simple Feedforward Neural Network, implementing BaseModel.
    """
    def __init__(self, model_config: SimpleNNConfig):
        super().__init__()
        self.config = model_config # Store model-specific config
        
        activation_layer_class: Type[nn.Module]
        if self.config.activation_fn == "ReLU":
            activation_layer_class = nn.ReLU
        elif self.config.activation_fn == "Sigmoid":
            activation_layer_class = nn.Sigmoid
        elif self.config.activation_fn == "Tanh":
            activation_layer_class = nn.Tanh
        elif self.config.activation_fn == "GELU":
            activation_layer_class = nn.GELU
        else:
            raise ValueError(f"Unsupported activation function: {self.config.activation_fn}")

        self.layers = nn.Sequential(
            nn.Linear(self.config.input_dim, self.config.hidden_dim),
            activation_layer_class(),
            nn.Dropout(self.config.dropout_rate),
            nn.Linear(self.config.hidden_dim, self.config.output_dim)
        )
        logger.debug(f"SimpleNN initialized with input_dim={self.config.input_dim}, "
                     f"hidden_dim={self.config.hidden_dim}, output_dim={self.config.output_dim}, "
                     f"dropout={self.config.dropout_rate}, activation={self.config.activation_fn}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the simple NN."""
        return self.layers(x)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        A single training step for the simple NN.
        Expects 'features' and 'labels' in the batch.
        """
        inputs = batch['features']
        targets = batch['labels']
        
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        
        # For classification, 'predictions' usually refers to raw logits before softmax.
        return {'loss': loss, 'predictions': logits}

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        A single validation step for the simple NN.
        Expects 'features' and 'labels' in the batch.
        """
        inputs = batch['features']
        targets = batch['labels']
        
        logits = self.forward(inputs)
        loss = self.loss_fn(logits, targets)
        
        return {'loss': loss, 'predictions': logits}

    def configure_optimizer(self, optimizer_name: str, learning_rate: float) -> optim.Optimizer:
        """
        Configures the optimizer for the simple NN based on the provided name and learning rate.
        """
        optimizer_class = getattr(optim, optimizer_name)
        return optimizer_class(self.parameters(), lr=learning_rate)

    def configure_loss_fn(self, loss_fn_name: str) -> nn.Module:
        """
        Configures the loss function for the simple NN based on the provided name.
        """
        loss_fn_class = getattr(nn, loss_fn_name)
        return loss_fn_class() # Assuming no special arguments needed for common loss fns
```

#### **3. `aiforge/plugins/simple_nn/dataset.py`**
A concrete implementation of `BaseDataset` that generates dummy classification data.

```python
# aiforge/plugins/simple_nn/dataset.py
import torch
from torch.utils.data import Dataset
from aiforge.core.data_pipeline import BaseDataset, BasePreprocessor # Import BasePreprocessor
from aiforge.core.logger import logger
from typing import Dict, Any, List, Optional

class Dataset(BaseDataset):
    """
    A dummy dataset for simple classification.
    Can also apply preprocessing if a preprocessor is provided.
    """
    def __init__(self, preprocessor: Optional[BasePreprocessor] = None):
        super().__init__()
        self.data: List[Dict[str, torch.Tensor]] = []
        self.preprocessor = preprocessor
        logger.debug("DummyDataset initialized.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        if self.preprocessor:
            return self.preprocessor.transform(item) # Apply transform on retrieval
        return item

    def load_data(self, data_path: str, num_samples: int = 1000, input_dim: int = 10, num_classes: int = 2):
        """Generates dummy data for classification."""
        logger.info(f"Generating {num_samples} dummy samples for input_dim={input_dim}, num_classes={num_classes}...")
        self.data = []
        # Use a fixed seed for reproducibility of dummy data
        torch.manual_seed(42) 
        for i in range(num_samples):
            features = torch.randn(input_dim) * 2 + 1 # Add some variance and mean offset
            labels = torch.randint(0, num_classes, (1,)).squeeze(0) 
            self.data.append({'features': features, 'labels': labels})
        logger.info("Dummy data loaded.")
```

#### **4. `aiforge/plugins/simple_nn/preprocessor.py`**
A concrete implementation of `BasePreprocessor` for simple feature scaling.

```python
# aiforge/plugins/simple_nn/preprocessor.py
import pickle
from aiforge.core.data_pipeline import BasePreprocessor
from aiforge.core.logger import logger
from typing import Any, List, Dict, Optional
import torch

class Preprocessor(BasePreprocessor):
    """
    A dummy preprocessor for standardizing features (mean 0, std 1).
    Implements BasePreprocessor interface.
    """
    def __init__(self):
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        logger.debug("DummyPreprocessor initialized.")

    def fit(self, features_list: List[torch.Tensor], **kwargs):
        """Fits a standard scaler (mean and std dev) to a list of feature tensors."""
        if features_list:
            all_features = torch.stack(features_list)
            self.mean = torch.mean(all_features, dim=0)
            # Add epsilon to prevent division by zero in case of constant features
            self.std = torch.std(all_features, dim=0) + 1e-9 
            logger.info(f"Preprocessor fitted. Mean (avg): {self.mean.mean():.2f}, Std (avg): {self.std.mean():.2f}")
        else:
            logger.warning("No data provided to fit preprocessor.")

    def transform(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Applies standardization to 'features' in a single data item."""
        if self.mean is None or self.std is None:
            logger.warning("Preprocessor not fitted. Returning original data item.")
            return data_item
        
        if 'features' in data_item:
            data_item['features'] = (data_item['features'] - self.mean) / self.std
        return data_item

    def inverse_transform(self, data_item: Dict[str, Any]) -> Dict[str, Any]:
        """Applies inverse standardization to 'features' in a single data item."""
        if self.mean is None or self.std is None:
            logger.warning("Preprocessor not fitted. Cannot inverse transform.")
            return data_item
        
        if 'features' in data_item:
            data_item['features'] = (data_item['features'] * self.std) + self.mean
        return data_item

    def save(self, path: str):
        """Saves the preprocessor state (mean and std) using pickle."""
        # Overrides BasePreprocessor's warning
        with open(path, 'wb') as f:
            pickle.dump({'mean': self.mean, 'std': self.std}, f)
        logger.info(f"Preprocessor state saved to {path}")

    def load(self, path: str):
        """Loads the preprocessor state using pickle."""
        # Overrides BasePreprocessor's warning
        with open(path, 'rb') as f:
            state = pickle.load(f)
            self.mean = state['mean']
            self.std = state['std']
        logger.info(f"Preprocessor state loaded from {path}")
```

---

### **IV. Main Execution (`aiforge/main.py`)**

The entry point for running an experiment. It configures a specific experiment and passes it to the `Orchestrator`.

```python
# aiforge/main.py
import torch
from aiforge.core.orchestrator import Orchestrator
from aiforge.config.model_configs import ExperimentConfig, SimpleNNConfig, TrainingConfig
from aiforge.config.settings import settings
from aiforge.core.logger import logger

def run_simple_nn_experiment():
    logger.info("--- Setting up SimpleNN experiment configuration ---")

    # In a real scenario, data_path would point to actual data files.
    # For this demo, the SimpleNNDataset generates data internally.
    dummy_data_file = settings.DATA_DIR / "dummy_classification_data.pkl" 

    # 1. Model-specific configuration
    model_config = SimpleNNConfig(
        input_dim=10,        # Matches the dummy data generation
        hidden_dim=64,
        output_dim=2,        # For 2-class classification
        dropout_rate=0.3,
        activation_fn="ReLU"
    )

    # 2. Training loop configuration
    training_config = TrainingConfig(
        epochs=5,
        batch_size=16,
        learning_rate=0.005,
        optimizer_name="Adam",
        loss_fn_name="CrossEntropyLoss", # Appropriate for multi-class classification logits
        early_stopping_patience=2,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min"
    )

    # 3. Overall experiment configuration
    experiment_config = ExperimentConfig(
        experiment_name="simple_nn_classification_experiment",
        model_plugin="simple_nn",    # Refers to 'aiforge/plugins/simple_nn'
        dataset_plugin="simple_nn",  # Refers to 'aiforge/plugins/simple_nn'
        data_path=str(dummy_data_file),
        preprocess_args={"scaler_type": "standard"}, # Args for SimpleNN Preprocessor's fit method
        model_config=model_config,
        training_config=training_config
    )
    
    # 4. Initialize and run the Orchestrator
    orchestrator = Orchestrator(experiment_config)
    orchestrator.run_experiment()

if __name__ == "__main__":
    # Ensure Loguru's level is set based on AppSettings
    logger.level(settings.LOG_LEVEL)
    run_simple_nn_experiment()

```

#### **5. `requirements.txt`**

```
torch>=1.10.0
pydantic>=1.8.0
loguru>=0.5.3
# Optional:
# scikit-learn # if using sklearn for metrics/preprocessing
```

---

### **IV. How to Run & Key Principles**

**1. Setup:**
   - Create the directory structure as specified.
   - Place the Python files in their respective folders.
   - Install dependencies: `pip install -r requirements.txt`
   - You can create an `.env` file in the project root (`aiforge/`) to override settings, e.g.:
     ```
     AIFORGE_LOG_LEVEL=DEBUG
     AIFORGE_DEVICE=cpu
     ```

**2. Execution:**
   - Simply run `python aiforge/main.py` from your terminal in the `aiforge/` project root.
   - The `Orchestrator` will take care of everything:
     - Initializing Loguru.
     - Loading plugin components (`simple_nn` model, dataset, preprocessor).
     - Initializing the `SimpleNN` model, its `Adam` optimizer, and `CrossEntropyLoss`.
     - Generating dummy data.
     - Fitting and applying the `DummyPreprocessor`.
     - Running the training loop with validation and early stopping.
     - Saving model checkpoints and experiment results.

**3. Plugin Integration (Illustrating OCP & DIP):**
   - To integrate a **new model (e.g., a Transformer)**:
     1. Create `aiforge/plugins/my_transformer/`.
     2. Implement `aiforge/plugins/my_transformer/model.py` inheriting from `BaseModel`.
     3. Create a Pydantic config for it (e.g., `TransformerConfig`) inheriting `ModelSpecificConfig` in `aiforge/config/model_configs.py`.
     4. Update `main.py` to use `"my_transformer"` as `model_plugin` and pass `TransformerConfig`.
   - To integrate a **new dataset (e.g., an ImageDataset)**:
     1. Create `aiforge/plugins/image_dataset/`.
     2. Implement `aiforge/plugins/image_dataset/dataset.py` inheriting `BaseDataset`.
     3. (Optionally) Implement `aiforge/plugins/image_dataset/preprocessor.py` inheriting `BasePreprocessor`.
     4. Update `main.py` to use `"image_dataset"` as `dataset_plugin`.

**4. SOLID Principles in Action:**
   - **SRP (Single Responsibility Principle):**
     - `Orchestrator`: Manages experiment flow.
     - `BaseModel`: Defines model interface.
     - `BaseDataset`/`BasePreprocessor`: Handle data logic.
     - `AppSettings`/`ExperimentConfig`: Manage configuration.
   - **OCP (Open/Closed Principle):**
     - New models, datasets, or preprocessors can be added as plugins without modifying the `core/` modules. The abstract base classes provide stable interfaces that are open for extension but closed for modification.
   - **LSP (Liskov Substitution Principle):**
     - The `Orchestrator` interacts with `BaseModel`, `BaseDataset`, `BasePreprocessor`. Any concrete implementation (e.g., `Model` from `simple_nn` plugin) can be substituted without affecting the orchestrator's correctness.
   - **ISP (Interface Segregation Principle):**
     - `BaseModel` focuses on model operations. `BaseDataset` on dataset operations. There are no "fat" interfaces forcing unrelated methods.
   - **DIP (Dependency Inversion Principle):**
     - The `Orchestrator` (high-level module) depends on `BaseModel`, `BaseDataset`, `BasePreprocessor` (abstractions), not directly on `SimpleNN.Model` (low-level module). Concrete plugin implementations are dynamically loaded and injected.

---


---

### **AIForge: Enterprise-Ready AI Framework**

---

### **I. Updated Directory Structure**

```
aiforge/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── logger.py
│   ├── base_model.py         # Updated for DDP-aware checkpointing
│   ├── data_pipeline.py
│   └── orchestrator.py       # Updated for DDP, MLflow, and injected dependencies
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── model_configs.py      # Updated with distributed training options
├── plugins/
│   ├── __init__.py
│   └── simple_nn/
│       ├── __init__.py
│       ├── model.py
│       ├── dataset.py
│       └── preprocessor.py
├── utils/
│   ├── __init__.py
│   ├── metrics.py
│   └── callbacks.py          # Added MLflowCallback
├── container.py              # NEW: Dependency Injection Container definition
├── main.py                   # Updated to use the DI container and manage DDP setup
└── requirements.txt          # Updated dependencies
```

---

### **II. Updated & New Components**

#### **1. `requirements.txt`**
Add the new dependencies.

```
# aiforge/requirements.txt
torch>=1.10.0
pydantic>=1.8.0
loguru>=0.5.3
mlflow>=1.20.0
dependency-injector>=4.0.0
# Optional:
# scikit-learn # if using sklearn for metrics/preprocessing
```

#### **2. `aiforge/config/model_configs.py`**
Add distributed training configuration.

```python
# aiforge/config/model_configs.py
from pydantic import BaseModel, Field
from typing import Dict, Any, Literal, Optional

class TrainingConfig(BaseModel):
    """Configuration for the training loop."""
    epochs: int = Field(10, description="Number of training epochs")
    batch_size: int = Field(32, description="Batch size for training and validation")
    learning_rate: float = Field(0.001, description="Learning rate for the optimizer")
    optimizer_name: Literal["Adam", "SGD", "RMSprop"] = Field("Adam", description="Name of the PyTorch optimizer")
    loss_fn_name: Literal["CrossEntropyLoss", "MSELoss", "BCELoss", "L1Loss"] = Field("CrossEntropyLoss", description="Name of the PyTorch loss function")
    gradient_clip_norm: Optional[float] = Field(None, description="Clip gradients to this max norm, if provided")
    early_stopping_patience: Optional[int] = Field(None, description="Number of epochs to wait for improvement before stopping")
    early_stopping_monitor: str = Field("val_loss", description="Metric to monitor for early stopping")
    early_stopping_mode: Literal["min", "max"] = Field("min", description="Mode for early stopping (min for loss, max for accuracy)")
    
    # NEW: Distributed Training Configuration
    use_distributed: bool = Field(False, description="Whether to use DistributedDataParallel (requires torch.distributed.init_process_group)")
    local_rank: int = Field(0, description="Local rank for distributed training (will be set by launcher)")
    world_size: int = Field(1, description="Total number of distributed processes (will be set by launcher)")


class ModelSpecificConfig(BaseModel):
    """Base class for model-specific configurations. Plugins will extend this."""
    pass

class SimpleNNConfig(ModelSpecificConfig):
    """Configuration for the Simple Neural Network plugin."""
    input_dim: int = Field(..., description="Input dimension of the model")
    hidden_dim: int = Field(128, description="Dimension of the hidden layer")
    output_dim: int = Field(..., description="Output dimension of the model")
    dropout_rate: float = Field(0.2, ge=0.0, le=1.0, description="Dropout rate for regularization")
    activation_fn: Literal["ReLU", "Sigmoid", "Tanh", "GELU"] = Field("ReLU", description="Activation function to use in hidden layers")

class ExperimentConfig(BaseModel):
    """Overall experiment configuration."""
    experiment_name: str = Field("default_experiment", description="Unique name for the experiment results folder")
    model_plugin: str = Field(..., description="Name of the model plugin to use (e.g., 'simple_nn')")
    dataset_plugin: str = Field(..., description="Name of the dataset plugin to use (e.g., 'simple_nn')")
    
    data_path: str = Field(..., description="Path to the raw data file(s) for the dataset")
    preprocess_args: Dict[str, Any] = Field(default_factory=dict, description="Keyword arguments for the preprocessor's fit method")

    model_config: ModelSpecificConfig = Field(..., description="Model-specific configuration instance")
    training_config: TrainingConfig = Field(default_factory=TrainingConfig, description="Training loop configuration")

    class Config:
        pass
```

#### **3. `aiforge/core/base_model.py`**
Update `save_checkpoint` to handle `DDP` wrapped models.

```python
# aiforge/core/base_model.py
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, Any, Type, Optional
from aiforge.core.logger import logger

class BaseModel(nn.Module, ABC):
    """
    Abstract Base Class for all AI models.
    Adheres to:
    - OCP: Models extend BaseModel without modifying Orchestrator.
    - LSP: Concrete models are substitutable for BaseModel.
    """

    def __init__(self, model_config: Any): # model_config type is Any, as it's polymorphic
        super().__init__()
        self.model_config = model_config
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.loss_fn: Optional[nn.Module] = None
        self.device: Optional[torch.device] = None # Will be set by Orchestrator

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Defines the forward pass of the model."""
        pass

    @abstractmethod
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Defines a single training step.
        Should return a dictionary containing 'loss' and any relevant metrics or predictions.
        """
        pass

    @abstractmethod
    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> Dict[str, Any]:
        """
        Defines a single validation step.
        Should return a dictionary containing 'loss' and any relevant metrics or predictions.
        """
        pass

    @abstractmethod
    def configure_optimizer(self, optimizer_name: str, learning_rate: float) -> torch.optim.Optimizer:
        """
        Configures and returns the optimizer for the model's parameters.
        The model determines *how* its parameters are optimized, given the *choice* (optimizer_name)
        and hyperparameters (learning_rate) from the Orchestrator.
        """
        pass
    
    @abstractmethod
    def configure_loss_fn(self, loss_fn_name: str) -> nn.Module:
        """
        Configures and returns the loss function for the model.
        The model defines *what* loss function is instantiated, given the *choice* (loss_fn_name)
        from the Orchestrator.
        """
        pass

    def save_checkpoint(self, path: str, is_distributed: bool = False):
        """
        Saves the model's state dictionary.
        Handles DDP wrapped models by saving the underlying module's state.
        """
        if is_distributed and isinstance(self, nn.parallel.DistributedDataParallel):
            torch.save(self.module.state_dict(), path)
        else:
            torch.save(self.state_dict(), path)
        logger.info(f"Model checkpoint saved to {path}")

    def load_checkpoint(self, path: str):
        """Loads the model's state dictionary."""
        self.load_state_dict(torch.load(path, map_location=self.device))
        logger.info(f"Model checkpoint loaded from {path}")

```

#### **4. `aiforge/utils/callbacks.py`**
Add `MLflowCallback` for experiment tracking.

```python
# aiforge/utils/callbacks.py
from aiforge.core.logger import logger
from typing import Dict, Any, Optional
import mlflow # NEW
import json # NEW

class Callback:
    """Base class for training callbacks."""
    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        """Called at the end of each epoch."""
        pass
    
    # NEW: Lifecycle hooks for MLflow
    def on_experiment_start(self, config: Any, run_id: Optional[str] = None):
        """Called once at the start of the experiment."""
        pass

    def on_experiment_end(self, status: str):
        """Called once at the end of the experiment."""
        pass

class EarlyStopping(Callback):
    """
    Callback for early stopping during training.
    Stops training if the monitored metric does not improve for a given patience.
    """
    def __init__(self, patience: int, min_delta: float = 0.001, monitor: str = 'val_loss', mode: str = 'min'):
        if mode not in ['min', 'max']:
            raise ValueError("EarlyStopping mode must be 'min' or 'max'.")
        self.patience = patience
        self.min_delta = abs(min_delta) # delta should always be positive
        self.monitor = monitor
        self.mode = mode
        self.best_value: float = float('inf') if mode == 'min' else float('-inf')
        self.epochs_no_improve: int = 0
        self.stop_training: bool = False
        logger.info(f"EarlyStopping initialized: monitor='{monitor}', patience={patience}, min_delta={min_delta}, mode='{mode}'")

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        current_value = logs.get(self.monitor)
        if current_value is None:
            logger.warning(f"EarlyStopping monitor '{self.monitor}' not found in logs for epoch {epoch}. Skipping check.")
            return

        if self.mode == 'min':
            if current_value < self.best_value - self.min_delta:
                self.best_value = current_value
                self.epochs_no_improve = 0
                logger.debug(f"Epoch {epoch}: {self.monitor} improved to {current_value:.4f}.")
            else:
                self.epochs_no_improve += 1
                logger.debug(f"Epoch {epoch}: {self.monitor} did not improve ({current_value:.4f} vs best {self.best_value:.4f}). "
                             f"No improvement for {self.epochs_no_improve} epochs.")
        else: # mode == 'max'
            if current_value > self.best_value + self.min_delta:
                self.best_value = current_value
                self.epochs_no_improve = 0
                logger.debug(f"Epoch {epoch}: {self.monitor} improved to {current_value:.4f}.")
            else:
                self.epochs_no_improve += 1
                logger.debug(f"Epoch {epoch}: {self.monitor} did not improve ({current_value:.4f} vs best {self.best_value:.4f}). "
                             f"No improvement for {self.epochs_no_improve} epochs.")
        
        if self.epochs_no_improve >= self.patience:
            self.stop_training = True
            logger.info(f"Early stopping triggered for '{self.monitor}' after {epoch+1} epochs.")

class MLflowCallback(Callback): # NEW
    """
    Callback to log metrics, parameters, and artifacts to MLflow.
    Adheres to OCP: Extends Callback without changing Orchestrator core logic.
    """
    def __init__(self, experiment_name: str, run_id: Optional[str] = None):
        self.experiment_name = experiment_name
        self.run_id = run_id
        
        # MLflow setup (only for rank 0 in DDP)
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            mlflow.set_experiment(experiment_name)
            logger.info(f"MLflowCallback initialized for experiment: {experiment_name}")
            if run_id:
                logger.info(f"MLflow run_id: {run_id}")

    def on_experiment_start(self, config: Any, run_id: Optional[str] = None):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            if run_id:
                mlflow.start_run(run_id=run_id)
            else:
                mlflow.start_run()
            self.run_id = mlflow.active_run().info.run_id
            logger.info(f"MLflow run started with ID: {self.run_id}")
            
            # Log all Pydantic config as parameters
            mlflow.log_params(config.model_dump()) # Pydantic v2 .model_dump(), v1 .dict()

    def on_epoch_end(self, epoch: int, logs: Dict[str, Any]):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_metrics(logs, step=epoch)
            logger.debug(f"MLflow logged metrics for epoch {epoch}")

    def on_experiment_end(self, status: str, model_path: str, preprocessor_path: Optional[str] = None, config_path: Optional[str] = None):
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            with mlflow.start_run(run_id=self.run_id):
                mlflow.log_artifact(model_path)
                if preprocessor_path and preprocessor_path.exists():
                    mlflow.log_artifact(str(preprocessor_path))
                if config_path and config_path.exists():
                    mlflow.log_artifact(str(config_path))
                mlflow.set_tag("status", status)
            mlflow.end_run()
            logger.info(f"MLflow run {self.run_id} ended with status: {status}")

```

#### **5. `aiforge/core/orchestrator.py`**
Major updates for Distributed Training and integration with injected dependencies.

```python
# aiforge/core/orchestrator.py
import torch
import torch.nn as nn # NEW
import torch.optim as optim # NEW
from torch.utils.data import DataLoader, DistributedSampler # NEW
import torch.distributed as dist # NEW
from aiforge.core.logger import logger
from aiforge.core.base_model import BaseModel
from aiforge.core.data_pipeline import BaseDataset, BasePreprocessor
from aiforge.config.model_configs import ExperimentConfig, TrainingConfig
from aiforge.config.settings import settings
from aiforge.utils.callbacks import EarlyStopping, MLflowCallback, Callback # NEW: MLflowCallback
from typing import Type, Dict, Any, Optional, List, Tuple
import json
import importlib

class Orchestrator:
    """
    Manages the end-to-end AI experiment lifecycle:
    configuration, data loading, preprocessing, model training, and evaluation.
    Adheres to SOLID principles by depending on abstractions.
    """
    def __init__(self, 
                 experiment_config: ExperimentConfig,
                 ModelClass: Type[BaseModel], # NEW: Injected dependency
                 DatasetClass: Type[BaseDataset], # NEW: Injected dependency
                 PreprocessorClass: Optional[Type[BasePreprocessor]] = None, # NEW: Injected dependency
                 mlflow_run_id: Optional[str] = None # NEW: Passed from main.py if distributed
                 ):
        self.config = experiment_config
        self.mlflow_run_id = mlflow_run_id # Pass run_id for distributed mlflow logging

        # NEW: Distributed setup based on config
        self.is_distributed = self.config.training_config.use_distributed and dist.is_initialized()
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            self.device = torch.device(f"cuda:{self.rank}")
            logger.info(f"Rank {self.rank}/{self.world_size} initialized on {self.device}")
        else:
            self.rank = 0 # Default to rank 0 for non-distributed runs
            self.world_size = 1
            self.device = torch.device(settings.DEVICE)
            logger.info(f"Non-distributed run initialized on {self.device}")
        
        self.ModelClass = ModelClass
        self.DatasetClass = DatasetClass
        self.PreprocessorClass = PreprocessorClass

        self.model: Optional[BaseModel] = None
        self.dataset: Optional[BaseDataset] = None
        self.preprocessor: Optional[BasePreprocessor] = None
        
        self.train_dataloader: Optional[DataLoader] = None
        self.val_dataloader: Optional[DataLoader] = None
        
        self._initialize_model() # Now happens with loaded ModelClass
        
        # Setup result directory (only for rank 0 or non-distributed)
        self.experiment_results_dir = settings.RESULTS_DIR / self.config.experiment_name
        if self.rank == 0:
            self.experiment_results_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Experiment results will be stored in: {self.experiment_results_dir}")

        self.callbacks: List[Callback] = []
        if self.config.training_config.early_stopping_patience:
            self.callbacks.append(
                EarlyStopping(
                    patience=self.config.training_config.early_stopping_patience,
                    monitor=self.config.training_config.early_stopping_monitor,
                    mode=self.config.training_config.early_stopping_mode
                )
            )
        # NEW: MLflow callback (only for rank 0 or non-distributed)
        if self.rank == 0:
            self.mlflow_callback = MLflowCallback(self.config.experiment_name, run_id=self.mlflow_run_id)
            self.callbacks.append(self.mlflow_callback)
            self.mlflow_callback.on_experiment_start(self.config, run_id=self.mlflow_run_id) # Start MLflow run
        else:
            self.mlflow_callback = None


    def _initialize_model(self):
        """
        Initializes the model, its optimizer, and loss function.
        Model-specific configuration is passed to the model's constructor.
        Wraps the model in DDP if distributed training is enabled.
        """
        if self.ModelClass is None:
            raise RuntimeError("ModelClass not loaded.")

        self.model = self.ModelClass(model_config=self.config.model_config)
        self.model.device = self.device # Set model's device property
        self.model = self.model.to(self.device)
        
        # NEW: Wrap model with DistributedDataParallel
        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[self.device.index])
            logger.info(f"Rank {self.rank}: Model wrapped in DistributedDataParallel.")
        
        self.model.optimizer = self.model.configure_optimizer(
            self.config.training_config.optimizer_name,
            self.config.training_config.learning_rate
        )
        self.model.loss_fn = self.model.configure_loss_fn(
            self.config.training_config.loss_fn_name
        )
        logger.info(f"Rank {self.rank}: Model initialized: {self.model.__class__.__name__} on {self.device}")
        if self.rank == 0: # Only print architecture from rank 0
            logger.debug(f"Model Architecture:\n{self.model}")

    def prepare_data(self):
        """
        Loads the dataset and optionally applies preprocessing.
        Uses DistributedSampler if distributed training is enabled.
        """
        if self.DatasetClass is None:
            raise RuntimeError("DatasetClass not loaded.")

        if self.rank == 0: # Only log data loading from rank 0
            logger.info(f"Loading data from {self.config.data_path} using {self.DatasetClass.__name__}...")
        self.dataset = self.DatasetClass()
        self.dataset.load_data(self.config.data_path, # Pass rank/world_size if dataset needs it
                                rank=self.rank, 
                                world_size=self.world_size,
                                is_distributed=self.is_distributed) 

        if self.PreprocessorClass:
            self.preprocessor = self.PreprocessorClass()
            if self.rank == 0: # Only fit/save preprocessor from rank 0
                logger.info(f"Fitting preprocessor {self.preprocessor.__class__.__name__}...")
                all_features = [self.dataset[i]['features'] for i in range(len(self.dataset))]
                self.preprocessor.fit(all_features, **self.config.preprocess_args)
                self.preprocessor.save(self.experiment_results_dir / 'preprocessor.pkl')
                logger.info(f"Preprocessor fitted and saved to {self.experiment_results_dir / 'preprocessor.pkl'}")
            
            # Ensure preprocessor is loaded on all ranks if it was saved by rank 0
            if self.is_distributed:
                dist.barrier() # Wait for rank 0 to save preprocessor
            if self.rank != 0 and self.experiment_results_dir.exists():
                self.preprocessor.load(self.experiment_results_dir / 'preprocessor.pkl')

            self.dataset.preprocessor = self.preprocessor 
            logger.debug(f"Rank {self.rank}: Preprocessor state attached to dataset.")
        
        # NEW: DistributedSampler for DDP
        train_sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=True) if self.is_distributed else None
        val_sampler = DistributedSampler(self.dataset, num_replicas=self.world_size, rank=self.rank, shuffle=False) if self.is_distributed else None

        # Split dataset for training and validation (apply only on main dataset if using sampler)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size
        
        # Create dummy generator for random_split if not using DistributedSampler.
        # When using DistributedSampler, the dataset itself is not split, the sampler handles it.
        if not self.is_distributed:
            generator = torch.Generator().manual_seed(42) 
            train_dataset, val_dataset = torch.utils.data.random_split(self.dataset, [train_size, val_size], generator=generator)
        else:
            # When using DDP, usually the entire dataset is passed to DataLoader, and the sampler
            # takes care of partitioning data for each rank.
            train_dataset = self.dataset # The DistributedSampler will select subsets for each rank
            val_dataset = self.dataset # Same for validation for simplicity, in production you might have a dedicated val set.


        self.train_dataloader = DataLoader(
            train_dataset, 
            batch_size=self.config.training_config.batch_size, 
            shuffle=(train_sampler is None), # Shuffle only if no sampler is provided
            sampler=train_sampler
        )
        self.val_dataloader = DataLoader(
            val_dataset, 
            batch_size=self.config.training_config.batch_size, 
            shuffle=(val_sampler is None),
            sampler=val_sampler
        )
        if self.rank == 0:
            logger.info(f"Data prepared: {len(train_dataset)} training samples (per rank if distributed), "
                        f"{len(val_dataset)} validation samples (per rank if distributed).")
        logger.debug(f"Rank {self.rank}: Dataloaders created with batch size {self.config.training_config.batch_size}.")


    def train(self):
        """Executes the training loop."""
        if not self.model or not self.train_dataloader or not self.val_dataloader:
            raise RuntimeError("Model or data not prepared for training.")

        if self.rank == 0:
            logger.info(f"Starting training for {self.config.training_config.epochs} epochs...")
        
        best_monitor_value = float('inf') if self.config.training_config.early_stopping_mode == 'min' else float('-inf')
        
        for epoch in range(self.config.training_config.epochs):
            if self.is_distributed:
                self.train_dataloader.sampler.set_epoch(epoch) # Important for shuffling across epochs

            self.model.train()
            total_train_loss = 0
            for batch_idx, batch in enumerate(self.train_dataloader):
                self.model.optimizer.zero_grad()
                
                inputs = batch['features'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                step_outputs = self.model.training_step({'features': inputs, 'labels': targets}, batch_idx)
                loss = step_outputs['loss']
                
                loss.backward()

                if self.config.training_config.gradient_clip_norm:
                    # Clip gradients on all model parameters (including DDP module)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.training_config.gradient_clip_norm)

                self.model.optimizer.step()
                total_train_loss += loss.item()
            
            avg_train_loss = total_train_loss / len(self.train_dataloader)
            
            # Validation step
            avg_val_loss, val_metrics = self.evaluate(self.val_dataloader)
            
            # For DDP, gather metrics from all ranks (only if actually distributed and on rank 0)
            if self.is_distributed:
                dist.barrier() # Wait for all ranks to complete epoch and evaluation
                # Reduce average train loss across all ranks
                gathered_train_loss = [torch.tensor(0.0).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(gathered_train_loss, torch.tensor(avg_train_loss).to(self.device))
                avg_train_loss = torch.tensor(gathered_train_loss).mean().item()

                # Reduce average val loss across all ranks
                gathered_val_loss = [torch.tensor(0.0).to(self.device) for _ in range(self.world_size)]
                dist.all_gather(gathered_val_loss, torch.tensor(avg_val_loss).to(self.device))
                avg_val_loss = torch.tensor(gathered_val_loss).mean().item()

                # Reduce other metrics (e.g., accuracy)
                for metric_name, metric_value in list(val_metrics.items()): # Use list to modify dict during iteration
                    gathered_metric = [torch.tensor(0.0).to(self.device) for _ in range(self.world_size)]
                    dist.all_gather(gathered_metric, torch.tensor(metric_value).to(self.device))
                    val_metrics[metric_name] = torch.tensor(gathered_metric).mean().item()
            
            if self.rank == 0: # Only log and apply callbacks on rank 0
                logs = {
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    **val_metrics
                }
                logger.info(f"Epoch {epoch+1}/{self.config.training_config.epochs} | "
                            f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | "
                            f"Val Metrics: {val_metrics}")

                # Callbacks
                for callback in self.callbacks:
                    callback.on_epoch_end(epoch, logs)
                    if isinstance(callback, EarlyStopping) and callback.stop_training:
                        logger.info(f"Early stopping triggered after {epoch+1} epochs.")
                        self.model.save_checkpoint(self.experiment_results_dir / "best_model.pth", is_distributed=self.is_distributed)
                        return # Exit training loop

                # Update best model based on monitor metric
                monitor_value = logs.get(self.config.training_config.early_stopping_monitor)
                if monitor_value is not None:
                    if (self.config.training_config.early_stopping_mode == 'min' and monitor_value < best_monitor_value) or \
                       (self.config.training_config.early_stopping_mode == 'max' and monitor_value > best_monitor_value):
                        best_monitor_value = monitor_value
                        self.model.save_checkpoint(self.experiment_results_dir / "best_model.pth", is_distributed=self.is_distributed)
                        logger.debug(f"Saved best model to {self.experiment_results_dir / 'best_model.pth'}")

        if self.rank == 0:
            logger.info("Training complete.")
            # If no early stopping, ensure the best model is loaded if it was ever saved.
            if (self.experiment_results_dir / "best_model.pth").exists():
                self.model.load_checkpoint(self.experiment_results_dir / "best_model.pth")
        
        if self.is_distributed: # Ensure all ranks exit gracefully
            dist.barrier()


    def evaluate(self, dataloader: DataLoader) -> Tuple[float, Dict[str, Any]]:
        """Evaluates the model on a given dataloader."""
        if not self.model:
            raise RuntimeError("Model not initialized for evaluation.")

        self.model.eval()
        total_loss = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(dataloader):
                inputs = batch['features'].to(self.device)
                targets = batch['labels'].to(self.device)
                
                step_outputs = self.model.validation_step({'features': inputs, 'labels': targets}, batch_idx)
                loss = step_outputs['loss']
                total_loss += loss.item()
                
                if 'predictions' in step_outputs:
                    # Ensure predictions from DDP model are gathered correctly if needed,
                    # here we just take the output of the local model
                    # For simplicity, gathering only for accuracy calculation.
                    all_preds.append(step_outputs['predictions'].cpu())
                all_targets.append(targets.cpu())

        avg_loss = total_loss / len(dataloader)
        
        metrics = {}
        if all_preds and all_targets:
            if self.config.training_config.loss_fn_name == "CrossEntropyLoss":
                preds = torch.cat(all_preds)
                targets = torch.cat(all_targets)
                correct = (preds.argmax(dim=1) == targets).sum().item()
                accuracy_val = correct / len(targets)
                metrics['accuracy'] = accuracy_val
            # Add other metrics as needed based on the task

        return avg_loss, metrics

    def run_experiment(self):
        """Runs the entire experiment: data prep, train, final eval."""
        if self.rank == 0:
            logger.info(f"--- Starting Experiment: {self.config.experiment_name} ---")
            
            # Save experiment config (only on rank 0)
            config_path = self.experiment_results_dir / "experiment_config.json"
            with open(config_path, "w") as f:
                f.write(self.config.json(indent=2))
            logger.info(f"Experiment configuration saved to {config_path}")

        self.prepare_data()
        self.train()
        
        # Final evaluation and results saving (only on rank 0)
        if self.rank == 0:
            logger.info("Running final evaluation on validation set...")
            final_val_loss, final_val_metrics = self.evaluate(self.val_dataloader)
            logger.info(f"Final Validation Loss: {final_val_loss:.4f} | Final Metrics: {final_val_metrics}")
            
            # Save final model
            final_model_path = self.experiment_results_dir / "final_model.pth"
            self.model.save_checkpoint(final_model_path, is_distributed=self.is_distributed)
            
            final_results = {
                "final_val_loss": final_val_loss,
                "final_val_metrics": final_val_metrics,
                "model_path": str(final_model_path),
                "preprocessor_path": str(self.experiment_results_dir / 'preprocessor.pkl') if self.preprocessor else None,
                "config_path": str(config_path)
            }
            with open(self.experiment_results_dir / "final_results.json", "w") as f:
                json.dump(final_results, f, indent=2)

            logger.info(f"Experiment '{self.config.experiment_name}' finished.")
            logger.info(f"Results available in {self.experiment_results_dir}")
            
            # NEW: End MLflow run
            if self.mlflow_callback:
                self.mlflow_callback.on_experiment_end("FINISHED", str(final_model_path), 
                                                        preprocessor_path=self.experiment_results_dir / 'preprocessor.pkl' if self.preprocessor else None,
                                                        config_path=config_path)
        
        if self.is_distributed: # Ensure all ranks exit gracefully
            dist.destroy_process_group() # Clean up DDP process group
```

#### **6. `aiforge/plugins/simple_nn/dataset.py`**
Update `load_data` to handle `DistributedSampler` arguments, even if not directly used for splitting here.

```python
# aiforge/plugins/simple_nn/dataset.py
import torch
from torch.utils.data import Dataset
from aiforge.core.data_pipeline import BaseDataset, BasePreprocessor # Import BasePreprocessor
from aiforge.core.logger import logger
from typing import Dict, Any, List, Optional

class Dataset(BaseDataset):
    """
    A dummy dataset for simple classification.
    Can also apply preprocessing if a preprocessor is provided.
    """
    def __init__(self, preprocessor: Optional[BasePreprocessor] = None):
        super().__init__()
        self.data: List[Dict[str, torch.Tensor]] = []
        self.preprocessor = preprocessor
        logger.debug("DummyDataset initialized.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        if self.preprocessor:
            return self.preprocessor.transform(item) # Apply transform on retrieval
        return item

    def load_data(self, data_path: str, num_samples: int = 1000, input_dim: int = 10, num_classes: int = 2, # NEW: DDP args
                  rank: int = 0, world_size: int = 1, is_distributed: bool = False): # NEW
        """
        Generates dummy data for classification.
        DDP args (rank, world_size) are accepted but not directly used for data generation
        since DistributedSampler handles partitioning from the full dataset.
        """
        if rank == 0: # Only log data generation from rank 0
            logger.info(f"Generating {num_samples} dummy samples for input_dim={input_dim}, num_classes={num_classes}...")
        self.data = []
        # Use a fixed seed for reproducibility of dummy data
        torch.manual_seed(42) 
        for i in range(num_samples):
            features = torch.randn(input_dim) * 2 + 1 # Add some variance and mean offset
            labels = torch.randint(0, num_classes, (1,)).squeeze(0) 
            self.data.append({'features': features, 'labels': labels})
        if rank == 0:
            logger.info("Dummy data loaded.")

```

#### **7. `aiforge/container.py` (NEW FILE)**
This file defines the `dependency-injector` container, which will manage the creation and wiring of all components.

```python
# aiforge/container.py
from dependency_injector import containers, providers
import importlib
from typing import Type, Optional

from aiforge.core.orchestrator import Orchestrator
from aiforge.core.base_model import BaseModel
from aiforge.core.data_pipeline import BaseDataset, BasePreprocessor
from aiforge.config.model_configs import ExperimentConfig, SimpleNNConfig, TrainingConfig, ModelSpecificConfig
from aiforge.config.settings import settings
from aiforge.core.logger import logger

class Container(containers.DeclarativeContainer):
    """
    Dependency Injection Container for AIForge components.
    Adheres to DIP: Orchestrator receives abstractions, not concretions.
    """
    wiring_config = containers.WiringConfiguration(modules=["aiforge.main"])

    # --- Configuration providers ---
    experiment_config = providers.Singleton(
        ExperimentConfig,
        experiment_name=providers.Object(None), # Will be overridden at runtime
        model_plugin=providers.Object(None),    # Will be overridden at runtime
        dataset_plugin=providers.Object(None),  # Will be overridden at runtime
        data_path=providers.Object(None),       # Will be overridden at runtime
        model_config=providers.Object(None),    # Will be overridden at runtime
        training_config=providers.Singleton(TrainingConfig) # Can be overridden or used as default
    )

    # --- Plugin Loader provider ---
    # This provider dynamically loads the classes based on plugin names in ExperimentConfig
    # and ensures they conform to the abstract base classes.
    @providers.Factory
    def plugin_model_class(experiment_config_instance: ExperimentConfig) -> Type[BaseModel]:
        model_module_path = f"aiforge.plugins.{experiment_config_instance.model_plugin}.model"
        try:
            plugin_model_module = importlib.import_module(model_module_path)
            model_class = getattr(plugin_model_module, "Model")
            if not issubclass(model_class, BaseModel):
                raise TypeError(f"Model class '{model_class.__name__}' from plugin "
                                f"'{experiment_config_instance.model_plugin}' does not inherit from BaseModel.")
            return model_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load ModelClass for plugin '{experiment_config_instance.model_plugin}': {e}")
            raise

    @providers.Factory
    def plugin_dataset_class(experiment_config_instance: ExperimentConfig) -> Type[BaseDataset]:
        dataset_module_path = f"aiforge.plugins.{experiment_config_instance.dataset_plugin}.dataset"
        try:
            plugin_dataset_module = importlib.import_module(dataset_module_path)
            dataset_class = getattr(plugin_dataset_module, "Dataset")
            if not issubclass(dataset_class, BaseDataset):
                raise TypeError(f"Dataset class '{dataset_class.__name__}' from plugin "
                                f"'{experiment_config_instance.dataset_plugin}' does not inherit from BaseDataset.")
            return dataset_class
        except (ImportError, AttributeError) as e:
            logger.error(f"Failed to load DatasetClass for plugin '{experiment_config_instance.dataset_plugin}': {e}")
            raise

    @providers.Factory
    def plugin_preprocessor_class(experiment_config_instance: ExperimentConfig) -> Optional[Type[BasePreprocessor]]:
        preprocessor_module_path = f"aiforge.plugins.{experiment_config_instance.dataset_plugin}.preprocessor"
        try:
            plugin_preprocessor_module = importlib.import_module(preprocessor_module_path)
            preprocessor_class = getattr(plugin_preprocessor_module, "Preprocessor")
            if not issubclass(preprocessor_class, BasePreprocessor):
                raise TypeError(f"Preprocessor class '{preprocessor_class.__name__}' from plugin "
                                f"'{experiment_config_instance.dataset_plugin}' does not inherit from BasePreprocessor.")
            return preprocessor_class
        except (ImportError, AttributeError):
            logger.warning(f"No PreprocessorClass found for plugin '{experiment_config_instance.dataset_plugin}'.")
            return None

    # --- Orchestrator provider (injects dependencies) ---
    orchestrator = providers.Factory(
        Orchestrator,
        experiment_config=experiment_config,
        ModelClass=plugin_model_class,
        DatasetClass=plugin_dataset_class,
        PreprocessorClass=plugin_preprocessor_class,
        mlflow_run_id=providers.Object(None) # Will be set at runtime for MLflow
    )

# --- Dynamic Configuration Example (Outside of container definition, typically in main.py) ---
# To configure the ExperimentConfig dynamically, you'd do:
# container.experiment_config.override(ExperimentConfig(
#     experiment_name="my_dynamic_exp",
#     model_plugin="simple_nn",
#     dataset_plugin="simple_nn",
#     data_path="path/to/data",
#     model_config=SimpleNNConfig(input_dim=10, output_dim=2)
# ))
```

#### **8. `aiforge/main.py`**
Update the main entry point to utilize the DI container and manage DDP setup.

```python
# aiforge/main.py
import torch
import torch.distributed as dist # NEW
import os # NEW
import mlflow # NEW

from aiforge.core.orchestrator import Orchestrator
from aiforge.config.model_configs import ExperimentConfig, SimpleNNConfig, TrainingConfig
from aiforge.config.settings import settings
from aiforge.core.logger import logger
from aiforge.container import Container # NEW: Import the DI Container

def setup_distributed(rank, world_size, backend='nccl'): # NEW
    """Initializes the distributed environment."""
    os.environ['MASTER_ADDR'] = os.environ.get('MASTER_ADDR', 'localhost')
    os.environ['MASTER_PORT'] = os.environ.get('MASTER_PORT', '12355')
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    logger.info(f"Distributed process group initialized for rank {rank} / {world_size}.")
    torch.cuda.set_device(rank) # Set device for current process


def cleanup_distributed(): # NEW
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed process group destroyed.")


def run_simple_nn_experiment(config: ExperimentConfig, mlflow_run_id: Optional[str] = None): # NEW: Accepts config
    # Configure the DI container with the specific experiment config
    container = Container()
    container.experiment_config.override(config)
    container.mlflow_run_id.override(mlflow_run_id) # Pass MLflow run_id if available

    # Resolve Orchestrator from the container
    orchestrator: Orchestrator = container.orchestrator()
    orchestrator.run_experiment()

def main():
    # Ensure Loguru's level is set based on AppSettings
    logger.level(settings.LOG_LEVEL)

    # 1. --- Define experiment configurations (can be loaded from YAML/JSON too) ---
    dummy_data_file = settings.DATA_DIR / "dummy_classification_data.pkl" 

    model_config_instance = SimpleNNConfig(
        input_dim=10,        # Matches the dummy data generation
        hidden_dim=64,
        output_dim=2,        # For 2-class classification
        dropout_rate=0.3,
        activation_fn="ReLU"
    )

    training_config_instance = TrainingConfig(
        epochs=5,
        batch_size=16,
        learning_rate=0.005,
        optimizer_name="Adam",
        loss_fn_name="CrossEntropyLoss",
        early_stopping_patience=2,
        early_stopping_monitor="val_loss",
        early_stopping_mode="min",
        
        # NEW: Distributed training defaults (will be overridden by launcher)
        use_distributed=False, 
        local_rank=0,
        world_size=1
    )

    experiment_config_instance = ExperimentConfig(
        experiment_name="simple_nn_classification_experiment_enterprise", # New experiment name
        model_plugin="simple_nn",
        dataset_plugin="simple_nn",
        data_path=str(dummy_data_file),
        preprocess_args={"scaler_type": "standard"},
        model_config=model_config_instance,
        training_config=training_config_instance
    )

    # 2. --- Handle Distributed Training Setup ---
    # `torch.distributed.launch` or `torchrun` sets these env vars
    is_distributed_launch = int(os.environ.get('RANK', -1)) != -1 # Check if launched by torchrun
    if is_distributed_launch:
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        experiment_config_instance.training_config.use_distributed = True
        experiment_config_instance.training_config.local_rank = local_rank
        experiment_config_instance.training_config.world_size = world_size
        
        setup_distributed(global_rank, world_size) # Initialize DDP for this process
        logger.info(f"Main: DDP launcher detected. Running as global_rank {global_rank} / {world_size}.")
    else:
        logger.info("Main: Running in single-process mode.")

    # 3. --- MLflow setup (only for rank 0) ---
    mlflow_run_id: Optional[str] = None
    # For distributed runs, only rank 0 starts/ends the MLflow run.
    # Other ranks log to the same run_id.
    current_rank = int(os.environ.get('RANK', 0)) # Default to 0 if not DDP
    if current_rank == 0:
        mlflow.set_experiment(experiment_config_instance.experiment_name)
        with mlflow.start_run() as run:
            mlflow_run_id = run.info.run_id
            logger.info(f"Main: MLflow run initiated with ID: {mlflow_run_id}")
            # Log all config parameters directly to MLflow from rank 0
            mlflow.log_params(experiment_config_instance.model_dump())
    
    # NEW: Pass MLflow run_id to all orchestrator instances for logging
    if is_distributed_launch:
        # Use a barrier and broadcast the MLflow run_id from rank 0 to other ranks
        if current_rank == 0:
            run_id_tensor = torch.tensor([int(mlflow_run_id, 16)]).to(f"cuda:{local_rank}") # Convert UUID string to int
        else:
            run_id_tensor = torch.tensor([0]).to(f"cuda:{local_rank}")
        
        dist.barrier()
        dist.broadcast(run_id_tensor, src=0)
        if current_rank != 0:
            mlflow_run_id = hex(run_id_tensor.item())[2:] # Convert back to hex string
            logger.info(f"Main: Rank {current_rank} received MLflow run ID: {mlflow_run_id}")

    # 4. --- Run the experiment ---
    try:
        run_simple_nn_experiment(experiment_config_instance, mlflow_run_id)
    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
    finally:
        cleanup_distributed() # Clean up DDP process group

if __name__ == "__main__":
    main()

```

---

### **V. How to Run & Key Principles**

**1. Setup:**
   - Create the directory structure as specified.
   - Place the Python files in their respective folders.
   - Install dependencies: `pip install -r requirements.txt`
   - You can create an `.env` file in the project root (`aiforge/`) to override settings (e.g., `AIFORGE_DEVICE=cpu`).
   - For MLflow, ensure you have an MLflow tracking server running or it will log locally (`mlflow ui` to view local runs).

**2. Execution:**

   **a. Single-Process (non-distributed) Run:**
     - Simply run `python aiforge/main.py` from your terminal in the `aiforge/` project root.
     - MLflow will start a run and log to your default tracking URI (usually `mlruns/` locally).

   **b. Multi-GPU (Distributed) Run (requires PyTorch Distributed):**
     - You'll need a machine with multiple GPUs and PyTorch installed with CUDA support.
     - Use `torchrun` (or `torch.distributed.launch` for older PyTorch versions) to launch:
       ```bash
       torchrun --standalone --nproc_per_node=2 aiforge/main.py
       ```
       (Replace `2` with the number of GPUs you want to use.)
     - Each GPU will run a separate process, and the `Orchestrator` instances will coordinate. MLflow will log from rank 0.

**3. Key Principles in Action:**

   -   **Distributed Training (PyTorch DDP)**:
        -   The `Orchestrator` now detects if `dist.is_initialized()` and adapts.
        -   `torch.cuda.set_device(rank)` is called for each process.
        -   `nn.parallel.DistributedDataParallel` wraps the model.
        -   `DistributedSampler` is used with `DataLoader` to ensure each GPU gets a unique subset of data.
        -   Metrics are gathered from all ranks to rank 0 for aggregated logging.
        -   Model checkpoints are saved from the `model.module` in DDP, ensuring only the core model state is saved.
        -   `dist.barrier()` ensures synchronization between ranks, especially for preprocessor loading and MLflow logging.

   -   **Experiment Tracking (MLflow)**:
        -   `MLflowCallback` is a new `Callback` that the `Orchestrator` uses.
        -   It logs `ExperimentConfig` parameters at the start of the run (from rank 0).
        -   It logs epoch-end metrics (`train_loss`, `val_loss`, `accuracy`) during training.
        -   It logs the final model checkpoint, preprocessor, and experiment configuration as artifacts at the end of the run.
        -   Crucially, MLflow logging is only performed by `rank 0` in a distributed setting to avoid duplicate logs.

   -   **Dependency Injection Container (`dependency-injector`)**:
        -   The `Container` class in `aiforge/container.py` defines how `Orchestrator` and its dependencies are constructed.
        -   `Orchestrator`'s `__init__` method now directly accepts `ModelClass`, `DatasetClass`, `PreprocessorClass`, adhering to **DIP**.
        -   The `plugin_model_class`, `plugin_dataset_class`, and `plugin_preprocessor_class` providers in the `Container` dynamically import and type-check the plugin classes, effectively replacing the old `_load_plugin_components` logic within `Orchestrator`.
        -   `main.py` is simplified; it configures the `Container` and then `orchestrator = container.orchestrator()` takes care of the complex instantiation.

This enhanced AIForge framework is now equipped for enterprise-grade AI development, offering scalability, rigorous tracking, and flexible component management.
