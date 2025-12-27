



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

This framework provides a solid foundation for building and managing complex AI experiments with high engineering standards.
