---
title: callbacks
---



> Lightning callbacks for checkpoint and log management via ProjectIO.



::: {#2 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
from nbdev.showdoc import *
```
:::


::: {#3 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
from __future__ import annotationsfrom pathlib import Pathfrom typing import Any, Optionaltry:    from lightning.pytorch.callbacks import Callback, ModelCheckpoint    from lightning.pytorch import Trainer, LightningModule    HAS_LIGHTNING = Trueexcept ImportError:    # Lightning is optional - provide stub for type hints    HAS_LIGHTNING = False    class Callback:  # type: ignore        """Stub callback for when Lightning is not installed."""        pass    class ModelCheckpoint:  # type: ignore        pass    Trainer = Any  # type: ignore    LightningModule = Any  # type: ignorefrom projio.core import ProjectIO, PIO
```
:::


## IOCheckpointCallback

A callback that uses ProjectIO to place checkpoints and optionally tracks producers.

::: {#5 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
class IOCheckpointCallback(Callback):
    """Lightning callback that uses ProjectIO for checkpoint paths.
    
    This callback integrates with ProjectIO to:
    - Place checkpoints in the configured checkpoint directory
    - Apply datestamp prefixes/directories as configured
    - Optionally track which training script produced each checkpoint
    
    Parameters:
        io: ProjectIO instance (default: creates new one).
        run: Run name for subdirectory organization.
        filename: Checkpoint filename template with {epoch}, {step}, etc.
        datestamp: Override datestamp behavior.
        track_producer: Record producer info for checkpoints.
        producer_script: Script path to record as producer.
    
    Example:
        >>> callback = IOCheckpointCallback(run="experiment_1")
        >>> trainer = Trainer(callbacks=[callback])
    """
    
    def __init__(
        self,
        io: ProjectIO | None = None,
        run: str | None = None,
        filename: str = "{epoch:02d}-{step:06d}",
        datestamp: bool | None = None,
        track_producer: bool = False,
        producer_script: str | Path | None = None,
    ):
        super().__init__()
        self.io = io or ProjectIO()
        self.run = run
        self.filename = filename
        self.datestamp = datestamp
        self.track_producer = track_producer
        self.producer_script = Path(producer_script) if producer_script else None
        self.cached_checkpoint_dir: Path | None = None
    
    @property
    def checkpoint_dir(self) -> Path:
        """Get the checkpoint directory for this callback."""
        if self.cached_checkpoint_dir is None:
            self.cached_checkpoint_dir = self.io.tensorboard_run(
                run=self.run,
                datestamp=self.datestamp
            ).parent.parent / "checkpoints"
            if self.run:
                self.cached_checkpoint_dir = self.cached_checkpoint_dir / self.run
        return self.cached_checkpoint_dir
    
    def get_checkpoint_path(self, epoch: int, step: int, ext: str = ".ckpt") -> Path:
        """Build checkpoint path for given epoch and step.
        
        Parameters:
            epoch: Current epoch number.
            step: Current global step.
            ext: File extension.
            
        Returns:
            Full path to checkpoint file.
        """
        name = self.filename.format(epoch=epoch, step=step)
        return self.io.checkpoint_path(
            name=name,
            ext=ext,
            run=self.run,
            datestamp=self.datestamp
        )
    
    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        """Called when training starts - ensures checkpoint directory exists."""
        if not HAS_LIGHTNING:
            return
        # Pre-create the checkpoint directory
        _ = self.checkpoint_dir
    
    def on_save_checkpoint(
        self,
        trainer: "Trainer",
        pl_module: "LightningModule",
        checkpoint: dict
    ) -> None:
        """Called when a checkpoint is saved - track producer if enabled."""
        if not HAS_LIGHTNING or not self.track_producer:
            return
        if self.producer_script:
            # Get the checkpoint path that will be saved
            epoch = trainer.current_epoch
            step = trainer.global_step
            ckpt_path = self.get_checkpoint_path(epoch, step)
            self.io.track_producer(
                target=ckpt_path,
                producer=self.producer_script,
                kind="checkpoint"
            )
```
:::


## IOLogCallback

A callback that routes logs and tensorboard runs through ProjectIO.

::: {#7 .cell 0='e' 1='x' 2='p' 3='o' 4='r' 5='t'}
``` {.python .cell-code}
class IOLogCallback(Callback):
    """Lightning callback that routes logs through ProjectIO.
    
    This callback integrates with ProjectIO to:
    - Set up TensorBoard log directories
    - Apply datestamp prefixes/directories as configured
    - Provide consistent logging paths across experiments
    
    Parameters:
        io: ProjectIO instance (default: creates new one).
        run: Run name for subdirectory organization.
        datestamp: Override datestamp behavior.
    
    Example:
        >>> callback = IOLogCallback(run="experiment_1")
        >>> trainer = Trainer(callbacks=[callback])
    """
    
    def __init__(
        self,
        io: ProjectIO | None = None,
        run: str | None = None,
        datestamp: bool | None = None,
    ):
        super().__init__()
        self.io = io or ProjectIO()
        self.run = run
        self.datestamp = datestamp
        self.cached_log_dir: Path | None = None
    
    @property
    def log_dir(self) -> Path:
        """Get the TensorBoard log directory for this callback."""
        if self.cached_log_dir is None:
            self.cached_log_dir = self.io.tensorboard_run(
                run=self.run,
                datestamp=self.datestamp
            )
        return self.cached_log_dir
    
    def on_train_start(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        """Called when training starts - configure trainer log directory."""
        if not HAS_LIGHTNING:
            return
        # Ensure log directory exists
        log_dir = self.log_dir
        
        # Try to configure the trainer's logger if it has a log_dir attribute
        if hasattr(trainer, 'logger') and trainer.logger is not None:
            logger = trainer.logger
            if hasattr(logger, 'log_dir'):
                # Some loggers allow setting log_dir (external Lightning API)
                try:
                    setattr(logger, 'log_dir', str(log_dir))
                except (AttributeError, TypeError):
                    pass
```
:::


## Integration Example

::: {#9 .cell}
``` {.python .cell-code}
# Example of using both callbacks together
import tempfile

with tempfile.TemporaryDirectory() as tmp:
    io = ProjectIO(root=tmp, use_datestamp=False, auto_create=True)
    
    ckpt_cb = IOCheckpointCallback(io=io, run="exp1")
    log_cb = IOLogCallback(io=io, run="exp1")
    
    print(f"Checkpoint dir: {ckpt_cb.checkpoint_dir}")
    print(f"Log dir: {log_cb.log_dir}")
    print(f"Checkpoint path (epoch 5, step 1000): {ckpt_cb.get_checkpoint_path(5, 1000)}")
```
:::


::: {#10 .cell 0='h' 1='i' 2='d' 3='e'}
``` {.python .cell-code}
import nbdev; nbdev.nbdev_export()
```
:::


