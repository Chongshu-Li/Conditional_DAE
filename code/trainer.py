#  Copyright (C) 2022 Canon Medical Systems Corporation. All rights reserved
from datetime import datetime
from pathlib import Path
from typing import Optional
import math

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from utilities import move_to
from data_descriptor import DataDescriptor


class Trainer:

    def __init__(self, model, train_dataloader, val_dataloader,
                 optimiser, train_step, val_step, callback_dict, device, **kwargs):

        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.optimiser = optimiser
        self.train_step = train_step
        self.val_step = val_step
        self.callback_dict = callback_dict
        self.device = device
        self.additional_params = kwargs

        self.writer = None

        self.reset_state()


    def reset_state(self, params: dict = {}):
        self.state = {}
        self.state["callbacks"] = self.callback_dict
        self.state["train_it"] = 0
        self.state["epoch_no"] = 0
        self.state["writer"] = self.writer
        self.state.update(self.additional_params)

        self.state.update(params)


    def set_up_tensorboard(self):
        dt_string = datetime.now().strftime("%b%d__%H-%M-%S")
        prefix = self.state["identifier"]+"_" if "identifier" in self.state else ""
        logdir = Path(__file__).parent / "runs" / (prefix + dt_string)
        logdir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir=logdir)


    def callback(self, trigger_key: str):
        if trigger_key in self.state["callbacks"]:
            for c in self.state["callbacks"][trigger_key]:
                c(self)


    def train_epoch(self):
        self.model.train()
        self.callback("before_train_epoch")

        for batch in self.train_dataloader:

            self.state["epoch_no"] = self.state["train_it"] // self.state["epoch_len"] + 1
            if self.state["train_it"] % self.state["epoch_len"] == 0:
                print(f"\nEpoch [{self.state['epoch_no']}]:  ")

            self.state["train_batch"] = move_to(batch, self.device)
            self.state["train_it"] += 1
            self.callback("before_train_step")
            self.train_step(self)
            self.callback("after_train_step")

            if self.state["train_it"] % self.state["epoch_len"] == 0:
                # End of training epoch.

                self.callback("after_train_epoch")
                self.val_epoch()
                self.callback("after_epoch")
                self.model.train()

                self.state["progress"].update(1)
                print("\n")

            if self.state.get("max_train_iterations", float("inf")) <= self.state["train_it"]:
                print("Stopping due to max epochs condition.")
                self.state["STOP_TRAINING"] = True

                break

        return self


    def val_epoch(self, dataloader=None):
        """Runs validation and records average validation loss for early stopping."""
        self.callback("before_val_epoch")
        self.model.eval()

        dl = dataloader if dataloader is not None else self.val_dataloader
        total_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for batch in dl:
                self.state["val_batch"] = move_to(batch, self.device)
                self.callback("before_val_step")
                
                # Compute per-batch loss and store it in self.state["batch_loss"]
                self.val_step(self)
                self.callback("after_val_step")

                # Accumulate
                batch_loss = self.state.get("batch_loss")
                total_loss += batch_loss
                n_batches += 1

                if n_batches >= self.state["val_epoch_len"]:
                    break

        avg_val_loss = total_loss / n_batches
        # Record for early stopping
        self.state["val_loss"] = avg_val_loss
        # Log to TensorBoard
        if self.writer:
            self.writer.add_scalar("Val/Loss", avg_val_loss, self.state["epoch_no"])

        self.callback("after_val_epoch")
        return self

    def train(self, max_epochs=None, epoch_len=None, lr=None,
              accumulation_steps=1, val_epoch_len=None, patience=10):
        """Training loop with early stopping based on validation loss."""
        print(f"Starting training of {self.additional_params.get('identifier', 'model')}:")
        # Setup tensorboard writer once
        if not self.state.get("writer"):
            if not self.writer:
                self.set_up_tensorboard()
            self.state["writer"] = self.writer

        # State initialization
        self.state["accumulation_steps"] = accumulation_steps
        current_it = self.state.get("train_it", 0)
        self.state["epoch_len"] = len(self.train_dataloader) if epoch_len is None else epoch_len
        self.state["val_epoch_len"] = len(self.val_dataloader) if val_epoch_len is None else val_epoch_len
        if max_epochs:
            self.state["max_train_iterations"] = max_epochs * self.state["epoch_len"]
        if lr is not None:
            for pg in self.optimiser.param_groups:
                pg["lr"] = lr

        # Early stopping parameters
        self.state.setdefault("best_val_loss", float("inf"))
        self.state.setdefault("patience_cnt", 0)
        self.state["patience"] = patience

        self.callback("before_start")

        total_epochs = math.ceil(self.state.get("max_train_iterations", float("inf")) / self.state["epoch_len"])
        self.state["progress"] = tqdm(total=total_epochs, initial=current_it // self.state["epoch_len"])

        while True:
            try:
                self.callback("before_epoch")
                self.train_epoch()  # internally calls val_epoch()
            except KeyboardInterrupt:
                self.state["STOP_TRAINING"] = True

            # Early stopping check after validation
            val_loss = self.state.get("val_loss", None)
            if val_loss is not None:
                if val_loss < self.state["best_val_loss"]:
                    self.state["best_val_loss"] = val_loss
                    self.state["patience_cnt"] = 0
                    # Save the best model
                    self.save(name=f"best_{self.get_id()}")
                else:
                    self.state["patience_cnt"] += 1
                    if self.state["patience_cnt"] >= self.state["patience"]:
                        print(f"Early stopping: no improvement for {self.state['patience']} epochs.")
                        self.state["STOP_TRAINING"] = True

            # Exit conditions
            if self.state.get("STOP_TRAINING", False):
                del self.state["STOP_TRAINING"]
                break

            self.state["progress"].update(1)

        self.callback("after_end")
        self.state["progress"].close()
        return self


    def get_saveable_state(self):

        return {"epoch_no": self.state["epoch_no"],
                "train_it": self.state["train_it"],
                "model_state_dict": self.model.state_dict(),
                "model_class": self.model.__class__.__name__,
                "optimiser_state_dict": self.optimiser.state_dict()}


    def save(self, path=None, name=None, **kwargs):

        name = self.get_id() if name is None else name
        path = path if path is not None else Path(__file__).parent.parent / "saved_models" / f"{name}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)

        saveable =  self.get_saveable_state()
        saveable.update(kwargs)

        torch.save(saveable, path)


    def load(self, identifier=None, load_optimiser=True, exclude_keys=[], dir="saved_models"):

        if identifier is None:
            identifier = self.get_id()

        path = Path(__file__).parent.parent / dir / f"{identifier}.pt"

        if not path.exists():
            raise FileNotFoundError(f"Could not find the saved model named: {identifier}.pt in {path.parent.resolve()}")

        checkpoint = torch.load(path)
        for key in exclude_keys:
            if key in checkpoint["model_state_dict"]:
                del checkpoint["model_state_dict"][key]

        strict = False if exclude_keys else True
        self.model.load_state_dict(checkpoint["model_state_dict"], strict=strict)
        if load_optimiser:
            self.optimiser.load_state_dict(checkpoint["optimiser_state_dict"])
        if "epoch_no" in checkpoint:
            self.state["epoch_no"] = checkpoint["epoch_no"]
        if "train_it" in checkpoint:
            self.state["train_it"] = checkpoint["train_it"]
        return self


    def get_id(self):
        return self.additional_params.get("identifier")


    def set_data(self, data: Optional[DataDescriptor]):

        if isinstance(data, DataDescriptor):
            dd = data
        elif data is None:
            # Skip setting dataloaders for now
            return self
        else:
            raise ValueError(f"Unknown data type: {type}")

        self.train_dataloader = dd.get_dataloader("train")
        self.val_dataloader = dd.get_dataloader("val")
        return self



