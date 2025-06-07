# dso_hunyuan_lora.py
"""
A complete, multi-stage framework for DSO fine-tuning with support for
Parameter-Efficient Fine-Tuning using LoRA.

---
Prerequisites:
---
pip install torch accelerate transformers omegaconf safetensors Pillow tqdm
pip install trimesh
pip install -U hy3dgen
pip install peft

---
Workflow:
---
The workflow remains the same three stages. The --mode finetune stage
is now enhanced with LoRA capabilities, configurable in DSOConfig.

1. --mode generate
   Example:
     python dso_hunyuan_lora.py --mode generate \
       --image_paths "path/to/your/images/*.png" \
       --output_dir "dso_dataset"

2. --mode simulate
   Example:
     python dso_hunyuan_lora.py --mode simulate --data_dir "dso_dataset"

3. --mode finetune
   Example (with LoRA enabled by default):
     accelerate launch dso_hunyuan_lora.py --mode finetune --data_dir "dso_dataset"
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
from accelerate.logging import get_logger
from tqdm import tqdm
from copy import deepcopy
import abc
import os
import json
from PIL import Image
import argparse
from pathlib import Path
import glob
import numpy as np
import trimesh
from hy3dgen.shapegen.pipelines import Hunyuan3DDitPipeline
from peft import LoraConfig, get_peft_model

logger = get_logger(__name__)


# --- Configuration and Abstract Base Classes (ABCs) ---

class DSOConfig:
    """Configuration class for all hyperparameters."""
    # Training
    max_train_steps: int = 10000
    learning_rate: float = 1e-4  # LoRA can often handle a higher learning rate
    batch_size: int = 4
    dataloader_num_workers: int = 4
    gradient_accumulation_steps: int = 2
    
    # DSO Specific
    beta: float = 1.0
    
    # LoRA Configuration
    use_lora: bool = True
    lora_r: int = 16  # The rank of the LoRA matrices.
    lora_alpha: int = 16 # The scaling factor.
    lora_dropout: float = 0.1
    # Target modules for LoRA. These are the names of the layers to adapt.
    # For DiT models, targeting the attention projections is standard.
    lora_target_modules: list = ["to_q", "to_k", "to_v", "to_out.0"]
    
    # Logging
    log_interval: int = 10


class BaseLikelihoodCalculator(abc.ABC):
    @abc.abstractmethod
    def compute_log_prob(self, model: nn.Module, batch: dict) -> torch.Tensor:
        pass


# --- Concrete Implementations for Hunyuan3D ---

class StabilitySimulator:
    def score(self, meshes: list) -> torch.Tensor:
        scores = []
        for mesh in meshes:
            if not isinstance(mesh, trimesh.Trimesh) or not mesh.vertices.shape[0] > 0:
                scores.append(-100.0)
                continue
            bounds = mesh.bounds
            center_z = (bounds[0, 2] + bounds[1, 2]) / 2.0
            score = -center_z
            scores.append(score)
        return torch.tensor(scores, dtype=torch.float32)


class Hunyuan3DDiTLikelihoodCalculator(BaseLikelihoodCalculator):
    def __init__(self, pipeline: Hunyuan3DDitPipeline):
        self.scheduler = pipeline.scheduler

    def compute_log_prob(self, model: nn.Module, batch: dict) -> torch.Tensor:
        latents_x0 = batch['x0']
        prompt_embeds = batch['cond']
        batch_size = latents_x0.shape[0]
        noise = torch.randn_like(latents_x0)
        timesteps = torch.randint(0, self.scheduler.config.num_train_timesteps, (batch_size,), device=latents_x0.device)
        noisy_latents = self.scheduler.add_noise(latents_x0, noise, timesteps)
        predicted_noise = model(noisy_latents, timesteps, prompt_embeds).sample
        loss_per_item = F.mse_loss(predicted_noise, noise, reduction='none').mean(dim=tuple(range(1, predicted_noise.ndim)))
        return -loss_per_item


# --- Stage 1: Generation ---
def run_generation(args):
    logger.info(f"Starting Stage 1: Generation. Outputting to {args.output_dir}")
    accelerator = Accelerator()
    pipeline = Hunyuan3DDitPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    pipeline = pipeline.to(accelerator.device)
    image_paths = sorted(glob.glob(args.image_paths))
    if not image_paths:
        raise FileNotFoundError(f"No images found matching the pattern: {args.image_paths}")
    logger.info(f"Found {len(image_paths)} source images.")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), args.batch_size), disable=not accelerator.is_main_process):
            batch_paths = image_paths[i:i + args.batch_size]
            batch_images = [Image.open(p).convert("RGB") for p in batch_paths]
            output = pipeline(image=batch_images, num_inference_steps=50)

            for j, image_path in enumerate(batch_paths):
                sample_id = f"{Path(image_path).stem}_{i + j}"
                sample_dir = output_dir / sample_id
                sample_dir.mkdir(exist_ok=True)
                torch.save(output.latents[j].cpu(), sample_dir / "latent.pt")
                torch.save(output.prompt_embeds[j].cpu(), sample_dir / "condition.pt")
                with open(sample_dir / "metadata.json", 'w') as f:
                    json.dump({'source_image': str(image_path)}, f)
                mesh = output.meshes[j]
                mesh.export(sample_dir / "mesh.obj")


# --- Stage 2: Simulation ---
def run_simulation(args):
    logger.info(f"Starting Stage 2: Simulation on data in {args.data_dir}")
    simulator = StabilitySimulator()
    sample_dirs = [p for p in Path(args.data_dir).iterdir() if p.is_dir()]
    logger.info(f"Found {len(sample_dirs)} generated samples to simulate.")

    for sample_dir in tqdm(sample_dirs):
        sim_result_path = sample_dir / "simulation.json"
        if sim_result_path.exists():
            continue
        mesh_path = sample_dir / "mesh.obj"
        if not mesh_path.exists():
            continue
        try:
            mesh = trimesh.load_mesh(mesh_path)
            reward = simulator.score([mesh])[0].item()
        except Exception as e:
            logger.warning(f"Failed to load or simulate mesh {mesh_path}: {e}")
            reward = -100.0
        with open(sim_result_path, 'w') as f:
            json.dump({'reward': reward}, f)


# --- Stage 3: Fine-tuning ---
class PrecomputedDSODataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.samples = []
        logger.info("Scanning for valid samples...")
        for sample_dir in tqdm(self.data_dir.iterdir()):
            if (sample_dir.is_dir() and
                (sample_dir / "latent.pt").exists() and
                (sample_dir / "condition.pt").exists() and
                (sample_dir / "simulation.json").exists()):
                self.samples.append(sample_dir)
        logger.info(f"Found {len(self.samples)} valid samples for training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_path = self.samples[idx]
        latent = torch.load(sample_path / "latent.pt")
        condition = torch.load(sample_path / "condition.pt")
        with open(sample_path / "simulation.json", 'r') as f:
            reward = json.load(f)['reward']
        return {'x0': latent, 'cond': condition.squeeze(0), 'reward': torch.tensor(reward, dtype=torch.float32)}

class OfflineDSOFinetuner:
    def __init__(self, config: DSOConfig, model: nn.Module, likelihood_calculator: BaseLikelihoodCalculator, train_loader: DataLoader):
        self.config = config
        self.model = model
        self.likelihood_calculator = likelihood_calculator
        self.train_loader = train_loader
        self.accelerator = Accelerator(
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            log_with="tensorboard",
            project_dir="dso_runs"
        )
        with self.accelerator.main_process_first():
            self.ref_model = deepcopy(self.model)
            for p in self.ref_model.parameters():
                p.requires_grad = False
            self.ref_model.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.learning_rate)
        self.model, self.optimizer, self.ref_model, self.train_loader = self.accelerator.prepare(
            self.model, self.optimizer, self.ref_model, self.train_loader
        )

    def compute_dso_loss(self, batch: dict) -> torch.Tensor:
        batch['t'] = torch.rand(batch['x0'].shape[0], device=self.accelerator.device)
        batch['eps'] = torch.randn_like(batch['x0'])
        log_probs_model = self.likelihood_calculator.compute_log_prob(self.model, batch)
        with torch.no_grad():
            log_probs_ref = self.likelihood_calculator.compute_log_prob(self.ref_model, batch)
        log_ratio = log_probs_model - log_probs_ref
        rewards = batch['rewards']
        loss = (log_ratio - self.config.beta * rewards).pow(2).mean()
        return loss

    def finetune(self):
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers("dso_lora_project")
        progress_bar = tqdm(range(self.config.max_train_steps), disable=not self.accelerator.is_local_main_process)
        global_step = 0
        self.model.train()

        while global_step < self.config.max_train_steps:
            for batch in self.train_loader:
                if global_step >= self.config.max_train_steps:
                    break
                with self.accelerator.accumulate(self.model):
                    loss = self.compute_dso_loss(batch)
                    self.accelerator.backward(loss)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                if self.accelerator.sync_gradients:
                    progress_bar.update(1)
                    global_step += 1
                    if global_step % self.config.log_interval == 0:
                        metrics = {
                            'loss': self.accelerator.gather(loss).mean().item(),
                            'reward': self.accelerator.gather(batch['rewards']).mean().item()
                        }
                        logger.info(f"Step {global_step}: Loss = {metrics['loss']:.4f}, Avg Reward = {metrics['reward']:.4f}")
                        if self.accelerator.is_main_process:
                            self.accelerator.log(metrics, step=global_step)
        self.accelerator.end_training()
        logger.info("Fine-tuning complete.")

def run_finetuning(args):
    logger.info(f"Starting Stage 3: Fine-tuning on data from {args.data_dir}")
    config = DSOConfig()
    pipeline = Hunyuan3DDitPipeline.from_pretrained(args.model_path, torch_dtype=torch.float16)
    model_to_finetune = pipeline.dit

    # --- NEW: Apply LoRA if configured ---
    if config.use_lora:
        logger.info("Applying LoRA to the model...")
        lora_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            target_modules=config.lora_target_modules,
            # For DiT, it's common to only adapt attention layers.
            # You can find module names by printing `model_to_finetune.named_modules()`
        )
        model_to_finetune = get_peft_model(model_to_finetune, lora_config)
        logger.info("LoRA applied successfully.")
        model_to_finetune.print_trainable_parameters()

    dataset = PrecomputedDSODataset(args.data_dir)
    if len(dataset) == 0:
        raise ValueError("No valid training samples found.")
    data_loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.dataloader_num_workers
    )
    likelihood_calculator = Hunyuan3DDiTLikelihoodCalculator(pipeline)
    tuner = OfflineDSOFinetuner(
        config=config, model=model_to_finetune, likelihood_calculator=likelihood_calculator, train_loader=data_loader
    )
    tuner.finetune()

# --- Main Script Dispatcher ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-stage DSO Fine-tuning with LoRA.")
    parser.add_argument("--mode", type=str, required=True, choices=['generate', 'simulate', 'finetune'])
    parser.add_argument("--image_paths", type=str, help="Path pattern to source images (e.g., 'assets/*.png').")
    parser.add_argument("--output_dir", type=str, help="Directory to save generated data.")
    parser.add_argument("--data_dir", type=str, help="Directory with pre-computed data.")
    parser.add_argument("--model_path", type=str, default="Tencent-Hunyuan/Hunyuan3D-2", help="Path to pretrained model.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size for data generation.")
    args = parser.parse_args()

    if args.mode == 'generate':
        if not all([args.image_paths, args.output_dir]):
            parser.error("--mode generate requires --image_paths and --output_dir")
        run_generation(args)
    elif args.mode == 'simulate':
        if not args.data_dir:
            parser.error("--mode simulate requires --data_dir")
        run_simulation(args)
    elif args.mode == 'finetune':
        if not args.data_dir:
            parser.error("--mode finetune requires --data_dir")
        run_finetuning(args)