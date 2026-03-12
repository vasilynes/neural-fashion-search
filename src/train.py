from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from peft import get_peft_model, LoraConfig, PeftModel
import torch
from torch import optim
from pathlib import Path
from faiss import IndexFlatIP
import torch.nn.functional as F
import logging
import sys

def setup_logger(name, log_file, level=logging.INFO):  
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level) 
    
    if not logger.handlers:
        fh = logging.FileHandler(log_file)
        fh.setLevel(level) 
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)
    
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))  
        logger.addHandler(ch)
    
    return logger

logger = setup_logger("training", "logs/training.log")

class Trainer:
    def __init__(
            self, 
            model, 
            processor, 
            device, 
            experiment_name, 
            experiment_params, 
            sampler,
            training_state,
            train_loader,
            embed_loader, 
            val_loader,
            logger,
        ):
        self.model = model
        self.processor = processor
        self.device = device
        self.experiment_name = experiment_name
        self.experiment_params = experiment_params
        self.sampler = sampler
        self.training_state = training_state
        self.train_loader = train_loader
        self.embed_loader = embed_loader
        self.val_loader = val_loader
        self.logger = logger

    def get_scheduler(self, total_steps, warmup_steps): 
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer=self.optimizer,
            start_factor=1e-8,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=total_steps - warmup_steps
        )
        scheduler = optim.lr_scheduler.SequentialLR(
            optimizer=self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        return scheduler

    def load_state(self):
        self.optimizer.load_state_dict(self.training_state['optimizer_state_dict'])
        self.scheduler.load_state_dict(self.training_state['scheduler_state_dict'])
        self.best_val_loss = self.training_state['best_val_loss']
        self.start_epoch = self.training_state['epoch'] + 1

    def save_state(self, epoch):
        checkpoint_dir = Path(f"checkpoints/{self.experiment_name}_best")
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.model.save_pretrained(f"checkpoints/{self.experiment_name}_best")
                
        torch.save({
            'epoch': epoch,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
        }, checkpoint_dir / 'training_state.pt')

    def embed_dataset(self):
        self.model.eval()
        all_image_embeds = []
        all_text_embeds = []
        with torch.no_grad():
            for batch in self.embed_loader:
                inputs = self.processor(
                    text=batch['caption'],
                    images=batch['image'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(self.device) 
                
                outputs = self.model(**inputs)

                all_image_embeds.append(outputs.image_embeds.cpu())
                all_text_embeds.append(outputs.text_embeds.cpu())

        return (
            torch.cat(all_image_embeds, dim=0),  
            torch.cat(all_text_embeds, dim=0),  
        )
        
    def mine_hard_negatives(self, image_embeds, text_embeds, top_k=10):
        image_embeds = F.normalize(image_embeds, dim=-1).numpy()
        text_embeds = F.normalize(text_embeds, dim=-1).numpy()

        n = image_embeds.shape[0]
        d = image_embeds.shape[1]

        index = IndexFlatIP(d)
        index.add(text_embeds)

        _, indices = index.search(image_embeds, top_k + 1)

        hard_negatives = {}
        for i in range(n):
            neighbors = [idx for idx in indices[i] if idx != i][:top_k]
            hard_negatives[i] = neighbors

        return hard_negatives
    
    def set_hard_negatives(self):
        image_embeds, text_embeds = self.embed_dataset()
        hard_negatives = self.mine_hard_negatives(image_embeds, text_embeds, top_k=self.experiment_params['top_k'])
        self.sampler.update_hard_negatives(hard_negatives)
    
    def get_val_loss(self):
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                val_loss += self.get_batch_loss(batch).item()
        return val_loss / len(self.val_loader)
    
    def get_batch_loss(self, batch):
        inputs = self.processor(
            text=batch['caption'],
            images=batch['image'],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=77,
        ).to(self.device)
        outputs = self.model(**inputs, return_loss=True)
        return outputs.loss

    def run(self):
        params_to_update = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = optim.AdamW(params_to_update, self.experiment_params['lr'])

        accumulation_steps = self.experiment_params['accumulation_steps']
        steps_per_epoch = (len(self.train_loader) + accumulation_steps - 1) // accumulation_steps
        total_steps = self.experiment_params['epochs'] * steps_per_epoch
        warmup_steps = int(0.1 * total_steps) 

        self.scheduler = self.get_scheduler(total_steps, warmup_steps)

        if self.training_state:
            self.load_state()
        else: 
            self.best_val_loss = float('inf')
            self.start_epoch = 0

        patience = self.experiment_params['patience']
        patience_count = 0
        for epoch in range(self.start_epoch, self.experiment_params['epochs']):
            self.model.eval()
            self.set_hard_negatives()

            self.model.train()
            self.optimizer.zero_grad()
            
            train_loss = 0.0
            for batch_idx, batch in enumerate(self.train_loader):
                loss = self.get_batch_loss(batch)
                train_loss += loss.item()
                loss /= accumulation_steps
                loss.backward()

                if (batch_idx + 1) % accumulation_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step()            

            if len(self.train_loader) % accumulation_steps != 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.scheduler.step()

            avg_train_loss = train_loss / len(self.train_loader)
            
            self.model.eval()
            avg_val_loss = self.get_val_loss()

            self.logger.info(f"Epoch {epoch+1}/{self.experiment_params['epochs']} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")

            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                patience_count = 0
                self.save_state(epoch)
            else:
                patience_count += 1
                if patience_count >= patience:
                    self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                    return

def inject_lora(model, lora_params):
    target_modules = [
        f"vision_model.encoder.layers.{i}.self_attn.{proj}"
        for i in lora_params['vision_layers'] for proj in lora_params['projections']
    ] + [
        f"text_model.encoder.layers.{i}.self_attn.{proj}"
        for i in lora_params['text_layers'] for proj in lora_params['projections']
    ]

    logger.info(f"Target modules ({len(target_modules)}):")
    for m in target_modules:
        logger.info(f"  {m}")

    lora_config = LoraConfig(
        r=lora_params['r'], 
        lora_alpha=lora_params['lora_alpha'], 
        target_modules=target_modules,
        lora_dropout=lora_params['dropout'],
        bias='none'
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model 

def train_routine(train_loader, embed_loader, val_loader, train_params, sampler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip", use_fast=False)

    for experiment_name, lora_params in train_params['lora'].items():
        model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")

        if lora_params['resume']:
            logger.info(f"\nResuming experiment: {experiment_name}...")

            checkpoint_dir = Path(f"checkpoints/{experiment_name}_best")
            if not checkpoint_dir.exists():
                logger.warning(f"Experiment {experiment_name} has nothing to resume from")
                continue
    
            model = PeftModel.from_pretrained(model, checkpoint_dir)
            training_state = torch.load(checkpoint_dir / 'training_state.pt', map_location=device)
        else: 
            logger.info(f"\nStarting experiment: {experiment_name}...")
            model = inject_lora(model, lora_params)
            training_state = None

        model = model.to(device)

        trainer = Trainer(
            model,
            processor, 
            device,
            experiment_name,
            train_params['training'], 
            sampler, 
            training_state,
            train_loader,
            embed_loader, 
            val_loader,
            logger
        )

        trainer.run()

        logger.info(f"\nExperiment: {experiment_name} is finished")
