from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from peft import get_peft_model, LoraConfig
import torch
from torch import optim
from pathlib import Path
from faiss import IndexFlatIP
import torch.nn.functional as F

def inject_lora(model, lora_params):
    target_modules = [
        f"vision_model.encoder.layers.{i}.self_attn.{proj}"
        for i in lora_params['vision_layers'] for proj in lora_params['projections']
    ] + [
        f"text_model.encoder.layers.{i}.self_attn.{proj}"
        for i in lora_params['text_layers'] for proj in lora_params['projections']
    ]

    print(f"Target modules ({len(target_modules)}):")
    for m in target_modules:
        print(f"  {m}")

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

def get_scheduler(optimizer, total_steps):
    warmup_steps = int(0.1 * total_steps)
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=1e-8,
        end_factor=1.0,
        total_iters=warmup_steps
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=total_steps - warmup_steps
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )
    return scheduler

def embed_dataset(model, processor, embed_loader, device):
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    with torch.no_grad():
        for batch in embed_loader:
            inputs = processor(
                text=batch['caption'],
                images=batch['image'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)  

            outputs = model(**inputs)
            all_image_embeds.append(outputs.image_embeds.cpu())
            all_text_embeds.append(outputs.text_embeds.cpu())

    return (
        torch.cat(all_image_embeds, dim=0),  
        torch.cat(all_text_embeds, dim=0),  
    )

def mine_hard_negatives(image_embeds, text_embeds, top_k=10):
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

def set_hard_negatives(model, processor, embed_loader, device, experiment_params, sampler):
    image_embeds, text_embeds = embed_dataset(model, processor, embed_loader, device)
    hard_negatives = mine_hard_negatives(image_embeds, text_embeds, top_k=experiment_params['top_k'])
    sampler.update_hard_negatives(hard_negatives)

def run_experiment(model, processor, train_loader, embed_loader, val_loader, experiment_params, experiment_name, device, sampler, training_state=None):
    params_to_update = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.AdamW(params_to_update, experiment_params['lr'])

    total_steps = experiment_params['epochs'] * len(train_loader)   
    scheduler = get_scheduler(optimizer, total_steps)

    if training_state:
        optimizer.load_state_dict(training_state['optimizer_state_dict'])
        scheduler.load_state_dict(training_state['scheduler_state_dict'])
        best_val_loss = training_state['best_val_loss']
        patience_count = training_state['patience_count']
        start_epoch = training_state['epoch'] + 1
    else: 
        best_val_loss = float('inf')
        patience_count = 0
        start_epoch = 0

    accumulation_steps = experiment_params['accumulation_steps']
    patience = experiment_params['patience']

    for epoch in range(start_epoch, experiment_params['epochs']):
        model.eval()
        set_hard_negatives(model, processor, embed_loader, device, experiment_params, sampler)

        model.train()
        optimizer.zero_grad()
        train_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            inputs = processor(
                text=batch['caption'],
                images=batch['image'],
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=77,
            ).to(device)
            outputs = model(**inputs, return_loss=True)
            loss = outputs.loss / accumulation_steps
            loss.backward()
            train_loss += loss.item()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()            

        if (batch_idx + 1) % accumulation_steps != 0:
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
        avg_train_loss = train_loss / len(train_loader)

        model.eval()
        
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                inputs = processor(
                    text=batch['caption'],
                    images=batch['image'],
                    return_tensors='pt',
                    padding=True,
                    truncation=True,
                    max_length=77,
                ).to(device)
                outputs = model(**inputs, return_loss=True)
                val_loss += outputs.loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{experiment_params['epochs']} | train loss: {avg_train_loss:.4f} | val loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_count = 0

            checkpoint_dir = Path(f"checkpoints/{experiment_name}_best")
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            model.save_pretrained(f"checkpoints/{experiment_name}_best")
            
            torch.save({
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'patience_count': patience_count,
            }, checkpoint_dir / 'training_state.pt')
        else:
            patience_count += 1
            if patience_count >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
def train_routine(train_loader, embed_loader, val_loader, train_params, sampler):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained("patrickjohncyh/fashion-clip", use_fast=False)

    for experiment_name, lora_params in train_params['lora'].items():
        if lora_params['resume']:
            print(f"\nResuming experiment: {experiment_name}...")

            checkpoint_dir = Path(f"checkpoints/{experiment_name}_best")
            if not checkpoint_dir.exists():
                print(f"Experiment {experiment_name} has nothing to resume from")
                continue
    
            model = AutoModelForZeroShotImageClassification.from_pretrained(checkpoint_dir)
            training_state = torch.load(checkpoint_dir / 'training_state.pt')
        else: 
            print(f"\nStarting experiment: {experiment_name}...")

            model = AutoModelForZeroShotImageClassification.from_pretrained("patrickjohncyh/fashion-clip")
            model = inject_lora(model, lora_params)
            training_state = None

        model = model.to(device)

        run_experiment(model, processor, train_loader, embed_loader, val_loader, train_params['training'], experiment_name, device, sampler, training_state)


    