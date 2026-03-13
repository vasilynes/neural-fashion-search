from transformers import AutoProcessor, AutoModelForZeroShotImageClassification
from peft import PeftModel
from src.config import config
import torch
import torch.nn.functional as F
import json

def embed_dataset(model, test_loader, processor, device):
    model.eval()
    all_image_embeds = []
    all_text_embeds = []
    with torch.no_grad():
        for batch in test_loader:
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

    return torch.cat(all_image_embeds, dim=0), torch.cat(all_text_embeds, dim=0)

def similarity(em1, em2):
    em1 = F.normalize(em1, dim=-1)
    em2 = F.normalize(em2, dim=-1)

    return em1 @ em2.T

def recall_at_k(similarity, k_values=[1, 5, 10]):
    num_queries = similarity.size(0)
    results = {}
    for k in k_values:
        top_k_indices = torch.topk(similarity, k, dim=1).indices
        correct = (top_k_indices == torch.arange(num_queries).unsqueeze(1)).any(dim=1).sum().item()
        results[f"R@{k}"] = correct / num_queries
    return results

def recall_at_k_filtered(similarity, correct_indices, k_values=[1, 5, 10]):
    results = {}
    for k in k_values:
        top_k_indices = torch.topk(similarity, k, dim=1).indices  # [M, k]
        correct = sum(
            correct_indices[i].item() in top_k_indices[i].tolist()
            for i in range(len(correct_indices))
        )
        results[f"R@{k}"] = correct / len(correct_indices)
    return results

def test_routine(test_loaders, test_params):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    processor = AutoProcessor.from_pretrained('patrickjohncyh/fashion-clip', use_fast=False)

    for checkpoint in test_params['checkpoints']:
        model = AutoModelForZeroShotImageClassification.from_pretrained('patrickjohncyh/fashion-clip')
        if checkpoint != 'baseline':
            model = PeftModel.from_pretrained(model, config.CHECKPOINT_DIR / checkpoint)
        model.to(device)

        full_loader = test_loaders['full']
        image_embeds, text_embeds = embed_dataset(model, full_loader, processor, device)
        sim = similarity(image_embeds, text_embeds)

        checkpoint_results = {
            'full': {
                'i2t': recall_at_k(sim),
                't2i': recall_at_k(sim.T)
            }
        }

        for loader_name, test_loader in list(test_loaders.items())[1:]:
            indices = torch.tensor(test_loader.dataset.indices)
            filtered_text_embeds = text_embeds[indices] 
            
            filtered_sim = similarity(filtered_text_embeds, image_embeds)
            
            # for t2i: each text query should retrieve its matching image
            # but correct index in full image set is indices[i], not i
            checkpoint_results[loader_name] = {
                't2i': recall_at_k_filtered(filtered_sim, indices)
            }

        output_path = config.METRICS_DIR / f"{checkpoint}_recalls.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(checkpoint_results, f, indent=2, default=lambda x: float(x))

