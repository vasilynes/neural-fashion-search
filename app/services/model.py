import torch 
import torch.nn.functional as F
from qdrant_client.models import SparseVector
from peft import PeftModel
from tqdm.notebook import tqdm

class ModelService:
    def __init__(self, processor, device, dense_model: PeftModel, sparse_model):
        self.processor = processor
        self.device = device
        self.dense_model = dense_model
        self.sparse_model = sparse_model

    def embed_text(self, batch):
        inputs = self.processor(
            text=batch,
            truncation=True,
            padding=True, 
            return_tensors='pt',
            max_length=77
        ).to(self.dense_model.device)

        with torch.no_grad():
            # Manually replicating get_text_features from modeling_clip.py
            # as calling it directly on PeftModel object breaks
            text_outputs = self.dense_model.base_model.model.text_model(**inputs) # Use base model's text encoder to obtain BaseModelOutputWithPooling
            text_embeds = text_outputs.pooler_output
            text_embeds = self.dense_model.base_model.model.text_projection(text_embeds)  # Project pooled output into the common space

        return F.normalize(text_embeds, dim=-1).cpu().numpy()

    def embed_image(self, image):
        inputs = self.processor(
            images=[image],
            return_tensors='pt',
        ).to(self.dense_model.device)

        with torch.no_grad():
            # Manually replicating get_image_features from modeling_clip.py
            # as calling it directly on PeftModel object breaks
            vision_outputs = self.dense_model.base_model.model.vision_model(**inputs) # Use base model's vision encoder to obtain BaseModelOutputWithPooling
            image_embeds = vision_outputs.pooler_output
            image_embeds = self.dense_model.base_model.model.visual_projection(image_embeds)  # Project pooled output into the common space

        return F.normalize(image_embeds, dim=-1).cpu().numpy()

    def embed_text_sparse(self, queries, batch_size=32):
        embeddings = self.sparse_model.embed(queries, batch_size=batch_size)
        return [
            SparseVector(
                indices=embeds.indices.tolist(),
                values=embeds.values.tolist()
            ) for embeds in embeddings
        ]

    def embed_multimodal(self, image, text, beta=0.6):
        inputs = self.processor(
            text=[text],
            images=[image],
            truncation=True,
            padding=True,
            return_tensors='pt',
            max_length=77
        ).to(self.device)

        with torch.no_grad():
            outputs = self.dense_model(**inputs)
            image_embed = outputs.image_embeds
            text_embed = outputs.text_embeds

        image_embed = F.normalize(image_embed, dim=-1)
        text_embed = F.normalize(text_embed, dim=-1)

        query_embed = (beta * image_embed) + ((1.0 - beta) * text_embed)

        return F.normalize(query_embed, dim=-1).cpu().numpy()
