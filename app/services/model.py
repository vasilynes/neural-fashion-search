import torch 
import torch.nn.functional as F
from qdrant_client.models import SparseVector
from peft import PeftModel

class ModelService:
    def __init__(self, processor, device, dense_model: PeftModel, sparse_model):
        self.processor = processor
        self.device = device
        self.dense_model = dense_model
        self.sparse_model = sparse_model

    def embed_text(self, text):
        inputs = self.processor(
            text=[text],
            truncation=True,
            padding=True, 
            return_tensors='pt',
            max_length=77
        ).to(self.device)

        with torch.no_grad():
            # Manually replicating get_text_features from modeling_clip.py
            # as calling it directly on PeftModel object breaks
            text_outputs = self.dense_model.base_model.model.text_model(**inputs) # Use base model's text encoder to obtain BaseModelOutputWithPooling
            text_embeds = text_outputs.pooler_output
            text_embeds = self.dense_model.base_model.model.text_projection(text_embeds)  # Project pooled output into the common space

        return F.normalize(text_embeds, dim=-1).cpu().numpy()[0]

    def embed_image(self, image):
        inputs = self.processor(
            images=[image],
            return_tensors='pt',
        ).to(self.device)

        with torch.no_grad():
            # Manually replicating get_image_features from modeling_clip.py
            # as calling it directly on PeftModel object breaks
            vision_outputs = self.dense_model.base_model.model.vision_model(**inputs) # Use base model's vision encoder to obtain BaseModelOutputWithPooling
            image_embeds = vision_outputs.pooler_output
            image_embeds = self.dense_model.base_model.model.visual_projection(image_embeds)  # Project pooled output into the common space

        return F.normalize(image_embeds, dim=-1).cpu().numpy()[0]

    def embed_text_sparse(self, text):
        embeds = list(self.sparse_model.embed([text]))[0]
        return SparseVector(
            indices=embeds.indices.tolist(),
            values=embeds.values.tolist()
        )