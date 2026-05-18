## Neural Fashion Search Engine
This project adapts the [FashionCLIP model](https://huggingface.co/patrickjohncyh/fashion-clip) to the [H&M Personalized Fashion Recommendations dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=images) (100+ SKUs).

The API of the adapted model is exposed via the search interface frontend, the user can search items by text, images or combined.
This creates a domain-adapted, multimodal search architecture.

### Architecture
1. ML Models:
   * Dense: the FashionCLIP model (ViT-B/32), adapted through rank-8 LoRA in 4 vision layers and 6 text layers for semantic text-image bridging
   * Sparse: the SPLADE model for exact text matching and vocabulary expansion
2. Database: Qdrant for parallel dense/sparse indexing and native Reciprocal Rank Fusion (RRF)
3. Backend: FastAPI
4. Frontend: React + TailwindCSS

The Qdrant database is utilized to store embeddings: dense text/image embeddings of the adapted FashionCLIP model encoders and sparse text embeddings of the SPLADE model. 
The items are then retrieved via ANN search. 

If the user searches by text, the query is embedded via the adapted FashionCLIP text encoder and via SPLADE sparse encoder. 
The database is searched by both, the results are fused via RFF or direct weighting with some adjustable weight (default: 0.25) on dense embeddings.

If they search by image, only dense FashionCLIP embeddings are used. 

If the user searches in the multimodal mode, FashionCLIP encoders are used to embed the text and the image simultaneously. 
Then latent space arithmetics is then used to put an adjustable weight (default: 0.5) on the image embedding:

`query_embed = (beta * image_embed) + ((1.0 - beta) * text_embed)`.

The resulting query is used to search the database, that is, the multimodal search is always dense, SPLADE model is not used.
### Exploratory Data Analysis
