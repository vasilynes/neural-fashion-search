## Neural Fashion Search Engine
This project adapts the [FashionCLIP model](https://huggingface.co/patrickjohncyh/fashion-clip) to the [H&M Personalized Fashion Recommendations dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=images) (100+ SKUs).

The API of the adapted model is exposed via the search interface frontend, the user can search items by text, images or combined.
This creates a domain-adapted, multimodal search architecture.

### TL;DR
* Domain adaptation works: fine-tuning FashionCLIP on the H&M dataset improved strict recall (R@1) by 45%.
* Modality gap: pure dense search suffers from weak signal on short user queries (hubness). It requires a sparse retriever (SPLADE) to handle exact keyword matching and typos.
* Optimality: An asymmetric hybrid search (`alpha=.25` dense weight) maximizes performance for real-world e-commerce according to the tests.

### Demonstrations
#### Modifiers
##### Strictness
<img width="1000" height="563" alt="crew_neck_shirt" src="https://github.com/user-attachments/assets/048c761f-c6ec-493f-967d-bd43e1963847" />

Dense search is sensitive to modifiers. When using SPLADE with weight 0.75 (`alpha=.25`), "crew-neck shirt" query returns some v-neck shirts. Putting more weight on dense search resolves this and the hybrid search returns crew-neck shirts only.

##### Vocabulary gap
<img width="1000" height="563" alt="goth_outfit" src="https://github.com/user-attachments/assets/4179fda0-8151-4f68-81b8-76e58757c537" />

Abstract user queries may hurt sparse search, since those exact words may not exist in the H&M catalog. Searching "goth outfit" returns generic outfits, while putting more weight on dense search allows to map abstract human concepts ("goth") to visual reality.

<img width="1000" height="563" alt="winter_dress" src="https://github.com/user-attachments/assets/629b0bda-f81b-4a2e-824c-c12fc61f0c13" />

SPLADE fails to suggest dresses appropriate for the winter, but search results with a higher weight on the dense component contain knitted dresses.

#### Multimodality
<img width="1000" height="563" alt="black_leather_floral_dress" src="https://github.com/user-attachments/assets/30d72889-5f58-4271-8091-4adc2480457f" />

The floral dress continuously modified by the "black leather" specification. Image concept arithmetic allows to evaluate complex queries by fusing image and text vectors in the latent space `(β * Image + (1-β) * Text)`. Users can visually search for items while applying text modifiers.

<img width="1000" height="563" alt="floral_leather_boots" src="https://github.com/user-attachments/assets/6c08c2e9-b7bb-42d9-a5e0-e03150e3494c" />

In turn, modifying leather boots with the "floral" text allows to retrieve leather boots with floral pattern on them. 

### Architecture
1. ML Models:
   * Dense: the FashionCLIP model (ViT-B/32), adapted through rank-8 LoRA in 4 vision layers and 6 text layers and contrastive loss for semantic text-image bridging
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
For details on data cleaning & analysis, see:
* `ml/notebooks/01_dataset_cleaning.ipynb`
* `ml/notebooks/02_text_analysis.ipynb`
* `ml/notebooks/03_categorical_analysis.ipynb`
### Domain Adaptation
Adapted FashionCLIP demonstrates increase in its recall (i2t = image-to-text, t2i = text-to-image):
<table>
  <tr>
    <th></th>
    <th colspan="2">baseline</th>
    <th colspan="2">adapted</th>
  </tr>
  <tr>
    <th></th>
    <th>i2t</th>
    <th>t2i</th>
    <th>i2t</th>
    <th>t2i</th>
  </tr>
  <tr>
    <td>R@1</td>
    <td>0.26</td>
    <td>0.25</td>
    <td>0.38</td>
    <td>0.37</td>
  </tr>
  <tr>
    <td>R@5</td>
    <td>0.52</td>
    <td>0.50</td>
    <td>0.70</td>
    <td>0.68</td>
  </tr>
  <tr>
    <td>R@10</td>
    <td>0.65</td>
    <td>0.61</td>
    <td>0.81</td>
    <td>0.80</td>
  </tr>
</table>

Domain adaptation demonstrates symmetric (both i2t and t2i) improvement across every metric. 

The strictest R@1 improved by 44-46%. 

R@10 for both directions of inference is ~0.80, meaning 20% of queries failed to find the correct term within top 10 objects.
### Ablation, HitRate@10 and Mean Reciprocal Rank
The search engine is based on combining dense and sparse embeddings. To evaluate it fully, LLM (Qwen2.5-3B-Instruct) was used on 10 000 unique descriptions to generate synthetic user queries of the following types:

1. Synonyms, where the modifiers are altered to their synonyms
2. Typos, where the descriptions are polluted
3. User queries, which mimic real user queries
4. Attribute swaps, which swap key modifiers to their opposites or disjunctive variations

The first two measure search resilience.

The third measures how well the search tolerates the distribution shift from full descriptions to short user queries with specific vocabulary.

The fourth tests how well the search distinguishes between modifiers.

HR@10 is a portion of queries that resulted in retrieving at least one item from the set of objects that are described by the string, from which the query was derived. MRR measures the average reciprocal index of the first occurrence of such an object.

For attribute swaps, the HR@10 logic is inverted: a "hit" is scored only if the engine successfully avoids retrieving the original item. Therefore, a higher HR@10 indicates the model successfully respected the swapped modifier.

Both metrics are computed using weighted average fusion with `alphas = [.0, .25, .5, .75, 1]`, where `alpha` is the weight put on the dense component, and Max-Scaling normalization. For more details, see `ml/notebooks/scripts/metrics_by_alpha.py` script.

The metrics are then recomputed using RRF, which formally corresponds to alpha = .5 (both models are allowed to contribute equally), but retrieves results differently.

<img width="1640" height="600" alt="Screenshot_20260519_043258" src="https://github.com/user-attachments/assets/eb67dc0d-7832-4fb3-8807-8a4ba2d12f07" />

* SPLADE surpasses the dense model on Typos and Real User queries  due to its learned vocabulary expansion.
* Dense model surpasses SPLADE on Attribute swaps. Hypothetically, this demonstrates the power of contrastive training: the dense model is anchored by visuals and maps modifiers apart, avoiding SPLADE's partial-word-match mistake.
* RRF excels at Synonyms, beating the ceiling of either model alone.

Except Synonyms, other types are slightly worse off from RRF and are better off from setting `alpha=.25` or near it, leading to asymmetric hybrid search, where lexical expansion does the heavy lifting. 

However, since the data is fully synthetic, the study doesn't demonstrate the full distribution dynamics, so that further explorations are needed.
### Text failure analysis
Here top 10 returned products are investigated.
#### Hubness
Bra type is one of the top-20 product types. Searching "bra" using RRF returned only 5 bras, failures are:

<img width="1058" height="245" alt="Screenshot_20260519_045602" src="https://github.com/user-attachments/assets/0851e893-eefd-4c26-8b80-093ae40ff8f3" />

Using weighted fusion search with `alpha=1.0` (pure dense) failed to return any bras at all, while searching with `alpha=.25` (75% sparse) returned bras only. 

Some other such failures include "dress", "bikini", "black", "dark"...
They are completely resolved by using SPLADE with `alpha=.25`, and not RRF. This justifies SPLADE necessity for textual embeddings.

Not all types are retrieved badly. For example, "trousers", "pants", "T-shirt", "glasses" queries returned exclusively the specified objects, when searched using RRF.

Searching "white bra", "black dress", "white bikini" or even "turquoise bikini" also retrieved correct objects of that color with RRF and even with alpha=1.0. However, color specification may fail as well (e.g., "black top").

In fact, searching by all product types, it's possible to identify the items, that pollute results the most often:

<img width="1277" height="233" alt="Screenshot_20260519_051903" src="https://github.com/user-attachments/assets/6d8aa8f2-f8cf-43c5-a252-22728c9c2cac" />

This hints at the problem with the dense model. The single-word representation is too fragile, the single-query signal is too weak and fails to point at the correct item-neighborhood. 

Instead, the vector points at some generic destination (a hub) which is close to the centroid, since the conditional expectation $\mathbb{E}[X \mid \mathsf{query}]$ is calculated over many items, if the query is general enough (e.g., "black"). This can be understood as a particular case of the curse of dimensionality, namely, the Hubness problem, as shown in [Radovanović et al. (2010)](https://www.jmlr.org/papers/v11/radovanovic10a.html).

In turn, the search engine in the RRF mode sometimes puts so much weight on the dense vector component, that a strong dense similarity can override a zero SPLADE score, resulting in the observed pollutions.

#### Weak clustering
Another interesting failure is regarding weak clustering. Small categories are expected to cluster at the top, instead the model puts high probability on some incorrect types, leading to correct objects being mingled with incorrect types. One such an example is "tote bag", where a small bag receives 0.50 and some tote bag gets 0.25 probability.
#### Augmentation, Precision@10
LLM (Qwen2.5-3B-Instruct) is used to extract query chains from descriptions of the items in the dataset. 

The chains are of the following form:
$$L0 \rightarrow L1 \rightarrow L2 \rightarrow L3 \rightarrow L4 \rightarrow L5$$, 
where L0-queries are the most basic attribute with a product type (typically 2-3 words) and L5 are comprehensive queries capturing all key features.

An example: <br>
`cotton dress` <br>
`cotton dress with sleeves` <br>
`cotton dress with long sleeves` <br>
`cotton dress with long sleeves and V-neck` <br>
`cotton dress with long sleeves and V-neck in floral` <br>
`cotton dress with long sleeves and V-neck in floral print`

LLM-as-a-judge (Qwen2.5-3B-Instruct) approach is used to classify each returned item as either relevant (1) to the query or irrelevant (0). Then Precision@10 is calculated.

For more details on propmts for augmentation and evaluation, see `ml/notebooks/search_engine/query.py`.

<img width="1016" height="630" alt="Screenshot_20260519_055258" src="https://github.com/user-attachments/assets/783070de-7a73-45e4-ab9f-4e629b1d8b08" />

The search in the RRF mode is stable under query augmentation, the fluctuations in the $0.5\%$ interval can be attributed to noise. 

Some mismatches:

**Query 1**</br>
`microfibre hipster briefs with a high waist`</br>
**Returned description**</br>
`Microfibre hipster briefs with a low waist, lined gusset, wide sides and cutaway coverage at the back.`</br>
The query specifies "high waist" and the product is "low waist".

**Query 2**</br>
`woven dress`</br>
**Returned description**</br>
`Long-sleeved dress in a short, fitted style.`</br>
The dress is not specified to be woven.

**Query 3**</br>
`lace briefs with low waist and lined gusset`</br>
**Returned description**</br>
`Studio Collection. Briefs in patterned mesh with neat lace trims. Low waist, a lined gusset and cutaway coverage at the back. Studio Collection AW20...`</br>
The query asks for fully lace briefs and the search returns briefs with lace trims only.

**Query 4**</br>
`organic cotton t-shirt`</br>
**Returned description**</br>
`Long-sleeved top in organic cotton jersey with a ribbed neckline in a contrasting colour and a gently rounded hem.`</br>
Product mismatch, potentially due to visual ambiguity of t-shirt versus top types.

**Query 4**</br>
`soft sweatshirt fabric top with frills at the front, long sleeves, raglan shoulders, and ribbing around the neckline`</br>
**Returned description**</br>
`Top in soft sweatshirt fabric made from a cotton blend with a round neckline and long raglan sleeves. Ribbing around the neckline, cuffs and hem. Soft brushed inside.`</br>
The query itself is very specific, but it failed to retrieve the correct product: the frills at the front are not present in the description.

Failures under augmentation can be explained in the following ways:
1. The probability of word mismatch grows with query length, overspecified queries allow for more mismatches;
2. Dense search has too much signal, e.g., specifying sizes fails because they are indistinguishable in a picture;
3. The LLM which generated queries could hallucinate more on longer queries;
4. There are simply not enough products for an overspecified query, that is, as $query length \rightarrow \infty$, the number of relevant products approaches 1 and the LLM is too strict and discards "best possible" matches, that the search suggests;
5. The description itself is incomplete, while search may retrieve items by visual recognition too.

The fourth explanation means that the metric P@10 itself punishes longer queries. Nevertheless, this demonstrates, that at least 6 out of 10 retrieved products are good enough on average, according to the LLM, and this "goodness" is stable across long enough queries. 
### Image failure analysis
Random samples of images were not found to fail, the engine excels even at hard images. It easily finds close-ups of objects, repeating the patterns and types of the cloth and material; it finds images with items presented in a collection, e.g., a series of socks, T-shirts or whole garments sets. 
It didn't fail on objects from different categories of various colors randomly sampled from the dataset. 

Since bras and dresses were identified to be problematic categories in text search, 20 bras (10 black and 10 colored) and 10 dresses were sampled and searched by their images, the returned results were fully correct. 

Then precisely the objects from the bra and bikini queries were taken and searched by their images, each image retrieved exactly.

The searched-with image is trivially the top-1 object returned with probabability around 1.0, the others all have probabilities around 0.8 to 0.9. This is unlike to textual queries, where the first returned object has probability around 0.5 and subsequent ones have even less.

These observations can be naturally explained by information completeness, since an image has roughly 100% SNR. More deeply, images, unlike text, already form a smooth manifold, so they are naturally isolated from each other. This gives theoretical guarantees to searching in the image mode.
