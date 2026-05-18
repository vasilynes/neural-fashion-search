## Neural Fashion Search Engine
This project adapts the [FashionCLIP model](https://huggingface.co/patrickjohncyh/fashion-clip) to the [H&M Personalized Fashion Recommendations dataset](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data?select=images) (100+ SKUs).

The API of the adapted model is exposed via the search interface frontend, the user can search items by text, images or combined.
This creates a domain-adapted, multimodal search architecture.

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
* `ml/notebooks/01_dataset_cleaning`
* `ml/notebooks/02_text_analysis`
* `ml/notebooks/03_categorical_analysis`
### Domain Adaptation
Adapted FashionCLIP demonstrates increase in its recall (i2t = imate-to-text, t2i = text-to-image):
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
The search engine is based on combining dense and sparse embeddings. To evaluate it fully, LLM was used on 10 000 unique descriptions to generate synthetic user queries of the following types:

1. Synonyms, where the modifiers are altered to their synonyms
2. Typos, where the descriptions are polluted
3. User queries, which mimic real user queries
4. Attribute swaps, which swap key modifiers to their opposites or disjunctive variations

The first two measure search resilience.

The third measures how well the search tolerates the distribution shift from full descriptions to short user queries with specific vocabulary.

The forth tests how well the search distinguishes between modifiers.

HR@10 is a portion of queries that resulted in retrieving at least one item from the set of objects that are described by the string, from which the query was derived. MRR measures the average reciprocal index of the first occurrence of such an object.

For attribute swaps, HR@10 logic is reversed: the hit is 1 only if the items are not in the set of objects that are described by the string, so this is effectively the failure rate, not the hit rate, but since all these metrics can be regarded as a certain measure of success of a query, they are formally grouped to HR@10.

Both metrics are computed using weighted average fusion with `alphas = [.0, .25, .5, .75, 1]`, where `alpha` is the weight put on the dense component, and Max-Scaling normalization. For more details, see `ml/notebooks/scripts/metrics_by_alpha.py` script.

The metrics are then recomputed using RRF, which formally corresponds to alpha = .5 (both models are allowed to contribute equally), but retrieves results differently.

<img width="1640" height="600" alt="Screenshot_20260519_043258" src="https://github.com/user-attachments/assets/eb67dc0d-7832-4fb3-8807-8a4ba2d12f07" />

* SPLADE surpasses the dense model on Typos and Real User queries  due to its learned vocabulary expansion.
* Dense model surpasses SPLADE on Attribute swaps. Hypothetically, this demonstrates the power of contrastive training: the dense model is anchored by visuals and maps modifiers apart, avoiding SPLADE's partial-word-match mistake.
* RRF excels at Synonyms, beating the ceiling of either model alone.

Except Synonyms, other types are slightly worse off from RRF and are better off from setting `alpha=.25` or near it, leading to asymmetric hybrid search, where lexical expansion does the heavy lifting. 

However, since the data is fully synthetic, the study doesn't demonstrate the full distribution dynamics, so that further explorations are needed.
### Failure analysis
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

This hints at the problem with the dense model. The single-word representation is too fragile, the single-query signal is too weak and fails to point at the correct query-neighborhood. 

Instead, the vector points at some generic destination (a hub) which is close to the centroid, since the conditional expectation $\mathbb{E}[X \mid \mathsf{query}]$ is calculated over many items, if the query is general enough (e.g., "black"). Formally, it's known as the Hubness problem, see [Radovanović et al. (2010)](https://www.jmlr.org/papers/v11/radovanovic10a.html).

In turn, the search engine in the RRF mode sometimes puts so much weight on the dense vector component, that a strong dense similarity can override a zero SPLADE score, resulting in the observed pollutions.

#### Weak clustering
Another interesting failure is regarding weak clustering. Small categories are expected to cluster at the top, instead the model puts high probability on some incorrect types, leading to correct objects being mingled with incorrect types. One such an example is "tote bag", where a small bag receives 0.50 and some tote bag gets 0.25 probability.

