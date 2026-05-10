# Few-Shot Learning: Matching Networks, Prototypical Networks, and Relation Network
 
**Team Nutella** — Nilarnab Debnath & Riya Mahesh
 
---
 
## What this is
 
An implementation and comparison of three metric-based meta-learning approaches for few-shot image classification, evaluated on the Omniglot benchmark under a 5-way 1-shot setting.
 
---
 
## Models
 
**Matching Networks** — classifies a query by computing a weighted sum over support labels, where weights come from a cosine-softmax attention kernel between embedded support and query images. No fine-tuning at inference.
 
**Prototypical Networks** — represents each class as the mean of its support embeddings (a prototype), then classifies by nearest prototype under squared Euclidean distance.
 
**Relation Network** — rather than using a fixed distance metric, learns a relation module (a small CNN) that takes the concatenated embeddings of a query and support image and outputs a similarity score in [0, 1]. We also add a **cross-attention extension** where support and query features attend to each other before being scored, improving accuracy from 98.2% to 98.51%.
 
All three use the same 4-block CNN backbone and are trained episodically.


## Implementation Details
Implementation details and procedure to run the code are given in a separate README.md file in the individual directories.