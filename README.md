# CoCoG

## Get started
You can then set up a conda environment with all dependencies like so:

```
conda env create -f cocog.yml
conda activate cocog
```

## File structure
The code is organized as follows:

- ConceptDecoder.ipynb: concept-based image generation and decision intervention
- ConceptEncoder.ipynb: extract concept embedding
- dataset.py: load images
- diffusion_prior.py: diffusion model and pipeline to generate clip embedding from concept embedding
- networks.py: concept encoder model
- utils.py: utility functions for plotting
