# Hierarchical Semantics Encoding (HSE)

This directory contains the residual quantization implementation used by DSLaw's **Hierarchical Semantics Encoding (HSE)** module. The directory name `rqvae` is retained from the earlier implementation.

Start the training process by executing the corresponding dataset's `run.sh`.

After training is completed, run `{dataset}_gen.ipynb` in `eval/` to export the case representations, semantic IDs, charge representations, and codebook vectors required by the **Case Representation Learning (CRL)** module.
