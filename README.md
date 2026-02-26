# Implied-Volatility-Forecasting-with-Diffusion-Models

This project extends the paper "Forecasting implied volatility surface with generative diffusion models" by Jin & Agarwal (2025), proposing a novel Diffusion Transformer (DiT) architecture that replaces the traditional convolution-based U-Net approach with a transformer-based model for volatility surface forecasting.

We replace the U-Net architecture with a Diffusion Transformer that:

- Captures global dependencies from the first layer via self-attention
- Models term structure explicitly through patch-to-patch attention
- Provides interpretable attention maps showing smile and term structure dynamics




# References

## Original Paper

Jin & Agarwal (2025). "Forecasting implied volatility surface with generative diffusion models." arXiv:2511.07571

## Diffusion Models

Ho et al. (2020). "Denoising Diffusion Probabilistic Models." NeurIPS.
Nichol & Dhariwal (2021). "Improved Denoising Diffusion Probabilistic Models." ICML.
Song et al. (2021). "Score-Based Generative Modeling through Stochastic Differential Equations." ICLR.

## Transformers for Diffusion

Peebles & Xie (2023). "Scalable Diffusion Models with Transformers." ICCV.
Rombach et al. (2022). "High-Resolution Image Synthesis with Latent Diffusion Models." CVPR.

## Volatility Modeling

Dupire (1994). "Pricing with a Smile." Risk Magazine.
Gatheral (2006). "The Volatility Surface: A Practitioner's Guide." Wiley.

Dupire (1994). "Pricing with a Smile." Risk Magazine.
Gatheral (2006). "The Volatility Surface: A Practitioner's Guide." Wiley.
