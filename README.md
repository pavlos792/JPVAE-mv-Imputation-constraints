# JPVAE-mv-Imputation-missing-rates
  
This MSc project investigates the research question:  
**“How does training with different types and rates of missingness affect the robustness of JPVAE in multi-view learning, and to what extent can this influences imputation quality and affects accuracy on downstream tasks such as classification?”**

Our work extends the [JPVAE](https://github.com/eso28599/JPVAE/blob/master/README.md) implementation [work by Ella Orme] by analyzing robustness under varying missingness scenarios. Specifically, we compare different latent space constraints to evaluate their impact on:
- How sensitive is the latent space of JPVAE to different rates of digit-level and view-level missingness during training using the MNIST dataset?  
- How does imputing a missing view from an unseen input affect classification accuracy on the test set compared to classification with complete views? What is the minimum sample size required before performance begins to deteriorate?  
- How does JPVAE generalize when applied to a different dataset, such as Fashion-MNIST?

The experiments are designed to highlight how constraints and data availability (100%, 50%, 5%, 1%) influence the ability of JPVAE to recover missing information and support downstream tasks. 

> 📝 **Note:** All experiments and analyses in this repository were conducted using **Google Colab**, leveraging its GPU/TPU resources for training and evaluation.  

