# JPVAE-mv-Imputation-constraints
  
This project investigates the research question:  
**“How does training with different types and rates of missingness affect the robustness of JPVAE in multi-view learning, and to what extent can this improve imputation quality and affect accuracy for downstream tasks such as classification?”**

Our work extends the [JPVAE implementation](https://github.com/eso28599/JPVAE/blob/master/README.md)) by analyzing robustness under varying missingness scenarios. Specifically, we compare different latent space constraints — orthogonality, eigenvalue regularization, and $C=0$ baseline — to evaluate their impact on:  
- **Imputation quality** of missing views,  
- **Stability of latent representations**, and  
- **Classification accuracy** on MNIST and Fashion-MNIST.  

The experiments are designed to highlight how constraints and data availability (100%, 50%, 5%, 1%) influence the ability of JPVAE to recover missing information and support downstream tasks.  
