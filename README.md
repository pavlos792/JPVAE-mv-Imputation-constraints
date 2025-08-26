# JPVAE-mv-Imputation-missing-rates
  
This MSc project investigates the research question:  
**“How does training with different types and rates of missingness affect the robustness of JPVAE in multi-view learning, and to what extent can this influences imputation quality and affects accuracy on downstream tasks such as classification?”**

Our work extends the [JPVAE](https://github.com/eso28599/JPVAE/blob/master/README.md) implementation [work by Ella Orme] by analyzing robustness under varying missingness scenarios. Specifically, we compare different latent space constraints to evaluate their impact on:
- How sensitive is the latent space of JPVAE to different rates of digit-level and view-level missingness during training using the MNIST dataset?  
- How does imputing a missing view from an unseen input affect classification accuracy on the test set compared to classification with complete views? What is the minimum sample size required before performance begins to deteriorate?  
- How does JPVAE generalize when applied to a different dataset, such as Fashion-MNIST?

The experiments are designed to highlight how constraints and data availability (100%, 50%, 5%, 1%) influence the ability of JPVAE to recover missing information and support downstream tasks. 

> 📝 **Note:** All experiments and analyses in this repository were conducted using **Google Colab**, leveraging its GPU/TPU resources for training and evaluation.  

## How to Run the Experiments

There are **three notebooks** provided for each dataset, corresponding to different latent space constraints:

- `JPVAE_100p_MNIST_CEval_zerorot_CNN.ipynb` — *Eigenvalue regularization constraint*  
- `JPVAE_100p_MNIST_Corth_zerorot_CNN.ipynb` — *Orthogonality constraint*  
- `JPVAE_100p_MNIST_Czero_zerorot_CNN.ipynb` — *No latent correlation (baseline)*  

### Step 1: Set training percentage
In the **second cell** of each notebook, define the training dataset percentage to use (e.g., `100`, `50`, `5`, or `1`).  

### Step 2: Rotate the dataset

In both `train_vae_withCorthog.py` and `train_vae_withCevals.py`, the `train_and_eval_split` function includes an option to rotate dataset images in order to introduce nonlinearity.  

This is controlled by the list `rotation_angles_pattern` (search for it with **CTRL+F**). The list defines the rotation angles for every sequence of four consecutive images. For example, setting rotation_angles_pattern = [0, 45, 90, 135] :

This pattern applies the following transformations:

- First image → unchanged (0°)  
- Second image → rotated by 45°  
- Third image → rotated by 90°  
- Fourth image → rotated by 135°  

The sequence then repeats for the next four images and goes on.  

⚠️ *Note:* All results provided in this repository were obtained **without any rotation applied** (default setting).  

### Step 3: Configure directories
In the **first cell**, set the path to your Python scripts:  

sys.path.append('/content/drive/MyDrive/Colab_Notebooks/MNIST/Classification/')

to save your results and figures (for the 100% case)
- figures_base_dir_original = '/content/drive/MyDrive/Colab_Notebooks/MNIST/Classification/Baseline_100p/'
- results_base_dir = '/content/drive/MyDrive/Colab_Notebooks/MNIST/Classification/Baseline_100p/'

### Step 4: Organize folders

Inside each baseline folder (e.g., `Baseline_100p`), create three subfolders:

- `C_zero_zerorot_CNN`  
- `C_orth_zerorot_CNN`  
- `C_eval_zerorot_CNN`  

For each different percentage (e.g., `Baseline_50p`, `Baseline_5p`, `Baseline_1p`), create the same three subfolders and update the paths in the notebook accordingly.  

Example folder structure:

```text
Classification/
├── Baseline_100p/
│   ├── C_zero_zerorot_CNN/
│   ├── C_orth_zerorot_CNN/
│   └── C_eval_zerorot_CNN/
├── Baseline_50p/
│   ├── C_zero_zerorot_CNN/
│   ├── C_orth_zerorot_CNN/
│   └── C_eval_zerorot_CNN/
├── Baseline_5p/
│   ├── C_zero_zerorot_CNN/
│   ├── C_orth_zerorot_CNN/
│   └── C_eval_zerorot_CNN/
└── Baseline_1p/
    ├── C_zero_zerorot_CNN/
    ├── C_orth_zerorot_CNN/
    └── C_eval_zerorot_CNN/
```

### Step 5: Plotting

In each subfolder, three CSV files will be generated:  

- `orthogonal_results.csv`  
- `eval_results.csv`  
- `zero_results.csv`  

To visualize results, collate the outputs from all percentages (100%, 50%, 5%, 1%) into a single CSV file for each method (orthogonal, eval, zero). The format should follow the examples provided in the `results/` folder.  
Then, run the notebook:  

- `JPVAE_100p_missingview_plotting_comparison_FMNIST_allpcs.ipynb`  

Before running, update the path variables in the first cell:  

- **results_base_dir** → location of the combined CSV files  
- **figures_base_dir_original** → directory where you want the figures to be saved  
