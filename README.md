# fairGNN <img src="man/figures/logo.png" align="right" width="120"/>

**Fairness-Aware Gated Neural Networks in R**

[![CRAN status](https://www.r-pkg.org/badges/version/fairGNN)](https://CRAN.R-project.org/package=fairGNN)
[![R-CMD-check](https://github.com/rhysholland/fairGNN/actions/workflows/R-CMD-check.yaml/badge.svg)](https://github.com/rhysholland/fairGNN/actions)

---

## Overview

`fairGNN` provides tools for training, evaluating, and interpreting  
**fairness-aware gated neural networks (GNNs)** for subgroup analysis in prediction tasks.  

The package enables:
- Preprocessing of datasets into features, outcomes, and protected attributes  
- Training of GNNs with fairness-constrained losses  
- Analysis of expert subnetworks and subgroup-specific feature importance  
- Visualisation of gate weights, ROC and calibration curves, and Sankey diagrams  

This work underpins a manuscript currently in preparation for submission to *Bioinformatics*,  
where the package and its methodology will be delineated in full.  

---

## Installation

From CRAN (following acceptance — this README will be updated once live):

```r
install.packages("fairGNN")
```

From GitHub (development version):

```r
# install.packages("devtools")
devtools::install_github("rhysholland/fairGNN")
```

---

## Example usage

More detailed examples and tutorials will be provided in the vignettes and in the forthcoming paper.  
Below is a minimal illustration of the workflow.

```r
library(fairGNN)

# Example data
my_data <- data.frame(
  subjectid = 1:10,
  remission = sample(0:1, 10, replace = TRUE),
  gender = sample(c("M", "F"), 10, replace = TRUE),
  feature1 = rnorm(10),
  feature2 = rnorm(10)
)

# Prepare data
prepared <- prepare_data(
  data = my_data,
  outcome_var = "remission",
  group_var = "gender",
  group_mappings = list("M" = 0, "F" = 1),
  cols_to_remove = "subjectid"
)

# Define a simple hyperparameter grid
grid <- data.frame(
  lr = 0.01,
  hidden_dim = 8,
  dropout_rate = 0.5,
  lambda = 0.1,
  temperature = 0.5
)

# Train the model
gnn_results <- train_gnn(
  prepared_data = prepared,
  hyper_grid = grid,
  num_repeats = 2,
  epochs = 50,
  run_tuning = FALSE,
  best_params = grid
)

# Analyse results (ROC curve example)
analysis <- analyse_gnn_results(
  gnn_results,
  prepared,
  group_mappings = list("0" = "Male", "1" = "Female")
)

print(analysis$roc_plot)
```

---

## Documentation & Citation

- 📘 [Vignette: Introduction to fairGNN](./vignettes/introduction-to-fairGNN.html)  
- 📦 CRAN page: <https://CRAN.R-project.org/package=fairGNN> (will be live after submission)  
- 🐛 Issues: <https://github.com/rhysholland/fairGNN/issues>  

If you use `fairGNN` in your research, please cite the forthcoming Bioinformatics manuscript (details will be added here once accepted).  

Example citation (placeholder):  

```
Holland, R. (2025). fairGNN: Fairness-Aware Gated Neural Networks in R. Bioinformatics (under review).
```

---

## License

MIT © [Rhys Holland](mailto:rhys.holland@icloud.com)
