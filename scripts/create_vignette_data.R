# Script: create_vignette_data.R

# 1. Load libraries
library(fairGNN)
library(readxl)
library(here)

# 2. Load your original raw data
# This path now correctly finds the data inside your project
raw_data <- read_xlsx(here::here("raw_data/data-rem.xlsx"))

# 3. Prepare the Male/Female data
numeric_mappings_gender <- list('2' = 0, '1' = 1) # Male=0, Female=1
label_mappings_gender <- list('0' = "Male", '1' = "Female")

cols_to_drop <- c(
  "subjectid", "Row.names", "bloodsampleid.x", "madrs.total", "hrsd.total",
  "bdi.total", "bdi14wk0", "bdi20wk0", "f61score0", "f62score0",
  "f64score0", "f65score0", "k30"
)

prepared_data_gender <- prepare_data(
  data = raw_data,
  outcome_var = "hdremit.all",
  group_var = "sex",
  group_mappings = numeric_mappings_gender,
  cols_to_remove = cols_to_drop
)

# 4. Define the best hyperparameters from your original tuning
best_params <- data.frame(
  lr = 0.0005, hidden_dim = 64, dropout_rate = 0.5,
  lambda = 0.5, temperature = 0.5
)

# 5. Train the GNN model
gnn_results <- train_gnn(
  prepared_data = prepared_data_gender,
  run_tuning = FALSE,
  best_params = best_params,
  num_repeats = 5, # Using fewer repeats for speed
  epochs = 100
)

# 6. Run the expert analysis
expert_analyses <- analyse_experts(
  gnn_results = gnn_results,
  prepared_data = prepared_data_gender,
  group_mappings = label_mappings_gender
)

# 7. Save the final objects for the vignette
if (!dir.exists("vignettes/data")) {
  dir.create("vignettes/data", recursive = TRUE)
}
saveRDS(gnn_results, file = "vignettes/data/gnn_results.rds")
saveRDS(expert_analyses, file = "vignettes/data/expert_analyses.rds")

cat("\n--- Vignette data successfully created and saved! ---\n")
