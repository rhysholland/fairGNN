#' Train and Evaluate the Gated Neural Network
#'
#' This function performs a complete training and evaluation run, including
#' an optional hyperparameter tuning step and a final evaluation run.
#'
#' @param prepared_data A list object from the `prepare_data()` function.
#' @param hyper_grid A dataframe of hyperparameters to tune. Must include columns:
#'                   `lr`, `hidden_dim`, `dropout_rate`, `lambda`, `temperature`.
#' @param num_repeats Integer, the number of repeated train/test splits for evaluation.
#' @param epochs Integer, the number of training epochs per run.
#' @param output_dir A string path to a directory where results will be saved.
#' @param run_tuning Boolean, whether to run the hyperparameter tuning step.
#' @param best_params A list containing the best parameters (`lr`, `hidden_dim`, `dropout`, `lambda`, `temperature`) if `run_tuning = FALSE`.
#'
#' @return A list containing detailed results: `final_results`, `gate_weights`,
#'         `expert_weights`, `performance_summary`, `aif360_data`, and `tuning_results`.
#' @export
#' @import torch
#' @importFrom caret createDataPartition
#' @importFrom pROC roc auc
#' @importFrom dplyr bind_rows slice_max select mutate group_by summarise
#' @importFrom readr write_csv
#' @importFrom tibble tibble as_tibble
#' @importFrom stats sd var
#'
train_gnn <- function(prepared_data, hyper_grid, num_repeats = 20, epochs = 300, output_dir = "outputs", run_tuning = TRUE, best_params = NULL) {

  # Create output directory if it doesn't exist
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Unpack prepared_data for easier access
  X <- prepared_data$X
  y <- prepared_data$y
  group <- prepared_data$group
  feature_names <- prepared_data$feature_names
  all_subjectids <- prepared_data$subject_ids

  input_dim  <- ncol(X)
  num_groups <- length(unique(group))

  # ----------------------------------------------------------------------------------
  # Helper Function 1: Gated Model Definition
  # ----------------------------------------------------------------------------------
  gated_model <- nn_module(
    "GatedModel",
    initialize = function(input_dim, hidden_dim, num_groups, dropout_p = 0.5) {
      self$gate_layer <- nn_linear(input_dim, num_groups)
      self$subgroup_layers <- nn_module_list(lapply(1:num_groups, function(i) {
        nn_sequential(
          nn_linear(input_dim, hidden_dim),
          nn_batch_norm1d(hidden_dim),
          nn_relu(),
          nn_dropout(p = dropout_p),
          nn_linear(hidden_dim, hidden_dim),
          nn_relu(),
          nn_linear(hidden_dim, 1)
        )
      }))
    },
    forward = function(x, temperature_val = 0.5) {
      gate_weights <- nnf_softmax(self$gate_layer(x) / temperature_val, dim = 2)
      subgroup_logits <- lapply(self$subgroup_layers, function(layer) layer(x)$squeeze(2))
      subgroup_logits <- torch_stack(subgroup_logits, dim = 2)
      output_logits <- torch_sum(gate_weights * subgroup_logits, dim = 2)
      output_probs <- torch_sigmoid(output_logits)
      attr(output_probs, "gate_weights") <- gate_weights
      output_probs
    }
  )

  # ----------------------------------------------------------------------------------
  # Helper Function 2: Custom Fairness-Constrained Loss
  # ----------------------------------------------------------------------------------
  fairness_constrained_loss <- function(y_pred, y_true, groups, lambda = 1.0) {
    bce_loss <- nnf_binary_cross_entropy(y_pred, y_true)
    if (lambda == 0) return(bce_loss)

    epsilon <- 1e-7
    unique_groups_val <- sort(unique(as.numeric(groups)))

    if (length(unique_groups_val) < 2) return(bce_loss) # No fairness loss if only one group

    soft_tprs <- list()
    soft_fprs <- list()

    for (g_val in unique_groups_val) {
      mask <- groups == g_val
      if (torch_sum(mask)$item() > 0) {
        y_pred_group <- y_pred[mask]
        y_true_group <- y_true[mask]

        soft_tp <- torch_sum(y_pred_group * y_true_group)
        soft_fp <- torch_sum(y_pred_group * (1 - y_true_group))
        p <- torch_sum(y_true_group)
        n <- torch_sum(1 - y_true_group)

        soft_tprs[[as.character(g_val)]] <- soft_tp / (p + epsilon)
        soft_fprs[[as.character(g_val)]] <- soft_fp / (n + epsilon)
      }
    }

    tpr_tensor <- torch_stack(soft_tprs)
    fpr_tensor <- torch_stack(soft_fprs)

    tpr_variance <- torch_var(tpr_tensor)
    fpr_variance <- torch_var(fpr_tensor)

    fairness_loss <- tpr_variance + fpr_variance

    return(bce_loss + lambda * fairness_loss)
  }


  # ----------------------------------------------------------------------------------
  # Hyperparameter Tuning Loop
  # ----------------------------------------------------------------------------------
  tuning_results <- tibble()
  if (run_tuning) {
    cat("Starting hyperparameter tuning with", nrow(hyper_grid), "combinations...\n")
    for (j in 1:nrow(hyper_grid)) {
      params <- hyper_grid[j, ]
      cat(sprintf("\n--- Testing Combo %d/%d: LR=%.4f, Hidden=%d, Dropout=%.1f, Lambda=%.1f, Temp=%.1f ---\n",
                  j, nrow(hyper_grid), params$lr, params$hidden_dim, params$dropout_rate, params$lambda, params$temperature))

      repeat_auc_scores <- c()
      for (i in seq_len(num_repeats)) {
        set.seed(111 * i)
        train_idx <- caret::createDataPartition(y, p = 0.8, list = FALSE)
        X_train <- X[train_idx, ]; X_test <- X[-train_idx, ]
        y_train <- y[train_idx]; y_test <- y[-train_idx]
        group_train <- group[train_idx]

        x_train_t <- torch_tensor(as.matrix(X_train), dtype = torch_float())
        y_train_t <- torch_tensor(y_train, dtype = torch_float())
        group_train_t <- torch_tensor(group_train, dtype = torch_float())
        x_test_t <- torch_tensor(as.matrix(X_test), dtype = torch_float())

        net <- gated_model(input_dim, params$hidden_dim, num_groups, dropout_p = params$dropout_rate)
        optimiser <- optim_adam(net$parameters, lr = params$lr)

        for (epoch in 1:epochs) {
          net$train()
          optimiser$zero_grad()
          out <- net(x_train_t, temperature_val = params$temperature)
          loss <- fairness_constrained_loss(out, y_train_t, group_train_t, lambda = params$lambda)
          loss$backward()
          optimiser$step()
        }

        net$eval()
        with_no_grad({
          probs <- net(x_test_t, temperature_val = params$temperature)
        })
        roc_obj <- pROC::roc(response = y_test, predictor = as.numeric(probs), quiet = TRUE, levels = c(0, 1))
        repeat_auc_scores <- c(repeat_auc_scores, pROC::auc(roc_obj))
      }
      mean_auc <- mean(repeat_auc_scores, na.rm = TRUE)
      tuning_results <- dplyr::bind_rows(tuning_results, tibble::as_tibble(c(params, mean_auc = mean_auc)))
      cat("--- Result for Combo", j, ": Mean AUC =", round(mean_auc, 4), "---\n")
    }
    best_params <- tuning_results %>% dplyr::slice_max(order_by = mean_auc, n = 1)
    cat("\n--- Best Performing Combination Found ---\n")
    print(best_params)
  } else {
    if (is.null(best_params)) stop("If run_tuning is FALSE, best_params must be provided.")
    cat("\n--- Skipping tuning. Using provided parameters: ---\n")
    print(best_params)
  }


  # ----------------------------------------------------------------------------------
  # Final Model Training with Best Hyperparameters
  # ----------------------------------------------------------------------------------
  cat("\n--- Starting Final Model Run with Best Parameters ---\n")
  final_lr <- best_params$lr[1]
  final_hidden_dim <- best_params$hidden_dim[1]
  final_dropout <- best_params$dropout_rate[1]
  final_lambda <- best_params$lambda[1]
  final_temperature <- best_params$temperature[1]

  results_list <- list()
  gate_weights_list <- list()
  expert_weights_list <- list()

  for (i in seq_len(num_repeats)) {
    set.seed(42 * i)
    cat(sprintf("Running final iteration %d/%d.\n", i, num_repeats))
    train_idx <- caret::createDataPartition(y, p = 0.8, list = FALSE)

    X_train <- X[train_idx, , drop = FALSE]; X_test <- X[-train_idx, , drop = FALSE]
    y_train <- y[train_idx]; y_test <- y[-train_idx]
    group_train <- group[train_idx]; group_test <- group[-train_idx]
    subjectid_test <- if (!is.null(all_subjectids)) all_subjectids[-train_idx] else 1:length(y_test)

    x_train_t <- torch_tensor(as.matrix(X_train), dtype = torch_float())
    y_train_t <- torch_tensor(y_train, dtype = torch_float())
    group_train_t <- torch_tensor(group_train, dtype = torch_float())
    x_test_t <- torch_tensor(as.matrix(X_test), dtype = torch_float())

    net <- gated_model(input_dim, final_hidden_dim, num_groups, dropout_p = final_dropout)
    optimiser <- optim_adam(net$parameters, lr = final_lr)

    for (epoch in 1:epochs) {
      net$train()
      optimiser$zero_grad()
      out <- net(x_train_t, temperature_val = final_temperature)
      loss <- fairness_constrained_loss(out, y_train_t, group_train_t, lambda = final_lambda)
      loss$backward()
      optimiser$step()
    }

    net$eval()
    with_no_grad({
      logits <- net(x_test_t, temperature_val = final_temperature)
    })
    probs <- as.numeric(logits)
    results_list[[i]] <- tibble::tibble(iteration = i, subjectid = subjectid_test, true = y_test, prob = probs, group = group_test)

    gate_w_matrix <- as.matrix(attr(logits, "gate_weights"))
    gate_entropy <- as.numeric(-rowSums(gate_w_matrix * log(gate_w_matrix + 1e-10)))
    expert_weights_list[[i]] <- lapply(net$subgroup_layers, function(layer) as.numeric(layer[[1]]$weight))

    # Dynamically create gate probability columns
    gate_probs_df <- as.data.frame(gate_w_matrix)
    names(gate_probs_df) <- paste0("gate_prob_expert_", 0:(num_groups-1))

    gate_weights_list[[i]] <- dplyr::bind_cols(
      tibble::tibble(
        iteration = i, subjectid = subjectid_test, group = group_test,
        gate_entropy = gate_entropy
      ),
      gate_probs_df
    )
  }

  # ----------------------------------------------------------------------------------
  # Collate and Save Results
  # ----------------------------------------------------------------------------------
  final_results <- dplyr::bind_rows(results_list)
  gate_weights <- dplyr::bind_rows(gate_weights_list)

  roc_all <- pROC::roc(response = final_results$true, predictor = final_results$prob, quiet = TRUE, levels = c(0, 1))
  auc_final <- as.numeric(pROC::auc(roc_all))
  brier_final <- mean((final_results$true - final_results$prob)^2)
  cat(sprintf("\nFinal Overall AUC Across All Test Repeats: %.4f\n", auc_final))
  cat(sprintf("Final Overall Brier Score Across All Test Repeats: %.4f\n", brier_final))

  performance_summary <- tibble::tibble(Metric = c("AUC", "Brier Score"), Value = c(auc_final, brier_final))

  aif360_data <- final_results %>%
    dplyr::select(subjectid, true_label = true, predicted_prob = prob, sensitive_attr_numeric = group)

  # Save outputs to files
  readr::write_csv(final_results, file.path(output_dir, "gnn_final_predictions.csv"))
  readr::write_csv(gate_weights, file.path(output_dir, "gnn_gate_weights.csv"))
  readr::write_csv(aif360_data, file.path(output_dir, "gnn_aif360_data.csv"))
  if (run_tuning) {
    readr::write_csv(tuning_results, file.path(output_dir, "gnn_tuning_results.csv"))
  }
  saveRDS(expert_weights_list, file.path(output_dir, "gnn_expert_weights.rds"))

  # Return a comprehensive list of results
  return(list(
    final_results = final_results,
    gate_weights = gate_weights,
    expert_weights = expert_weights_list,
    performance_summary = performance_summary,
    aif360_data = aif360_data,
    tuning_results = if(run_tuning) tuning_results else "Tuning was skipped"
  ))
}
