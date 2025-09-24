#' Analyse and Visualize GNN Results
#'
#' Generates plots and statistical tests for interpreting the GNN model results.
#'
#' @param gnn_results A list object from the `train_gnn()` function.
#' @param prepared_data A list object from the `prepare_data()` function.
#' @param group_mappings A named list that maps the numeric group codes back to their character labels for plotting (e.g., `list('0' = "Male", '1' = "Female")`).
#' @param create_roc_plot Boolean, if TRUE, generates and returns a ROC curve plot.
#' @param create_calibration_plot Boolean, if TRUE, generates and returns a calibration plot.
#' @param analyse_gate_weights Boolean, if TRUE, performs gate weight analysis (density plot and t-test).
#' @param analyse_gate_entropy Boolean, if TRUE, performs gate entropy analysis.
#' @param verbose Logical, whether to print progress messages (default FALSE).
#'
#' @return A list containing ggplot objects and analysis tables.
#' @export
#' @import ggplot2
#' @importFrom stats t.test sd
#' @importFrom pROC roc auc
#' @importFrom dplyr mutate group_by summarise filter n left_join
#' @importFrom dplyr where
#' @import ggplot2
#' @importFrom stats t.test sd
#' @importFrom pROC roc auc
#' @importFrom dplyr mutate group_by summarise filter n left_join
#' @importFrom dplyr where
#'
analyse_gnn_results <- function(gnn_results, prepared_data, group_mappings,
                                create_roc_plot = TRUE,
                                create_calibration_plot = TRUE,
                                analyse_gate_weights = TRUE,
                                analyse_gate_entropy = TRUE,
                                verbose = FALSE) {

  # --- Input validation ---
  if (!all(c("final_results", "performance_summary", "gate_weights") %in% names(gnn_results))) {
    stop("gnn_results object is missing required components.")
  }

  output_list <- list()
  results_all <- gnn_results$final_results
  auc_gated <- gnn_results$performance_summary$Value[gnn_results$performance_summary$Metric == "AUC"]
  brier_gated <- gnn_results$performance_summary$Value[gnn_results$performance_summary$Metric == "Brier Score"]

  # --- Map group numbers to labels for plotting ---
  gate_data <- gnn_results$gate_weights %>%
    dplyr::mutate(group_label = as.character(group_mappings[as.character(group)]))

  # --- 1. ROC Curve Plot ---
  if (create_roc_plot) {
    if (verbose) message("Generating ROC Plot...")
    roc_data <- results_all %>%
      dplyr::mutate(true_factor = factor(true, levels = c(0, 1)))

    roc_obj <- pROC::roc(response = roc_data$true, predictor = roc_data$prob, quiet = TRUE, levels = c(0, 1))

    roc_plot <- ggplot(data.frame(specificity = 1 - roc_obj$specificities, sensitivity = roc_obj$sensitivities),
                       aes(x = specificity, y = sensitivity)) +
      geom_path(color = "darkblue", linewidth = 1) +
      geom_abline(linetype = "dashed") +
      labs(
        title = "ROC Curve",
        subtitle = paste("AUC =", round(auc_gated, 3)),
        x = "False Positive Rate (1 - Specificity)",
        y = "True Positive Rate (Sensitivity)"
      ) +
      theme_minimal()
    output_list$roc_plot <- roc_plot
  }

  # --- 2. Calibration Plot ---
  if (create_calibration_plot) {
    if (verbose) message("Generating Calibration Plot...")
    calibration_data <- results_all %>%
      dplyr::mutate(prob_bin = cut(prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)) %>%
      dplyr::group_by(prob_bin) %>%
      dplyr::summarise(
        mean_predicted_prob = mean(prob),
        observed_proportion = mean(true),
        .groups = "drop"
      ) %>%
      dplyr::filter(!is.na(prob_bin))

    calibration_plot <- ggplot(calibration_data, aes(x = mean_predicted_prob, y = observed_proportion)) +
      geom_abline(linetype = "dashed") +
      geom_line(color = "firebrick", linewidth = 1) +
      geom_point(color = "firebrick", size = 2) +
      coord_cartesian(xlim = c(0, 1), ylim = c(0, 1)) +
      labs(
        title = "Calibration Plot",
        subtitle = paste("Brier Score =", round(brier_gated, 3)),
        x = "Mean Predicted Probability",
        y = "Observed Proportion of Remission"
      ) +
      theme_minimal()
    output_list$calibration_plot <- calibration_plot
  }

  # --- 3. Gate Weight Analysis ---
  if (analyse_gate_weights) {
    if (verbose) message("Performing Gate Weight Analysis...")
    gate_density_plot <- ggplot(gate_data, aes(x = gate_prob_expert_1, fill = group_label)) +
      geom_density(alpha = 0.7) +
      labs(
        title = "Gate Weight Distribution by Subgroup",
        subtitle = "Shows routing preference for 'Expert 1'",
        x = "Gate Weight for 'Expert 1'",
        y = "Density",
        fill = "Actual Group"
      ) +
      theme_minimal()
    output_list$gate_density_plot <- gate_density_plot

    if (length(unique(gate_data$group_label)) > 1) {
      gate_ttest <- stats::t.test(gate_prob_expert_1 ~ group_label, data = gate_data)
      output_list$gate_weight_ttest <- gate_ttest
    }
  }

  # --- 4. Gate Entropy Analysis ---
  if (analyse_gate_entropy) {
    if (verbose) message("Performing Gate Entropy Analysis...")
    if (!"gate_entropy" %in% names(gate_data)) {
      warning("gate_entropy not found in results. Did you use the latest train_gnn function?")
    } else {
      entropy_density_plot <- ggplot(gate_data, aes(x = gate_entropy, fill = group_label)) +
        geom_density(alpha = 0.7) +
        labs(
          title = "Gate Entropy Distribution by Subgroup",
          subtitle = "Entropy quantifies routing uncertainty (0 = decisive)",
          x = "Gate Entropy",
          y = "Density",
          fill = "Actual Group"
        ) +
        theme_minimal()
      output_list$entropy_density_plot <- entropy_density_plot

      if (length(unique(gate_data$group_label)) > 1) {
        entropy_ttest <- stats::t.test(gate_entropy ~ group_label, data = gate_data)
        output_list$gate_entropy_ttest <- entropy_ttest
      }
    }
  }

  if (verbose) message("Analysis complete.")
  return(output_list)
}
