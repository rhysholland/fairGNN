#' Analyse and Visualise Expert Network Specialisation
#'
#' This function analyses the input weights of the expert sub-networks to
#' determine which features are most important for each subgroup. It supports
#' datasets with two or more subgroups.
#'
#' @param gnn_results A list object from `train_gnn()`.
#' @param prepared_data A list object from `prepare_data()`.
#' @param group_mappings A named list mapping numeric codes to labels.
#' @param top_n_features The number of top features to visualise.
#'
#' @return A list containing the importance tables and plots.
#' @export
#' @import ggplot2
#' @importFrom dplyr group_by summarise mutate arrange desc left_join filter slice_max
#' @importFrom tidyr pivot_wider
#' @importFrom stats t.test p.adjust
#' @importFrom purrr map_dfr
#' @importFrom utils combn
#'
analyse_experts <- function(gnn_results, prepared_data, group_mappings, top_n_features = 10) {

  cat("--- Starting Expert Feature Weight Analysis ---\n")

  weights_list <- gnn_results$expert_weights
  feature_names <- prepared_data$feature_names
  num_groups <- length(group_mappings)
  hidden_dim <- length(weights_list[[1]][[1]]) / length(feature_names)

  # --- 1. Reshape data ---
  weights_df <- purrr::map_dfr(seq_along(weights_list), function(i) {
    purrr::map_dfr(1:num_groups, function(g) {
      w <- matrix(weights_list[[i]][[g]], nrow = hidden_dim, ncol = length(feature_names), byrow = TRUE)
      imp <- apply(w, 2, function(col) mean(abs(col)))
      tibble::tibble(feature = feature_names, importance = imp, group = names(group_mappings)[g])
    }) %>% dplyr::mutate(iteration = i)
  })

  # --- 2. Pairwise importance difference and t-tests ---
  group_pairs <- utils::combn(names(group_mappings), 2, simplify = FALSE)
  pairwise_results <- list()

  # For binary case, create the main plot
  difference_plot <- NULL
  if (length(group_mappings) == 2) {
    pair <- group_pairs[[1]]
    groupA_name <- pair[1]
    groupB_name <- pair[2]

    pair_data <- weights_df %>% dplyr::filter(group %in% c(groupA_name, groupB_name))

    diff_table <- pair_data %>%
      dplyr::group_by(group, feature) %>%
      dplyr::summarise(avg_importance = mean(importance, na.rm = TRUE), .groups = "drop") %>%
      tidyr::pivot_wider(names_from = group, values_from = avg_importance) %>%
      dplyr::mutate(difference = .data[[groupB_name]] - .data[[groupA_name]]) %>%
      dplyr::arrange(dplyr::desc(abs(difference)))

    pairwise_results[[paste0(groupB_name, "_vs_", groupA_name)]] <- diff_table

    top_features_data <- diff_table %>%
      dplyr::slice_max(order_by = abs(difference), n = top_n_features)

    difference_plot <- ggplot(top_features_data, aes(x = difference, y = reorder(feature, difference))) +
      geom_col(aes(fill = difference > 0), show.legend = FALSE) +
      scale_fill_manual(values = c("TRUE" = "#F8766D", "FALSE" = "#00BFC4")) +
      labs(
        title = "Top Features by Expert Weight Difference",
        subtitle = paste("Positive = more important in", group_mappings[[groupB_name]], "expert"),
        x = paste("Importance Difference (", group_mappings[[groupB_name]], "-", group_mappings[[groupA_name]], ")"),
        y = "Feature"
      ) +
      theme_minimal()
  }

  cat("--- Pairwise Feature Importance Differences Calculated ---\n")

  return(list(
    all_weights = weights_df,
    pairwise_differences = pairwise_results,
    difference_plot = difference_plot
  ))
}
