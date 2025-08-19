#' Create a Sankey Plot to Visualise Patient Routing
#'
#' Generates a Sankey diagram showing how patients from different subgroups
#' are routed to different experts based on their learned feature profiles.
#'
#' @param raw_data The original, unscaled dataframe.
#' @param gnn_results The results object from `train_gnn()`.
#' @param expert_results The results object from `analyse_experts()`.
#' @param group_mappings A named list mapping numeric codes to labels (e.g., `list('0' = "Male", '1' = "Female")`).
#' @param group_var A string with the column name of the sensitive attribute in the raw_data.
#'
#' @return A ggplot object representing the Sankey diagram.
#' @export
#' @import ggplot2
#' @importFrom dplyr group_by summarise n arrange desc slice pull left_join mutate case_when filter select
#' @importFrom ggalluvial geom_alluvium geom_stratum
#'
plot_sankey <- function(raw_data, gnn_results, expert_results, group_mappings, group_var) {

  cat("--- Generating Sankey Plot ---\n")

  # --- 1. Identify Key Opposing Features from expert analysis ---
  # This version correctly finds the two most extreme features across all pairwise comparisons
  all_diffs <- do.call(rbind, expert_results$pairwise_differences)

  group_A_features <- all_diffs %>%
    dplyr::arrange(difference) %>%
    dplyr::slice(1:2) %>%
    dplyr::pull(feature) %>%
    unique()

  group_B_features <- all_diffs %>%
    dplyr::arrange(dplyr::desc(difference)) %>%
    dplyr::slice(1:2) %>%
    dplyr::pull(feature) %>%
    unique()

  # --- 2. Prepare Data for Plotting ---
  sankey_data_raw <- raw_data %>%
    dplyr::mutate(
      group_A_score = rowSums(dplyr::select(., dplyr::all_of(group_A_features)), na.rm = TRUE),
      group_B_score = rowSums(dplyr::select(., dplyr::all_of(group_B_features)), na.rm = TRUE)
    ) %>%
    dplyr::mutate(
      Feature_Profile = dplyr::case_when(
        group_A_score > group_B_score ~ "Profile A",
        TRUE ~ "Profile B"
      )
    )

  plot_data <- gnn_results$gate_weights %>%
    dplyr::left_join(sankey_data_raw, by = "subjectid") %>%
    dplyr::mutate(
      # This logic assumes a binary expert assignment for visualisation simplicity
      Assigned_Expert = ifelse(gate_prob_expert_1 >= 0.5,
                               paste(group_mappings[[2]], "Expert"),
                               paste(group_mappings[[1]], "Expert")),
      Actual_Group = as.character(group_mappings[as.character(group)])
    ) %>%
    dplyr::filter(!is.na(Actual_Group) & !is.na(Feature_Profile))

  # --- 3. Create the Plot ---
  sankey_flow_counts <- plot_data %>%
    dplyr::group_by(Actual_Group, Feature_Profile, Assigned_Expert) %>%
    dplyr::summarise(N = dplyr::n(), .groups = "drop")

  sankey_plot <- ggplot(sankey_flow_counts,
                        aes(axis1 = Actual_Group, axis2 = Feature_Profile, axis3 = Assigned_Expert, y = N)) +
    ggalluvial::geom_alluvium(aes(fill = Actual_Group), width = 1/8, alpha = 0.7) +
    ggalluvial::geom_stratum(width = 1/8, fill = "grey90", colour = "black") +
    geom_text(stat = "stratum", aes(label = after_stat(stratum)), size = 3.5) +
    scale_x_discrete(limits = c("Actual Group", "Learned Feature Profile", "Assigned Expert")) +
    labs(
      title = "Patient Routing Based on Feature Profile and Subgroup",
      y = "Number of Patients",
      fill = "Actual Group"
    ) +
    theme_minimal()

  return(sankey_plot)
}
