#' Create a Sankey Plot to Visualise Patient Routing
#'
#' @param raw_data The original, unscaled dataframe (must contain the features referenced by expert analysis).
#' @param gnn_results The results object from `train_gnn()` (uses $final_results and $gate_weights).
#' @param expert_results The results object from `analyse_experts()`.
#' @param group_mappings A named list or named character vector mapping codes to labels (e.g., list('0'="White",'1'="Non-White")).
#' @param group_var A string with the column name of the sensitive attribute in the raw_data.
#' @param verbose Logical, whether to print progress messages (default FALSE).
#' @return A ggplot object representing the Sankey diagram.
#' @export
#' @import ggplot2
#' @import ggalluvial
#' @importFrom dplyr group_by summarise n arrange desc slice pull left_join mutate filter select rename
#' @importFrom ggalluvial geom_alluvium geom_stratum stat_stratum
#' @import ggplot2
#' @import ggalluvial
#' @importFrom dplyr group_by summarise n arrange desc slice pull left_join mutate filter select rename
#' @importFrom ggalluvial geom_alluvium geom_stratum stat_stratum
#'
plot_sankey <- function(raw_data, gnn_results, expert_results, group_mappings, group_var, verbose = FALSE) {

  if (verbose) message("Generating Sankey Plot...")

  # -- Ensure we have a subject id in raw_data (subjectid preferred; else id) --
  if ("subjectid" %in% names(raw_data)) {
    raw_keyed <- raw_data
  } else if ("id" %in% names(raw_data)) {
    raw_keyed <- dplyr::rename(raw_data, subjectid = id)
  } else {
    stop("plot_sankey(): raw_data must contain either a 'subjectid' or an 'id' column.")
  }

  # -- Identify two most “opposite” features (binary case expected) --
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

  haveA <- group_A_features[group_A_features %in% names(raw_keyed)]
  haveB <- group_B_features[group_B_features %in% names(raw_keyed)]
  if (length(haveA) == 0L && length(haveB) == 0L) {
    stop("plot_sankey(): none of the selected opposing features were found in raw_data.")
  }

  # -- Build feature-profile table from RAW DATA --
  sankey_data_raw <- raw_keyed %>%
    dplyr::mutate(
      group_A_score = if (length(haveA)) rowSums(dplyr::select(., dplyr::all_of(haveA)), na.rm = TRUE) else 0,
      group_B_score = if (length(haveB)) rowSums(dplyr::select(., dplyr::all_of(haveB)), na.rm = TRUE) else 0,
      Feature_Profile = ifelse(group_A_score > group_B_score, "Profile A", "Profile B")
    ) %>%
    dplyr::select(subjectid, Feature_Profile, dplyr::all_of(group_var))

  # -- Subject ids and group labels from model outputs --
  if (is.null(gnn_results$final_results) || !"subjectid" %in% names(gnn_results$final_results)) {
    stop("plot_sankey(): gnn_results$final_results must contain 'subjectid'.")
  }
  final_preds <- gnn_results$final_results %>%
    dplyr::select(subjectid, group)

  # ==== SAFE GROUP LABEL MAPPING — NO case_when ====
  final_preds$group_chr <- as.character(final_preds$group)
  gm <- as.character(unlist(group_mappings, use.names = TRUE))
  names(gm) <- names(group_mappings)

  if (is.null(names(gm)) || anyNA(names(gm))) {
    unique_codes <- sort(unique(final_preds$group_chr))
    names(gm) <- unique_codes[seq_along(gm)]
  }

  mapped <- gm[final_preds$group_chr]
  mapped[is.na(mapped)] <- final_preds$group_chr[is.na(mapped)]
  final_preds$Actual_Group <- mapped
  # ================================================

  # -- Join by subjectid (gate weights + final preds + raw feature profile) --
  gw <- gnn_results$gate_weights
  if (is.null(gw) || !"subjectid" %in% names(gw)) {
    stop("plot_sankey(): Could not find 'subjectid' in gnn_results$gate_weights. Ensure train_gnn() saves subject IDs.")
  }

  sankey_joined <- gw %>%
    dplyr::select(subjectid, dplyr::starts_with("gate_prob_expert_"), gate_entropy) %>%
    dplyr::left_join(final_preds[, c("subjectid","Actual_Group")], by = "subjectid") %>%
    dplyr::left_join(sankey_data_raw, by = "subjectid") %>%
    dplyr::filter(!is.na(Actual_Group), !is.na(Feature_Profile))

  # -- Assign expert by max gate probability (binary simplifies to >= 0.5) --
  prob_cols <- grep("^gate_prob_expert_", names(sankey_joined), value = TRUE)
  if (length(prob_cols) < 2L) stop("Sankey requires at least two experts (found ", length(prob_cols), ").")

  if (length(prob_cols) == 2L) {
    assigned <- ifelse(sankey_joined[[prob_cols[2]]] >= 0.5, 2L, 1L)
  } else {
    assigned <- apply(sankey_joined[, prob_cols, drop = FALSE], 1, function(r) which.max(r))
  }

  # Expert labels using mapping if available; else “Expert k”
  expert_labels <- {
    labs <- unname(as.character(unlist(group_mappings, use.names = FALSE)))
    if (!length(labs)) paste("Expert", seq_along(prob_cols))
    else if (length(labs) >= length(prob_cols)) labs[seq_along(prob_cols)]
    else paste("Expert", seq_along(prob_cols))
  }

  sankey_plot_data <- sankey_joined %>%
    dplyr::mutate(Assigned_Expert = paste0(expert_labels[assigned], " Expert"))

  # -- Count flows and plot --
  sankey_flow_counts <- sankey_plot_data %>%
    dplyr::group_by(Actual_Group, Feature_Profile, Assigned_Expert) %>%
    dplyr::summarise(N = dplyr::n(), .groups = "drop")

  p <- ggplot(
    sankey_flow_counts,
    aes(axis1 = Actual_Group, axis2 = Feature_Profile, axis3 = Assigned_Expert, y = N)
  ) +
    ggalluvial::geom_alluvium(aes(fill = Actual_Group), width = 1/8, alpha = 0.7) +
    ggalluvial::geom_stratum(width = 1/8, fill = "grey90", colour = "black") +
    ggalluvial::stat_stratum(geom = "text", aes(label = after_stat(stratum)), size = 3.5) +
    scale_x_discrete(limits = c("Actual Group", "Learned Feature Profile", "Assigned Expert")) +
    labs(
      title = "Patient Routing Based on Feature Profile and Subgroup",
      y = "Number of Patients",
      fill = "Actual Group"
    ) +
    theme_minimal()

  return(p)
}
