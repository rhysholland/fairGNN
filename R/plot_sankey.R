#' Create a Sankey Plot to Visualise Patient Routing
#'
#' @param raw_data The original, unscaled dataframe (must contain the features referenced by expert analysis).
#' @param gnn_results The results object from `train_gnn()` (uses $final_results and $gate_weights).
#' @param expert_results The results object from `analyse_experts()`.
#' @param group_mappings A named list or named character vector mapping *codes* to *labels*
#'   (e.g., c("0"="Male","1"="Female")). If provided in the reverse orientation (labels→codes),
#'   or unnamed, this function will normalise it automatically.
#' @param group_var A string with the column name of the sensitive attribute in the raw_data.
#' @param verbose Logical, whether to print progress messages (default FALSE).
#' @return A ggplot object representing the Sankey diagram.
#' @export
#' @import ggplot2
#' @import ggalluvial
#' @importFrom dplyr group_by summarise n arrange desc slice pull left_join mutate filter select rename all_of
#' @importFrom stats setNames
#'
plot_sankey <- function(raw_data, gnn_results, expert_results, group_mappings, group_var, verbose = FALSE) {

  if (verbose) message("Generating Sankey Plot...")

  # -------- helpers --------
  coerce_chr <- function(x) {
    if (is.factor(x)) as.character(x) else as.character(x)
  }

  # Normalise any mapping (code<->label) to a *code->label* character named vector
  normalize_mapping <- function(codes_chr, mapping) {
    # Accept list or vector; flatten
    flat <- unlist(mapping, use.names = TRUE)
    # If unnamed, assume order corresponds to sorted unique codes
    if (is.null(names(flat))) {
      u <- sort(unique(codes_chr))
      flat <- setNames(coerce_chr(flat), u[seq_along(flat)])
      return(flat)
    }
    # We have names and values; decide orientation by overlap with codes
    nm  <- coerce_chr(names(flat))
    val <- coerce_chr(unname(flat))

    # Case 1: names look like codes -> already code->label
    if (any(nm %in% codes_chr) && !any(val %in% codes_chr)) {
      return(setNames(val, nm))
    }
    # Case 2: values look like codes -> invert (labels->codes given)
    if (!any(nm %in% codes_chr) && any(val %in% codes_chr)) {
      inv <- setNames(nm, val)
      return(inv)
    }
    # Case 3: both/none overlap (ambiguous). Prefer names as codes; if missing, supplement via inversion.
    out <- setNames(val, nm)
    missing <- setdiff(codes_chr, names(out))
    if (length(missing)) {
      inv <- setNames(nm, val)
      sup <- inv[missing]
      out <- c(out, sup[!is.na(sup)])
    }
    # Ensure all codes have some label (identity fallback)
    still_missing <- setdiff(codes_chr, names(out))
    if (length(still_missing)) {
      out <- c(out, setNames(still_missing, still_missing))
    }
    out
  }

  # Derive expert labels from gate means if mapping isn't 1:1
  derive_expert_labels <- function(df, prob_cols, group_col = "Actual_Group") {
    levs <- levels(df[[group_col]])
    if (is.null(levs)) levs <- unique(as.character(df[[group_col]]))

    means <- sapply(prob_cols, function(pc) tapply(df[[pc]], df[[group_col]], mean, na.rm = TRUE))
    if (is.null(dim(means))) means <- matrix(means, nrow = length(levs), dimnames = list(levs, prob_cols))
    if (is.null(rownames(means))) rownames(means) <- levs

    base <- rownames(means)[apply(means, 2, function(col) { i <- which.max(col); ifelse(length(i), i[1], NA_integer_) })]
    base[is.na(base)] <- paste("Expert", seq_along(base))
    base <- make.unique(base, sep = " ")  # handle ties/duplicates deterministically
    paste0(base, " Expert")
  }
  # -------------------------

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

  # ==== Normalised GROUP LABEL mapping (robust to orientation/types) ====
  final_preds$group_chr <- coerce_chr(final_preds$group)
  codes_present <- sort(unique(final_preds$group_chr))
  gm_code_to_label <- normalize_mapping(codes_present, group_mappings)

  # map; default to code if no label found
  mapped <- unname(gm_code_to_label[final_preds$group_chr])
  mapped[is.na(mapped)] <- final_preds$group_chr[is.na(mapped)]
  final_preds$Actual_Group <- mapped

  # lock factor order: prefer mapping order, else data order
  desired_order <- unique(unname(gm_code_to_label[codes_present]))
  desired_order[is.na(desired_order)] <- codes_present[is.na(desired_order)]
  final_preds$Actual_Group <- factor(final_preds$Actual_Group, levels = unique(desired_order))
  # ======================================================================

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

  # ---------- Expert labels from resolved subgroup labels ----------
  # Primary attempt: use mapping if it cleanly covers experts 1:1
  candidate <- unique(as.character(final_preds$Actual_Group))
  # keep candidate order aligned to mapping order where possible
  candidate <- desired_order[desired_order %in% candidate]
  use_mapping <- length(candidate) == length(prob_cols) && all(!is.na(candidate))

  if (use_mapping) {
    expert_labels <- paste0(make.unique(candidate, sep = " "), " Expert")
  } else {
    # Fallback: data-driven naming by group focus
    expert_labels <- derive_expert_labels(sankey_joined, prob_cols, group_col = "Actual_Group")
  }

  sankey_plot_data <- sankey_joined %>%
    dplyr::mutate(Assigned_Expert = expert_labels[assigned])
  # -----------------------------------------------------------------

  # -- Count flows and plot --
  sankey_flow_counts <- sankey_plot_data %>%
    dplyr::group_by(Actual_Group, Feature_Profile, Assigned_Expert) %>%
    dplyr::summarise(N = dplyr::n(), .groups = "drop")

  # lock orders for readability
  sankey_flow_counts$Feature_Profile <- factor(sankey_flow_counts$Feature_Profile, levels = c("Profile A","Profile B"))
  sankey_flow_counts$Assigned_Expert <- factor(sankey_flow_counts$Assigned_Expert, levels = unique(expert_labels))

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
