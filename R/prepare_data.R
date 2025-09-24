#' Prepare Data for GNN Training
#'
#' This function takes a raw dataframe, cleans it, defines the outcome and
#' group variables, and scales the feature matrix.
#'
#' @param data A dataframe containing the raw data.
#' @param outcome_var A string with the column name of the binary outcome (must be 0 or 1).
#' @param group_var A string with the column name of the sensitive attribute.
#' @param group_mappings A named list that maps the values in `group_var` to numeric codes (0, 1, 2...). For example, `list("Male" = 0, "Female" = 1)`.
#' @param cols_to_remove A character vector of column names to exclude from the feature matrix (e.g., IDs, highly collinear vars).
#' @importFrom dplyr select any_of where
#' @importFrom magrittr %>%
#'
#' @return A list containing:
#'   \item{X}{The scaled feature matrix.}
#'   \item{y}{The numeric outcome vector.}
#'   \item{group}{The numeric group vector.}
#'   \item{feature_names}{The names of the features used.}
#'   \item{subject_ids}{A vector of subject IDs, if a 'subjectid' column exists.}
#' @export
#' @importFrom dplyr select any_of where
#' @importFrom magrittr %>%
#' @examples
#' # Fictional data example
#' my_data <- data.frame(
#'   subjectid = 1:10,
#'   remission = sample(0:1, 10, replace = TRUE),
#'   gender = sample(c("M", "F"), 10, replace = TRUE),
#'   feature1 = rnorm(10),
#'   feature2 = rnorm(10)
#' )
#'
#' prepared_data <- prepare_data(
#'   data = my_data,
#'   outcome_var = "remission",
#'   group_var = "gender",
#'   group_mappings = list("M" = 0, "F" = 1),
#'   cols_to_remove = c("subjectid")
#' )
prepare_data <- function(data, outcome_var, group_var, group_mappings, cols_to_remove = NULL) {
  # --- Input Validation ---
  if (!outcome_var %in% names(data)) stop("outcome_var not found in data.")
  if (!group_var %in% names(data)) stop("group_var not found in data.")

  # --- Define Outcome (y) ---
  y <- as.numeric(data[[outcome_var]])
  y <- ifelse(is.na(y), 0, y) # Default NA to 0

  # --- Define Group Vector ---
  group_vec_raw <- data[[group_var]]

  # Create a named vector for mapping
  mapping_vector <- unlist(group_mappings)
  names(mapping_vector) <- names(group_mappings)

  # Apply the mapping
  group <- as.numeric(mapping_vector[as.character(group_vec_raw)])

  if (any(is.na(group))) {
    stop("NA values found in group vector after mapping.
         Check that all values in your group_var column are present in the group_mappings list.")
  }

  # --- Define Feature Matrix (X) ---
  all_cols_to_remove <- unique(c(outcome_var, group_var, cols_to_remove))
  X <- data %>%
    dplyr::select(-dplyr::any_of(all_cols_to_remove)) %>%
    dplyr::select(dplyr::where(is.numeric))

  feature_names <- colnames(X)
  X <- scale(X)

  # --- Extract Subject IDs if they exist ---
  subject_ids <- if ("subjectid" %in% names(data)) data$subjectid else NULL

  # --- Return ---
  return(list(
    X = X,
    y = y,
    group = group,
    feature_names = feature_names,
    subject_ids = subject_ids
  ))
}
