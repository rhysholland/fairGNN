if (getRversion() >= "2.15.1")  utils::globalVariables(
  c("group", "feature", "importance", "avg_importance", "difference",
    "reorder", "true", "specificity", "sensitivity", "prob", "prob_bin",
    "mean_predicted_prob", "observed_proportion", "gate_prob_expert_1",
    "group_label", "gate_entropy", "Actual_Group", "Feature_Profile",
    "Assigned_Expert", "N", "stratum", "subjectid", "self", ".")
)
#' @keywords internal
#' @importFrom stats complete.cases
NULL

utils::globalVariables(c("id", "group_A_score", "group_B_score"))
