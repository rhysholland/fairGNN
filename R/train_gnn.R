#' Train and Evaluate the Gated Neural Network (robust splits + safe ROC)
#'
#' @param prepared_data List from prepare_data(): X, y, group, feature_names, subject_ids
#' @param hyper_grid data.frame with columns: lr, hidden_dim, dropout_rate, lambda, temperature
#' @param num_repeats Integer, repeated train/test splits per combo & final run
#' @param epochs Integer, epochs per run
#' @param output_dir Directory to write csv/rds (defaults to tempdir())
#' @param run_tuning Logical, run hyperparameter search
#' @param best_params data.frame/list with lr, hidden_dim, dropout_rate, lambda, temperature if run_tuning=FALSE
#' @param save_outputs Logical, whether to save outputs to disk (default FALSE)
#' @param seed Optional seed for reproducible data splits. Defaults to NULL to respect the current RNG state.
#' @param verbose Logical, whether to print progress messages (default FALSE)
#' @return list(final_results, gate_weights, expert_weights, performance_summary, aif360_data, tuning_results)
#' @export
#' @import torch
#' @importFrom pROC roc auc
#' @importFrom dplyr bind_rows slice_max select
#' @importFrom readr write_csv
#' @importFrom tibble tibble as_tibble
#' @importFrom utils capture.output
#'
#'
train_gnn <- function(prepared_data, hyper_grid, num_repeats = 20, epochs = 300,
                      output_dir = tempdir(), run_tuning = TRUE, best_params = NULL,
                      save_outputs = FALSE, seed = NULL, verbose = FALSE) {

  # create output dir only if user explicitly requests saving
  if (save_outputs && !dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

  if (!is.null(seed)) {
    if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed)) {
      stop("seed must be a single finite numeric value or NULL.")
    }
    seed <- as.integer(seed)
    if (is.na(seed)) stop("seed must be coercible to an integer.")
  }
  base_seed <- seed
  seed_counter <- -1L
  next_split_seed <- function() {
    if (is.null(base_seed)) return(NULL)
    seed_counter <<- seed_counter + 1L
    base_seed + seed_counter
  }

  # -------- Unpack --------
  X <- prepared_data$X
  y <- as.numeric(prepared_data$y)      # 0/1 numeric
  group <- as.numeric(prepared_data$group)
  feature_names <- prepared_data$feature_names
  all_subjectids <- prepared_data$subject_ids

  if (!is.matrix(X) && !is.data.frame(X)) stop("prepared_data$X must be a matrix/data.frame")
  X <- as.matrix(X)
  if (any(!y %in% c(0,1))) stop("prepared_data$y must be binary 0/1")
  if (length(y) != nrow(X)) stop("Length of y must equal nrow(X)")
  if (length(group) != nrow(X)) stop("Length of group must equal nrow(X)")

  # Drop rows with NA in X/y/group to avoid downstream NA headaches
  if (anyNA(X) || anyNA(y) || anyNA(group)) {
    keep <- complete.cases(X) & !is.na(y) & !is.na(group)
    dropped <- sum(!keep)
    X <- X[keep, , drop = FALSE]
    y <- y[keep]
    group <- group[keep]
    if (!is.null(all_subjectids)) all_subjectids <- all_subjectids[keep]
    if (verbose) message(sprintf("[prepare] Dropped %d rows with NA in X/y/group.", dropped))
  }

  n <- nrow(X); input_dim <- ncol(X)
  num_groups <- length(unique(group))
  if (length(unique(y)) < 2L) stop("Only one class present in y; cannot train a classifier.")

  # -------- Manual stratified split (guaranteed both classes in TEST) --------
  stratified_split_both_in_test <- function(y, test_prop = 0.20, seed = NULL) {
    if (!is.null(seed)) {
      if (!is.numeric(seed) || length(seed) != 1L || !is.finite(seed)) {
        stop("seed must be a single finite numeric value or NULL.")
      }
      seed <- as.integer(seed)
      if (is.na(seed)) stop("seed must be coercible to an integer.")
      restore_rng <- if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
        old_seed <- get(".Random.seed", envir = .GlobalEnv)
        function() assign(".Random.seed", old_seed, envir = .GlobalEnv)
      } else {
        function() {
          if (exists(".Random.seed", envir = .GlobalEnv, inherits = FALSE)) {
            rm(".Random.seed", envir = .GlobalEnv)
          }
        }
      }
      on.exit(restore_rng(), add = TRUE)
      set.seed(seed)

    }
    idx0 <- which(y == 0); idx1 <- which(y == 1)
    stopifnot(length(idx0) > 0L && length(idx1) > 0L)
    n0_test <- max(1L, floor(length(idx0) * test_prop))
    n1_test <- max(1L, floor(length(idx1) * test_prop))
    test <- sort(c(sample(idx0, n0_test), sample(idx1, n1_test)))
    train <- setdiff(seq_along(y), test)
    list(train = train, test = test)
  }

  # -------- Safe AUC wrapper --------
  safe_auc <- function(y_true, probs) {
    keep <- is.finite(probs) & !is.na(y_true)
    y_ <- y_true[keep]
    p_ <- probs[keep]
    if (length(y_) == 0L) return(NA_real_)
    if (length(unique(y_)) < 2L) return(NA_real_)
    ro <- pROC::roc(response = y_, predictor = p_, quiet = TRUE, levels = c(0, 1))
    as.numeric(pROC::auc(ro))
  }

  # -------- Model --------
  gated_model <- nn_module(
    "GatedModel",
    initialize = function(input_dim, hidden_dim, num_groups, dropout_p = 0.5) {
      self$gate_layer <- nn_linear(input_dim, num_groups)
      self$subgroup_layers <- nn_module_list(lapply(seq_len(num_groups), function(i) {
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
      output_logits <- torch_sum(gate_weights * subgroup_logits, 2)
      attr(output_logits, "gate_weights") <- gate_weights
      output_logits
    }
  )

  # -------- Loss --------
  fairness_constrained_loss <- function(y_logits, y_true, groups, lambda = 1.0) {
    bce <- nnf_binary_cross_entropy_with_logits(y_logits, y_true)
    if (lambda == 0) return(bce)
    eps <- 1e-7
    y_prob <- torch_sigmoid(y_logits)
    ug <- sort(unique(as.numeric(groups)))
    soft_tprs <- list(); soft_fprs <- list()
    for (g in ug) {
      m <- (groups == g)
      if (torch_sum(m)$item() > 0) {
        yp <- y_prob[m]; yt <- y_true[m]
        p <- torch_sum(yt); n <- torch_sum(1 - yt)
        soft_tprs[[as.character(g)]] <- torch_sum(yp * yt) / (p + eps)
        soft_fprs[[as.character(g)]] <- torch_sum(yp * (1 - yt)) / (n + eps)
      }
    }
    tpr_v <- torch_var(torch_stack(soft_tprs))
    fpr_v <- torch_var(torch_stack(soft_fprs))
    bce + lambda * (tpr_v + fpr_v)
  }

  # -------- Tuning --------
  tuning_results <- tibble::tibble()
  if (run_tuning) {
    if (missing(hyper_grid) || is.null(hyper_grid) || nrow(hyper_grid) == 0)
      stop("run_tuning=TRUE but hyper_grid is missing/empty.")
    need <- c("lr","hidden_dim","dropout_rate","lambda","temperature")
    if (!all(need %in% names(hyper_grid)))
      stop("hyper_grid must include: lr, hidden_dim, dropout_rate, lambda, temperature")

    if (verbose) message("Starting hyperparameter tuning with ", nrow(hyper_grid), " combinations...")
    for (j in seq_len(nrow(hyper_grid))) {
      p <- hyper_grid[j, , drop = FALSE]
      if (verbose) message(sprintf("Combo %d/%d: LR=%.4g | Hidden=%d | Dropout=%.2f | Lambda=%.2f | Temp=%.2f",
                                   j, nrow(hyper_grid), p$lr, p$hidden_dim, p$dropout_rate, p$lambda, p$temperature))
      aucs <- c()

      for (r in seq_len(num_repeats)) {
        split_seed <- next_split_seed()
        sp <- stratified_split_both_in_test(y, test_prop = 0.20, seed = split_seed)
        tr <- sp$train; te <- sp$test

        Xtr <- X[tr,, drop=FALSE]; Xte <- X[te,, drop=FALSE]
        ytr <- y[tr]; yte <- y[te]
        gtr <- group[tr]

        xtr   <- torch_tensor(as.matrix(Xtr), dtype = torch_float())
        ytr_t <- torch_tensor(as.numeric(ytr), dtype = torch_float())
        gtr_t <- torch_tensor(as.numeric(gtr), dtype = torch_float())
        xte   <- torch_tensor(as.matrix(Xte), dtype = torch_float())

        net <- gated_model(input_dim, p$hidden_dim, num_groups, dropout_p = p$dropout_rate)
        opt <- optim_adam(net$parameters, lr = p$lr)

        for (epoch in seq_len(epochs)) {
          net$train(); opt$zero_grad()
          logits <- net(xtr, temperature_val = p$temperature)
          loss <- fairness_constrained_loss(logits, ytr_t, gtr_t, lambda = p$lambda)
          loss$backward(); opt$step()
        }

        net$eval()
        with_no_grad({ logits_te <- net(xte, temperature_val = p$temperature) })
        probs <- as.numeric(torch_sigmoid(logits_te))
        probs[!is.finite(probs)] <- NA_real_
        probs <- pmin(pmax(probs, 1e-6), 1 - 1e-6)

        aucs <- c(aucs, safe_auc(yte, probs))
      }

      mean_auc <- mean(aucs, na.rm = TRUE)
      tuning_results <- dplyr::bind_rows(tuning_results,
                                         tibble::as_tibble(c(p, mean_auc = mean_auc)))
      if (verbose) message("Mean AUC: ", ifelse(is.finite(mean_auc), sprintf("%.4f", mean_auc), "NA"))
    }
    best_params <- tuning_results %>% dplyr::slice_max(order_by = mean_auc, n = 1)
    if (verbose) { message("Best Performing Combination Found"); utils::capture.output(best_params) |> paste(collapse="\n") |> message() }
  } else {
    if (is.null(best_params)) stop("If run_tuning is FALSE, best_params must be provided.")
    best_params <- as.data.frame(best_params)
    need <- c("lr","hidden_dim","dropout_rate","lambda","temperature")
    if (!all(need %in% names(best_params)))
      stop("best_params must include: lr, hidden_dim, dropout_rate, lambda, temperature")
    if (verbose) { message("Skipping tuning. Using provided parameters."); utils::capture.output(best_params[1, need]) |> paste(collapse="\n") |> message() }
  }

  # -------- Final run with best params --------
  bp <- best_params
  final_lr          <- bp$lr[1]
  final_hidden_dim  <- bp$hidden_dim[1]
  final_dropout     <- bp$dropout_rate[1]
  final_lambda      <- bp$lambda[1]
  final_temperature <- bp$temperature[1]

  results_list <- list(); gate_weights_list <- list(); expert_weights_list <- list()

  for (r in seq_len(num_repeats)) {
    split_seed <- next_split_seed()
    sp <- stratified_split_both_in_test(y, test_prop = 0.20, seed = split_seed)
    tr <- sp$train; te <- sp$test

    Xtr <- X[tr,, drop=FALSE]; Xte <- X[te,, drop=FALSE]
    ytr <- y[tr]; yte <- y[te]
    gtr <- group[tr]; gte <- group[te]
    subj_te <- if (!is.null(all_subjectids)) all_subjectids[te] else seq_along(yte)

    xtr   <- torch_tensor(as.matrix(Xtr), dtype = torch_float())
    ytr_t <- torch_tensor(as.numeric(ytr), dtype = torch_float())
    gtr_t <- torch_tensor(as.numeric(gtr), dtype = torch_float())
    xte   <- torch_tensor(as.matrix(Xte), dtype = torch_float())

    net <- gated_model(input_dim, final_hidden_dim, num_groups, dropout_p = final_dropout)
    opt <- optim_adam(net$parameters, lr = final_lr)

    for (epoch in seq_len(epochs)) {
      net$train(); opt$zero_grad()
      logits <- net(xtr, temperature_val = final_temperature)
      loss <- fairness_constrained_loss(logits, ytr_t, gtr_t, lambda = final_lambda)
      loss$backward(); opt$step()
    }

    net$eval()
    with_no_grad({ logits_te <- net(xte, temperature_val = final_temperature) })
    probs <- as.numeric(torch_sigmoid(logits_te))
    probs[!is.finite(probs)] <- NA_real_
    probs <- pmin(pmax(probs, 1e-6), 1 - 1e-6)

    gw <- attr(logits_te, "gate_weights")
    gw_mat <- as.array(gw)
    gate_entropy <- as.numeric(-rowSums(gw_mat * log(gw_mat + 1e-10)))
    expert_weights_list[[r]] <- lapply(net$subgroup_layers, function(layer) as.numeric(layer[[1]]$weight))

    gate_probs_df <- as.data.frame(gw_mat)
    names(gate_probs_df) <- paste0("gate_prob_expert_", 0:(num_groups - 1))

    results_list[[r]] <- tibble::tibble(
      iteration = r, subjectid = subj_te, true = yte, prob = probs, group = gte
    )
    gate_weights_list[[r]] <- dplyr::bind_cols(
      tibble::tibble(iteration = r, subjectid = subj_te, group = gte, gate_entropy = gate_entropy),
      gate_probs_df
    )
  }

  final_results <- dplyr::bind_rows(results_list)
  gate_weights  <- dplyr::bind_rows(gate_weights_list)

  auc_final <- if (length(unique(final_results$true)) >= 2L) {
    safe_auc(final_results$true, final_results$prob)
  } else NA_real_
  brier_final <- mean((final_results$true - final_results$prob)^2)

  performance_summary <- tibble::tibble(
    Metric = c("AUC", "Brier Score"),
    Value  = c(auc_final, brier_final)
  )

  aif360_data <- final_results %>%
    dplyr::select(subjectid, true_label = true, predicted_prob = prob, sensitive_attr_numeric = group)

  if (save_outputs) {
    readr::write_csv(final_results, file.path(output_dir, "gnn_final_predictions.csv"))
    readr::write_csv(gate_weights,  file.path(output_dir, "gnn_gate_weights.csv"))
    readr::write_csv(aif360_data,   file.path(output_dir, "gnn_aif360_data.csv"))
    if (run_tuning) readr::write_csv(tuning_results, file.path(output_dir, "gnn_tuning_results.csv"))
    saveRDS(expert_weights_list, file.path(output_dir, "gnn_expert_weights.rds"))
  }

  list(
    final_results = final_results,
    gate_weights = gate_weights,
    expert_weights = expert_weights_list,
    performance_summary = performance_summary,
    aif360_data = aif360_data,
    tuning_results = if (run_tuning) tuning_results else "Tuning was skipped"
  )
}
