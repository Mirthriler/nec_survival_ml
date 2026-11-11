# =============================================================================
# SEER colorectal neuroendocrine carcinoma: core survival machine learning
# =============================================================================
# Input: 00_cleaned_analysis_cohort.csv exported by the original SEER pipeline.
# This file starts AFTER cohort selection, clinical recoding and OS/CSS derivation.
# It does not accept a raw SEER export or redistribute patient-level data.
#
# Command line:
#   Rscript seer_nec_survival_ml.R data/00_cleaned_analysis_cohort.csv results
# R / RStudio:
#   source("seer_nec_survival_ml.R")
#   main(input_file = "data/00_cleaned_analysis_cohort.csv", output_dir = "results")
#
# Install dependencies once in R (not executed automatically):
#   install.packages(c("data.table", "survival", "glmnet", "mlr3",
#     "mlr3pipelines", "mboost", "CoxBoost", "ranger", "partykit", "xgboost",
#     "gbm", "survAUC", "pracma", "coin", "sandwich"))
#   install.packages(c("mlr3proba", "mlr3extralearners"), repos = c(
#     mlrorg = "https://mlr-org.r-universe.dev", CRAN = "https://cloud.r-project.org"))
# Optional explanations:
#   install.packages(c("fastshap", "shapviz", "ggplot2"))
#
# Required columns:
#   patient_id, year_dx, surv_months, os_event, css_event,
#   and all predictors in CONFIG$predictors below.
# Values: time in months (>0); events 0=censored, 1=event; css_event may be NA.
# True SEER zero-month values must already have been recoded to 0.5 upstream.
# Unknown clinical categories must be explicit strings; no imputation is done.
# One eligible record per patient is required. patient_id is never a predictor.
#
# Analysis retained from the supplied code:
#   Development: 2004-2017; temporal validation: 2018-2022; endpoints: OS and CSS.
#   Cox-LASSO on the full development cohort -> fixed original predictors ->
#   eight survival models -> 5-fold development CV / inner grid tuning ->
#   final development fit -> untouched temporal validation.
#   Grade and regional node counts are excluded from the primary predictors.
#
# Interpretation:
# - Feature selection precedes outer CV. Development CV is conditional on the
#   preselected predictors and can be optimistic; it is NOT fully nested CV.
#   Temporal validation is the primary independent evaluation.
# - CSS uses the upstream endpoint definition; other-cause death is censored.
#   This is a cause-specific survival analysis, not cumulative-incidence modeling.
# - Treatment predictors reflect recorded treatment; this is not a causal model
#   or a model restricted to information available at diagnosis.
# - C-index, integrated Uno AUC and integrated Brier score are reported. Training
#   follow-up quantiles define evaluation horizons; inspect support in the later
#   cohort before interpreting long-horizon scores. Fold SDs are descriptive.
# - For score-only models, the original Kaplan baseline + PH distribution
#   compositor is retained for Brier scoring; this is a modeling assumption.
# - Models outside CONFIG$tune_grids use package defaults, recorded in outputs.
#   No equivalence to previously computed numeric results is asserted.
#
# Engineering changes: portable paths; explicit dependencies and input checks;
# deterministic fold seeds; exact dummy-to-predictor mapping; failed tuning folds
# cannot win; individual metric errors are logged; SHAP uses the evaluated GBM
# instead of fitting a second GBM with unrelated parameters.
#
# Output: aggregate CSVs, LASSO plots, configuration, sessionInfo and input hash.
# Optional SHAP uses development data only and explains the GBM log-risk score.
# Individual-level exports and serialized models are OFF by default. Keep local
# data/, results/, *.rds and .RData out of a public repository.
#
# API references:
# https://mlr3extralearners.mlr-org.com/
# https://mlr3proba.mlr-org.com/reference/mlr_measures_surv.graf.html
# https://mlr3proba.mlr-org.com/reference/mlr_measures_surv.uno_auc.html
# https://mlr3proba.mlr-org.com/reference/mlr_graphs_distrcompositor.html
# https://bgreenwell.github.io/fastshap/reference/explain.html
# =============================================================================

# 1. Configuration -------------------------------------------------------------

CONFIG <- list(
  seed = 20260910L,
  start_year = 2004L, dev_end_year = 2017L,
  test_start_year = 2018L, end_year = 2022L,
  endpoints = c("OS", "CSS"),
  predictors = c("age", "sex", "race", "marital_group", "site_group",
                 "histology", "t_stage", "n_stage", "m_stage", "tumor_size_group",
                 "surgery", "radiation", "chemotherapy", "income_group", "rural_group"),
  do_lasso = TRUE, lasso_lambda = "lambda.1se", lasso_folds = 10L,
  outer_folds = 5L, inner_folds = 5L, do_tuning = TRUE,
  explain_gbm = FALSE, shap_nsim = 100L, shap_max_n = 2000L,
  save_models = FALSE, export_individual_shap = FALSE,
  tune_grids = list(
    RSF = list(mtry.ratio = c(0.2, 0.4, 0.6, 0.8),
               min.node.size = c(5L, 15L, 30L), num.trees = c(800L, 1200L)),
    GBM = list(n.trees = c(500L, 1000L), interaction.depth = c(1L, 3L),
               shrinkage = c(0.01, 0.05), n.minobsinnode = c(5L, 10L)),
    XGBoost = list(nrounds = c(300L, 600L), eta = c(0.03, 0.10),
                   max_depth = c(2L, 4L), subsample = c(0.8, 1.0),
                   colsample_bytree = c(0.8, 1.0)),
    Ctree = list(maxdepth = c(3L, 5L), mincriterion = c(0.95, 0.99),
                 minbucket = c(20L, 40L, 80L))
  )
)

# Fixed clinical level definitions from the supplied preprocessing code.
# Sex and race levels are learned from the development cohort only.
FACTOR_LEVELS <- list(
  marital_group = c("Married", "Single", "Divorced/Separated", "Widowed", "Unknown/Other"),
  site_group = c("Right colon", "Transverse colon", "Left colon",
                 "Colon overlapping/NOS", "Rectosigmoid", "Rectum", "Unknown"),
  histology = c("NEC, NOS (8246)", "Large-cell NEC (8013)",
                "Small-cell carcinoma (8041-8045)"),
  t_stage = c("T0", "Tis", "T1", "T2", "T3", "T4", "TX/Unknown"),
  n_stage = c("N0", "N1", "N2", "N3", "NX/Unknown"),
  m_stage = c("M0", "M1", "MX/Unknown"),
  tumor_size_group = c("<=20 mm", "21-50 mm", ">50 mm", "Unknown"),
  surgery = c("No", "Yes", "Unknown"),
  radiation = c("No/Unknown", "Yes"), chemotherapy = c("No/Unknown", "Yes"),
  income_group = c("<$60k", "$60-79k", "$80-99k", ">=$100k", "Unknown"),
  rural_group = c("Metropolitan", "Nonmetropolitan", "Unknown"),
  grade_group = c("Grade I", "Grade II", "Grade III", "Grade IV", "Unknown/unavailable"),
  nodes_examined_group = c("0", "1-11", ">=12", "Unknown"),
  nodes_positive_group = c("0", "1-3", ">=4", "Not examined", "Unknown")
)

# 2. Dependencies and input validation -----------------------------------------

require_packages <- function(packages) {
  missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing)) stop("Install required packages: ", paste(missing, collapse = ", "),
                            ". See installation instructions at the top of this script.")
}

read_cohort <- function(path, cfg) {
  if (!file.exists(path)) stop("Input file does not exist: ", path)
  if (cfg$start_year > cfg$dev_end_year || cfg$test_start_year > cfg$end_year ||
      cfg$test_start_year != cfg$dev_end_year + 1L) stop("Invalid temporal split settings.")
  if (!length(cfg$endpoints) || any(!cfg$endpoints %in% c("OS", "CSS"))) {
    stop("endpoints must contain OS and/or CSS.")
  }
  prohibited <- c("patient_id", "year_dx", "split", "surv_months", "os_event",
                 "css_event", "time", "event", "vital_raw", "other_cod_raw", "cod_site_raw")
  if (!length(cfg$predictors) || anyDuplicated(cfg$predictors) ||
      length(intersect(cfg$predictors, prohibited))) stop("Invalid predictor list.")
  required <- unique(c("patient_id", "year_dx", "surv_months",
                       tolower(paste0(cfg$endpoints, "_event")), cfg$predictors))
  dat <- data.table::fread(path, encoding = "UTF-8", colClasses = list(character = "patient_id"))
  if (length(setdiff(required, names(dat)))) {
    stop("Missing input columns: ", paste(setdiff(required, names(dat)), collapse = ", "))
  }
  dat[, year_dx := suppressWarnings(as.numeric(year_dx))]
  if (any(!is.finite(dat$year_dx) | dat$year_dx != floor(dat$year_dx))) stop("Invalid year_dx.")
  if (any(dat$year_dx < cfg$start_year | dat$year_dx > cfg$end_year)) {
    stop("Input contains years outside the configured study period; filter upstream.")
  }
  if (!nrow(dat) || anyNA(dat$patient_id) || any(!nzchar(trimws(dat$patient_id))) ||
      anyDuplicated(dat$patient_id)) stop("Input must contain one nonblank ID per patient.")
  dat[, split := ifelse(year_dx <= cfg$dev_end_year, "Development", "TemporalValidation")]
  if (!all(c("Development", "TemporalValidation") %in% dat$split)) stop("Empty temporal cohort.")
  dat[, surv_months := suppressWarnings(as.numeric(surv_months))]
  if (any(!is.finite(dat$surv_months) | dat$surv_months <= 0)) {
    stop("surv_months must be positive and complete; do not impute missing survival time.")
  }
  for (ep in cfg$endpoints) {
    column <- tolower(paste0(ep, "_event"))
    value <- as.character(dat[[column]])
    if (any(!is.na(value) & !value %in% c("0", "1")) ||
        (ep == "OS" && anyNA(value))) stop("Invalid event coding: ", column)
    data.table::set(dat, j = column, value = as.integer(value))
  }
  for (column in cfg$predictors) {
    value <- dat[[column]]
    if (anyNA(value) || any(trimws(as.character(value)) == "")) {
      stop("Missing predictor: ", column, ". Recode or impute upstream using training data only.")
    }
    if (column == "age") {
      value <- suppressWarnings(as.numeric(value))
      if (any(!is.finite(value) | value < 0)) stop("Invalid age.")
    } else {
      value <- as.character(value)
      levels <- FACTOR_LEVELS[[column]]
      if (is.null(levels)) levels <- sort(unique(value[dat$split == "Development"]))
      if (any(!value %in% levels)) stop("Unrecognized or validation-only categories in ", column,
                                       "; define the coding upstream before modeling.")
      value <- factor(value, levels = levels)
    }
    data.table::set(dat, j = column, value = value)
  }
  dat
}

make_task <- function(dat, id) {
  mlr3proba::TaskSurv$new(id, backend = data.table::copy(dat), time = "time", event = "event")
}

# 3. Model definitions ---------------------------------------------------------

make_models <- function() {
  spec <- list(Coxnet = c("surv.glmnet", "lp", "encode"),
               Blackboost = c("surv.blackboost", "lp", "plain"),
               Coxboost = c("surv.coxboost", "lp", "encode"),
               RSF = c("surv.ranger", "crank", "plain"),
               Ctree = c("surv.ctree", "crank", "plain"),
               XGBoost = c("surv.xgboost.cox", "lp", "numeric"),
               GBM = c("surv.gbm", "lp", "numeric"),
               Coxph = c("surv.coxph", "lp", "plain"))
  build <- function(s, type = s[2]) {
    base <- mlr3::lrn(s[1], predict_type = type)
    require_packages(base$packages)
    if (s[3] == "plain") return(base)
    graph <- mlr3pipelines::po("encode", method = "one-hot")
    if (s[3] == "numeric") {
      graph <- graph %>>% mlr3pipelines::po("colapply", applicator = as.numeric)
    }
    mlr3::as_learner(graph %>>% base)
  }
  lapply(spec, function(s) {
    risk <- build(s)
    distribution <- if (s[1] %in% c("surv.ranger", "surv.ctree")) {
      build(s, "distr")
    } else {
      mlr3::as_learner(mlr3pipelines::ppl("distrcompositor", learner = build(s),
        estimator = "kaplan", form = "ph", overwrite = TRUE, graph_learner = TRUE))
    }
    list(risk = risk, distribution = distribution)
  })
}

set_params <- function(learner, params) {
  ids <- learner$param_set$ids()
  for (name in names(params)) {
    hit <- if (name %in% ids) name else ids[endsWith(ids, paste0(".", name))]
    if (length(hit) != 1L) stop("Unsupported/ambiguous parameter: ", name,
                               " for learner ", learner$id)
    learner$param_set$values[[hit]] <- params[[name]]
  }
  invisible(learner)
}

clone_model <- function(model, params = list()) {
  out <- lapply(model, function(x) x$clone(deep = TRUE))
  for (learner in out) set_params(learner, params)
  out
}

as_risk_prediction <- function(prediction, task) {
  lp <- prediction$lp
  crank <- prediction$crank
  if (!is.null(lp) && !is.null(crank)) return(prediction)
  # Rank-based measures can also use a mortality ranking from RSF / Ctree.
  score <- if (!is.null(lp)) lp else crank
  if (is.null(score) || any(!is.finite(score))) stop("Missing/nonfinite model risk predictions.")
  mlr3proba::PredictionSurv$new(task = task, row_ids = prediction$row_ids,
    truth = prediction$truth, lp = as.numeric(score), crank = as.numeric(score))
}

# 4. Cox-LASSO selection, once in the full development cohort --------------------

select_predictors <- function(dat, predictors, cfg, seed) {
  # Remove constants using development data only, even when selection is disabled.
  varying <- vapply(dat[, ..predictors], function(x) length(unique(x)) > 1L, logical(1))
  features <- predictors[varying]
  if (!length(features)) stop("No variable predictors in the development cohort.")
  empty <- data.table::data.table(encoded_feature = character(), coefficient = numeric())
  result <- list(selected = features, lambda = NA_real_, lambda_min = NA_real_,
    lambda_1se = NA_real_, fallback = "lasso_disabled", coefficients = empty, cvfit = NULL,
    removed_constants = predictors[!varying])
  if (!cfg$do_lasso) return(result)
  choice <- match.arg(cfg$lasso_lambda, c("lambda.1se", "lambda.min"))
  formula <- stats::reformulate(features, intercept = FALSE)
  x <- stats::model.matrix(formula, data = as.data.frame(dat[, ..features]),
                           na.action = stats::na.fail)
  # The assign attribute maps encoded columns exactly; prefix matching is unsafe.
  origin <- features[attr(x, "assign")]
  keep <- apply(x, 2, function(z) length(unique(z)) > 1L)
  x <- x[, keep, drop = FALSE]
  origin <- origin[keep]
  if (ncol(x) < 2L) {
    result$fallback <- "all_variable_predictors_fewer_than_two_encoded_columns"
    return(result)
  }
  if (cfg$lasso_folds < 3L || nrow(dat) < cfg$lasso_folds * 2L ||
      sum(dat$event) < cfg$lasso_folds) stop("Too few observations/events for LASSO folds.")
  set.seed(seed)
  foldid <- sample(rep(seq_len(cfg$lasso_folds), length.out = nrow(dat)))
  fit <- glmnet::cv.glmnet(x, survival::Surv(dat$time, dat$event), family = "cox",
    alpha = 1, foldid = foldid, nfolds = cfg$lasso_folds, standardize = TRUE)
  lambda <- fit[[choice]]
  beta <- as.matrix(stats::coef(fit, s = lambda))
  fallback <- "none"
  if (!any(beta != 0) && choice == "lambda.1se") {
    alternative <- as.matrix(stats::coef(fit, s = fit$lambda.min))
    if (any(alternative != 0)) {
      lambda <- fit$lambda.min
      beta <- alternative
      fallback <- "lambda.min"
    }
  }
  selected <- unique(origin[as.numeric(beta) != 0])
  if (!length(selected)) {
    selected <- features
    fallback <- "all_variable_predictors_no_nonzero_lasso"
  }
  result$selected <- selected
  result$lambda <- lambda
  result$lambda_min <- fit$lambda.min
  result$lambda_1se <- fit$lambda.1se
  result$fallback <- fallback
  result$cvfit <- fit
  result$coefficients <- data.table::data.table(encoded_feature = rownames(beta),
    coefficient = as.numeric(beta))[coefficient != 0][order(-abs(coefficient))]
  result
}

save_lasso <- function(selection, directory, endpoint) {
  data.table::fwrite(data.table::data.table(endpoint = endpoint, predictor = selection$selected,
    lambda_used = selection$lambda, lambda_min = selection$lambda_min,
    lambda_1se = selection$lambda_1se, fallback = selection$fallback),
    file.path(directory, "01_selected_predictors.csv"))
  data.table::fwrite(selection$coefficients, file.path(directory, "02_lasso_coefficients.csv"))
  if (is.null(selection$cvfit)) return(invisible(NULL))
  draw <- function(path, expression) {
    grDevices::png(path, width = 2000, height = 1600, res = 220)
    on.exit(grDevices::dev.off())
    force(expression)
    graphics::abline(v = log(selection$lambda), lty = 2)
  }
  draw(file.path(directory, "03_lasso_cv.png"), plot(selection$cvfit))
  draw(file.path(directory, "04_lasso_path.png"),
       plot(selection$cvfit$glmnet.fit, xvar = "lambda", label = FALSE))
}

# 5. Inner CV grid search ------------------------------------------------------

make_cv <- function(task, folds, seed) {
  if (folds < 2L || task$nrow < 2L * folds || sum(task$data(cols = "event")[[1]]) < folds) {
    stop("Too few observations/events for the requested CV folds.")
  }
  set.seed(seed)
  cv <- mlr3::rsmp("cv", folds = folds)
  cv$instantiate(task)
  cv
}

tune_model <- function(name, model, task, train_ids, cfg, seed) {
  grid <- cfg$tune_grids[[name]]
  if (!cfg$do_tuning || is.null(grid)) return(list(params = list(), log = NULL))
  candidates <- expand.grid(grid, KEEP.OUT.ATTRS = FALSE, stringsAsFactors = FALSE)
  inner <- make_task(task$data(rows = train_ids), paste0("inner_", name))
  cv <- make_cv(inner, cfg$inner_folds, seed)
  logs <- vector("list", nrow(candidates))
  best <- -Inf
  best_params <- NULL
  for (k in seq_len(nrow(candidates))) {
    params <- as.list(candidates[k, , drop = FALSE])
    # Validate parameter names for both risk and distribution pipelines up front.
    candidate <- clone_model(model, params)
    scores <- rep(NA_real_, cv$iters)
    errors <- rep(NA_character_, cv$iters)
    for (j in seq_len(cv$iters)) {
      scores[j] <- tryCatch({
        set.seed(seed + j)
        learner <- candidate$risk$clone(deep = TRUE)
        learner$train(inner, row_ids = cv$train_set(j))
        pred <- as_risk_prediction(learner$predict(inner, row_ids = cv$test_set(j)), inner)
        value <- as.numeric(pred$score(mlr3::msr("surv.cindex"), task = inner))
        if (!is.finite(value)) stop("Nonfinite C-index.")
        value
      }, error = function(e) { errors[j] <<- conditionMessage(e); NA_real_ })
    }
    # A candidate must succeed on every inner fold to be eligible.
    score <- if (all(is.finite(scores))) mean(scores) else NA_real_
    logs[[k]] <- data.table::data.table(model = name, candidate = k,
      mean_inner_cindex = score, successful_folds = sum(is.finite(scores)),
      params = paste(names(params), unlist(params), sep = "=", collapse = "; "),
      error = paste(unique(stats::na.omit(errors)), collapse = " | "))
    if (is.finite(score) && score > best) { best <- score; best_params <- params }
  }
  log <- data.table::rbindlist(logs)
  list(params = best_params, log = log)
}

# 6. Evaluation ---------------------------------------------------------------

fit_and_score <- function(name, model, task, train_ids, test_ids, seed) {
  errors <- character()
  attempt <- function(label, expression) {
    tryCatch(force(expression), error = function(e) {
      errors <<- c(errors, paste0(label, ": ", conditionMessage(e)))
      NULL
    })
  }
  trained <- clone_model(model)
  fit_predict <- function(learner) {
    set.seed(seed)
    learner$train(task, row_ids = train_ids)
    learner$predict(task, row_ids = test_ids)
  }
  risk <- attempt("risk_model", as_risk_prediction(fit_predict(trained$risk), task))
  distr <- attempt("distribution_model", fit_predict(trained$distribution))
  times <- task$data(rows = train_ids, cols = "time")[[1]]
  auc_times <- sort(unique(as.numeric(stats::quantile(times, seq(0.1, 0.9, 0.1)))))
  brier_max <- as.numeric(stats::quantile(times, 0.9))
  metric <- function(label, pred, measure) {
    if (is.null(pred)) return(NA_real_)
    result <- attempt(label, {
      value <- as.numeric(pred$score(measure, task = task, train_set = train_ids))
      if (length(value) != 1L || !is.finite(value)) stop("Nonfinite metric.")
      value
    })
    if (is.null(result)) NA_real_ else result
  }
  cindex <- metric("cindex", risk, mlr3::msr("surv.cindex"))
  auc <- metric("uno_auc", risk, mlr3::msr("surv.uno_auc", integrated = TRUE, times = auc_times))
  brier <- metric("brier", distr, mlr3::msr("surv.graf", t_max = brier_max))
  heldout_max <- max(task$data(rows = test_ids, cols = "time")[[1]])
  list(scores = data.table::data.table(model = name, cindex = cindex,
    uno_auc = auc, brier = brier, auc_times_months = paste(auc_times, collapse = ";"),
    brier_max_months = brier_max, heldout_max_months = heldout_max,
    horizon_exceeds_heldout_followup = max(auc_times, brier_max) > heldout_max,
    error = if (length(errors)) paste(errors, collapse = " | ") else NA_character_),
    model = trained, risk_ok = !is.null(risk), distribution_ok = !is.null(distr))
}

failed_score <- function(name, error) {
  data.table::data.table(model = name, cindex = NA_real_, uno_auc = NA_real_,
                         brier = NA_real_, error = error)
}

summarize_cv <- function(scores, folds) {
  average <- function(x) if (any(is.finite(x))) mean(x[is.finite(x)]) else NA_real_
  spread <- function(x) if (sum(is.finite(x)) > 1L) stats::sd(x[is.finite(x)]) else NA_real_
  summary <- scores[, .(cindex_mean = average(cindex), cindex_sd = spread(cindex),
    uno_auc_mean = average(uno_auc), uno_auc_sd = spread(uno_auc),
    brier_mean = average(brier), brier_sd = spread(brier),
    valid_cindex_folds = sum(is.finite(cindex)),
    complete_metric_folds = sum(is.finite(cindex) & is.finite(uno_auc) & is.finite(brier))),
    by = model]
  summary[, eligible_for_selection := valid_cindex_folds == folds]
  summary[, selected_best_by_dev_cindex := FALSE]
  eligible <- which(summary$eligible_for_selection & is.finite(summary$cindex_mean))
  if (length(eligible)) {
    best <- eligible[which.max(summary$cindex_mean[eligible])]
    summary[best, selected_best_by_dev_cindex := TRUE]
  }
  data.table::setorder(summary, -cindex_mean)
  summary
}

# 7. Optional explanations of the exact final GBM risk learner ------------------

explain_gbm <- function(learner, development_data, features, directory, cfg, seed) {
  require_packages(c("fastshap", "shapviz", "ggplot2"))
  if (cfg$shap_nsim < 2L || cfg$shap_max_n < 1L) stop("Invalid SHAP settings.")
  directory <- file.path(directory, "GBM_explain")
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  X <- as.data.frame(development_data[, ..features])
  set.seed(seed)
  index <- if (nrow(X) > cfg$shap_max_n) sample(seq_len(nrow(X)), cfg$shap_max_n) else seq_len(nrow(X))
  background <- X[index, , drop = FALSE]
  predict_risk <- function(object, newdata) {
    as.numeric(object$predict_newdata(newdata = newdata)$lp)
  }
  shap <- fastshap::explain(learner, X = background, pred_wrapper = predict_risk,
                            nsim = cfg$shap_nsim, adjust = TRUE)
  importance <- data.table::data.table(predictor = colnames(shap),
                                      mean_abs_shap = colMeans(abs(shap)))
  data.table::setorder(importance, -mean_abs_shap)
  data.table::fwrite(importance, file.path(directory, "01_mean_absolute_SHAP.csv"))
  visual <- shapviz::shapviz(as.matrix(shap), X = background)
  for (kind in c("beeswarm", "bar")) {
    plot <- shapviz::sv_importance(visual, kind = kind, max_display = min(20L, ncol(X))) +
      ggplot2::labs(title = "Final GBM: SHAP on the log-risk scale")
    ggplot2::ggsave(file.path(directory, paste0("02_SHAP_", kind, ".png")),
                    plot, width = 8, height = 7, dpi = 300)
  }
  # The evaluated GBM is trained on encoded columns. Report native influence at
  # that encoded-column level; SHAP above is at the original-predictor level.
  base <- learner$graph_model$pipeops[["surv.gbm"]]$learner_model
  native <- data.table::as.data.table(summary(base$model, plotit = FALSE))
  data.table::setnames(native, c("var", "rel.inf"), c("encoded_feature", "relative_influence"))
  data.table::fwrite(native, file.path(directory, "03_encoded_relative_influence.csv"))
  data.table::fwrite(data.table::data.table(background_n = nrow(background),
    nsim = cfg$shap_nsim, prediction_scale = "GBM log-risk",
    population = "development subsample", baseline = mean(predict_risk(learner, background))),
    file.path(directory, "04_explanation_settings.csv"))
  if (cfg$export_individual_shap) {
    data.table::fwrite(cbind(data.table::data.table(development_row = index),
                            data.table::as.data.table(shap)),
                      file.path(directory, "05_individual_SHAP_LOCAL_ONLY.csv"))
  }
  invisible(NULL)
}

# 8. One endpoint: development CV, final training and temporal validation --------

run_endpoint <- function(endpoint, cohort, models, cfg, output_dir) {
  directory <- file.path(output_dir, endpoint)
  dir.create(directory, recursive = TRUE, showWarnings = FALSE)
  seed <- cfg$seed + match(endpoint, c("OS", "CSS")) * 10000L
  event_column <- tolower(paste0(endpoint, "_event"))
  dat <- data.table::copy(cohort[!is.na(get(event_column))])
  dat[, `:=`(time = surv_months, event = as.integer(get(event_column)))]
  columns <- c("time", "event", cfg$predictors)
  development <- dat[split == "Development", ..columns]
  validation <- dat[split == "TemporalValidation", ..columns]
  if (!nrow(development) || !nrow(validation) || !sum(development$event) || !sum(validation$event)) {
    stop(endpoint, ": both temporal cohorts must contain observations and events.")
  }
  selection <- select_predictors(development, cfg$predictors, cfg, seed)
  features <- selection$selected
  save_lasso(selection, directory, endpoint)
  writeLines(selection$removed_constants, file.path(directory, "00_removed_constants.txt"))
  columns <- c("time", "event", features)
  development <- development[, ..columns]
  validation <- validation[, ..columns]
  dev_task <- make_task(development, paste0(endpoint, "_development"))
  all_task <- make_task(data.table::rbindlist(list(development, validation)), paste0(endpoint, "_all"))
  train_ids <- seq_len(nrow(development))
  test_ids <- seq.int(nrow(development) + 1L, all_task$nrow)
  cv <- make_cv(dev_task, cfg$outer_folds, seed + 1L)
  message("[", endpoint, "] Fixed predictors: ", paste(features, collapse = ", "))
  scores <- list()
  tuning_logs <- list()
  for (fold in seq_len(cv$iters)) {
    message("[", endpoint, "] Development fold ", fold, "/", cv$iters)
    for (name in names(models)) {
      tuned <- tune_model(name, models[[name]], dev_task, cv$train_set(fold), cfg, seed + 100L + fold)
      if (!is.null(tuned$log)) {
        tuned$log[, `:=`(phase = "outer_cv", outer_fold = fold)]
        tuning_logs[[length(tuning_logs) + 1L]] <- tuned$log
      }
      score <- if (is.null(tuned$params)) {
        failed_score(name, "All tuning candidates failed at least one inner fold.")
      } else {
        fit_and_score(name, clone_model(models[[name]], tuned$params), dev_task,
          cv$train_set(fold), cv$test_set(fold), seed + 1000L + fold)$scores
      }
      score[, `:=`(fold = fold, n_features = length(features))]
      scores[[length(scores) + 1L]] <- score
    }
  }
  dev_scores <- data.table::rbindlist(scores, fill = TRUE)
  dev_summary <- summarize_cv(dev_scores, cfg$outer_folds)
  best <- dev_summary[selected_best_by_dev_cindex == TRUE, model]
  if (!length(best)) best <- NA_character_
  data.table::fwrite(dev_scores, file.path(directory, "05_development_CV_scores.csv"))
  data.table::fwrite(dev_summary, file.path(directory, "06_development_summary.csv"))

  external <- list()
  final_parameters <- list()
  for (name in names(models)) {
    message("[", endpoint, "] Final development fit / temporal validation: ", name)
    # Inner task contains only training rows, so validation outcomes never enter tuning.
    tuned <- tune_model(name, models[[name]], all_task, train_ids, cfg, seed + 2000L)
    if (!is.null(tuned$log)) {
      tuned$log[, `:=`(phase = "final_development_tuning", outer_fold = NA_integer_)]
      tuning_logs[[length(tuning_logs) + 1L]] <- tuned$log
    }
    if (is.null(tuned$params)) {
      score <- failed_score(name, "All final tuning candidates failed at least one inner fold.")
    } else {
      configured <- clone_model(models[[name]], tuned$params)
      final_parameters[[name]] <- lapply(configured, function(x) x$param_set$values)
      result <- fit_and_score(name, configured, all_task, train_ids, test_ids, seed + 3000L)
      score <- result$scores
      if (cfg$save_models && result$risk_ok && result$distribution_ok) {
        saveRDS(list(model = result$model, features = features,
          factor_levels = lapply(development[, ..features], levels), endpoint = endpoint),
          file.path(directory, paste0(name, "_LOCAL_ONLY.rds")))
      }
      if (name == "GBM" && cfg$explain_gbm && result$risk_ok) {
        tryCatch(explain_gbm(result$model$risk, development, features, directory, cfg, seed + 4000L),
          error = function(e) {
            writeLines(conditionMessage(e), file.path(directory, "GBM_explanation_error.txt"))
            warning("GBM explanation failed: ", conditionMessage(e))
          })
      }
    }
    score[, `:=`(n_features = length(features), selected_from_development =
                   !is.na(best) & model == best)]
    external[[length(external) + 1L]] <- score
  }
  external <- data.table::rbindlist(external, fill = TRUE)
  data.table::fwrite(external, file.path(directory, "07_temporal_validation_scores.csv"))
  if (length(tuning_logs)) data.table::fwrite(data.table::rbindlist(tuning_logs, fill = TRUE),
                                             file.path(directory, "08_tuning_log.csv"))
  dput(final_parameters, file = file.path(directory, "09_final_parameters.R"))
  errors <- data.table::rbindlist(list(
    dev_scores[!is.na(error), .(phase = "development_CV", fold, model, error)],
    external[!is.na(error), .(phase = "temporal_validation", fold = NA_integer_, model, error)]))
  data.table::fwrite(errors, file.path(directory, "10_model_errors.csv"))
  data.table::data.table(endpoint = endpoint, development_n = nrow(development),
    validation_n = nrow(validation), development_events = sum(development$event),
    validation_events = sum(validation$event), missing_endpoint_excluded = sum(is.na(cohort[[event_column]])),
    best_model_by_development_CV = best, selected_predictors = paste(features, collapse = "; "),
    lasso_fallback = selection$fallback, lasso_lambda = selection$lambda,
    model_metric_errors = nrow(errors))
}

# 9. Entrypoint ---------------------------------------------------------------

main <- function(input_file = "data/00_cleaned_analysis_cohort.csv",
                 output_dir = "results", config = CONFIG) {
  require_packages(c("data.table", "survival", "glmnet", "mlr3", "mlr3proba",
                     "mlr3pipelines", "mlr3extralearners", "survAUC"))
  suppressPackageStartupMessages({
    library(data.table)
    library(mlr3)
    library(mlr3proba)
    library(mlr3pipelines)
    library(mlr3extralearners)
  })
  if (config$explain_gbm) require_packages(c("fastshap", "shapviz", "ggplot2"))
  set.seed(config$seed)
  cohort <- read_cohort(input_file, config)
  models <- make_models()  # Require all eight learners; do not silently skip models.
  dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
  if (!dir.exists(output_dir)) stop("Cannot create output directory: ", output_dir)
  on.exit(writeLines(capture.output(sessionInfo()), file.path(output_dir, "sessionInfo.txt")), add = TRUE)
  dput(config, file = file.path(output_dir, "analysis_config.R"))
  data.table::fwrite(data.table::data.table(input_md5 = unname(tools::md5sum(input_file)),
    run_started_utc = format(Sys.time(), tz = "UTC", usetz = TRUE), requested_models = length(models)),
    file.path(output_dir, "run_metadata.csv"))
  overview <- data.table::rbindlist(lapply(config$endpoints, function(endpoint) {
    run_endpoint(endpoint, cohort, models, config, output_dir)
  }))
  data.table::fwrite(overview, file.path(output_dir, "analysis_overview.csv"))
  message("Analysis finished. Inspect model errors and follow-up support flags in: ", output_dir)
  print(overview)
  invisible(overview)
}

if (sys.nframe() == 0L) {
  args <- commandArgs(trailingOnly = TRUE)
  if (any(args %in% c("--help", "-h"))) {
    cat("Usage: Rscript seer_nec_survival_ml.R [cleaned_cohort.csv] [output_directory]\n")
    cat("Defaults: data/00_cleaned_analysis_cohort.csv and results\n")
    cat("For configuration and optional SHAP, source the script and call main(config = CONFIG).\n")
  } else {
    if (length(args) > 2L) stop("Expected at most two arguments; use --help.")
    main(input_file = if (length(args) >= 1L) args[1] else "data/00_cleaned_analysis_cohort.csv",
         output_dir = if (length(args) >= 2L) args[2] else "results")
  }
}
