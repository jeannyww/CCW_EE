# Single dataset illustration of estimating equations - the R version of the estimating equations implemented in simulation_compare_iptw_bootstrap.py
# Jeanny Wang
# 2026-09-05: using R package deli, port of python delicatessen library
# Barrett M (2026). deli: M-Estimation and Empirical Sandwich Variance Estimation. R package version 0.1.0, https://github.com/r-causal/deli.
# https://r-causal.github.io/deli/articles/getting-started.html
# Utilizes pooled logistic regression estimating equations developed by Zivich et al 2025.
# Zivich PN, Cole SR, Shook-Sa BE, DeMonte JB, & Edwards JK. (2025). Estimating equations for survival analysis with pooled logistic regression. arXiv:2504.13291

# Libraries required: base R and deli (+ (minpack.lm::nls.lm()) if using LM algorithm)
library(deli)

# Setup
getwd()
options(max.print = 100)
# Load data single datasets, these are the data evaluated in simulation_compare_iptw_bootstrap.py
# These are the long version datasets
d_obs <- read.csv("./data/d_obs_long.csv")
d_cens1 <- read.csv("./data/d_cens1_long.csv")
d_cens0 <- read.csv("./data/d_cens0_long.csv")
# These are the wide transformations of the long datasets.
# Estimating equations implementation uses the wide dataset (1 row per 1 unique individual)
d <- read.csv("./data/d_obs_wide.csv")
d_c0 <- read.csv("./data/d_cens0_wide.csv")
d_c1 <- read.csv("./data/d_cens1_wide.csv")
# Grace period length
gp <- 2

##################################################
# SECTION - Vectors and inits
##################################################
### Input from wide dataset
# Covariate
W <- d[["W"]]
# For the observed treatment
trt_obs <- d[["trt_obs"]] # whether the treatment was received at any point in followup
t_trt <- d[["t_trt"]] # Time that the treatment was received during followup

# The times that a treatment could have been received within the grace period
trt_times <- d[((d["t_trt"] <= gp) & (d["trt_obs"] == 1)), "t_trt"]
unique_trt_times <- unique(trt_times)
n_trt_times <- length(unique_trt_times)

# History of treatment
trt_obs_lag <- d[["trt_obs_lag"]] # indicator of treatment lag
t_trt_lag <- d[["t_trt_lag"]] # Time of the treatment lag
# For the risk set matrix
delta <- d[["event"]] # indicator of whether event happened
t_delta <- d[["t_out"]] # time that event occured

### Input from the censored clones (data has already gone through the artifically censored process)
trt_clone1 <- d_c1[["trt_plan"]]
cens1 <- d_c1[["censor"]]
t_cens1 <- d_c1[["t_in_cens"]]
delta1 <- d_c1[["event"]]
t_delta1 <- d_c1[["t_out"]]

trt_clone0 <- d_c0[["trt_plan"]]
cens0 <- d_c0[["censor"]]
t_cens0 <- d_c0[["t_in_cens"]]
delta0 <- d_c0[["event"]]
t_delta0 <- d_c0[["t_out"]]

### All time points to predict
times_to_predict <- sort(unique(d[["t_out"]] - 1))
n_times_to_predict = length(times_to_predict)

### Creat inits - these are the sub-optimal inits that are used in the python version
inits_iptw = c(-24.0, -0.1309, -1.8034, -0.2509, -0.2721)
init_time_y1 <- c(-2.3807, -2.206558, -1.1257, -2.748212, -2.855993, -2.816546)
init_time_y0 <- c(0.1158363, 0.12840615, 0.16394126, 0.17089769, 0.17709666, 0.1834949)

# For the parameters of interest
init_risks1 = c(0.115, 0.13, 0.17970892, 0.17970892, 0.18355848, 0.18355848)
init_risks0 = c(0.0854, 0.092283, 0.099975, 0.114262, 0.121953, 0.135142)
#  For risk differences
init_rd <- c(0.0295, 0.0377, 0.0797, 0.0654, 0.0616, 0.0484)

# All inits for the entire psi stack
inits = c(init_rd, init_risks1, init_risks0, init_time_y1, init_time_y0, inits_iptw)

# !SECTION - Vectors and inits

##################################################
# SECTION - Estimating equations
##################################################

# Estimating equation for the IPTW model. This is an R translation of the Pooled Logistic Regression estimating equation implementation developed by Paul N Zivich III PhD et al 2025 (https://arxiv.org/abs/2504.13291). See also https://deli.readthedocs.io/en/latest/Examples/Zivich-PLogit.html, https://r-causal.github.io/deli/reference/ee_plogit.html
psi_iptw <- function(theta) {
    # Separating the thetas
    beta_a_lag <- theta[0:1]
    beta_w <- theta[2]
    beta_time <- theta[3:length(theta)]
    # Observed treatment matrix
    trt_obs_matrix <- sweep(
        x = outer(X = unique_trt_times, Y = t_trt, FUN = "=="), # create boolean matrix of outer product of the unique treatment times and the vector of the times of treatment
        MARGIN = 2, # multiple the outer product boolean matrix by the trt_obs, by column
        STATS = trt_obs, # multiplication happens by each column
        FUN = "*"
    )
    # Risk set matrix
    in_risk_set <- 1 * outer(X = unique_trt_times, Y = t_delta - 1, FUN = "<=")
    # Treatment lag matrix
    trt_lag_matrix <- sweep(
        x = outer(X = unique_trt_times, Y = t_trt_lag, FUN = "=="), # boolean matrix of outer product of the unique treatment lag times with the vector of the  times of treatment lagged
        MARGIN = 2, # by column-wise operation
        STATS = trt_obs_lag, # apply FUN with trt_obs_lag
        FUN = "*"
    )

    # Creating the log-odds for the prediction matrix
    logodds_a_lag <- (trt_lag_matrix * beta_a_lag) # broadcast multiplication with a scalar
    logodds_a_lag_matrix <- logodds_a_lag * in_risk_set # multiplying by the risk set matrix in case the person was censored or died at the lag treatment time
    logodds_w <- W * beta_w
    logodds_w_matrix <- matrix(data = rep(logodds_w, length(unique_trt_times)), ncol = length(logodds_w), byrow = TRUE)
    logodds_covariates <- logodds_a_lag_matrix + logodds_w_matrix
    # Log odds discrete indicators for time matrix
    time_design_matrix <- diag(n_trt_times)
    time_design_matrix[0:n_trt_times, 1] <- 1
    logodds_t <- time_design_matrix %*% beta_time
    # Predicted probability of treatment
    logodds_t_matrix <- matrix(data = rep(logodds_t, length(trt_obs)), ncol = length(trt_obs))

    trt_pred_matrix <- plogis(logodds_covariates + logodds_t_matrix)

    trt_residual_matrix <- (trt_obs_matrix - trt_pred_matrix) * in_risk_set

    a_lag_score <- matrix(rep(1, n_trt_times), nrow = 1) %*% (trt_residual_matrix * trt_lag_matrix)

    trt_residuals <- matrix(rep(1, n_trt_times), nrow = 1) %*% (trt_residual_matrix)

    w_score <- trt_residuals * matrix(W, nrow = 1)

    t_score <- trt_residual_matrix

    score <- rbind(a_lag_score, w_score, t_score)
}

# # Testing the IPTW modeling
# ee_test <- psi_iptw(inits_iptw)
# test_iptw <- MEstimator(stacked_equations = psi_iptw, init = inits_iptw) |> estimate(solver = "nleqslv")
# coef(test_iptw)
# #  theta_1      theta_2      theta_3      theta_4      theta_5
# # -25.59853928  -0.56198644  -1.96760816   0.12551080   0.01469285
# test_iptw <- MEstimator(stacked_equations = psi_iptw, init = inits_iptw) |> estimate(solver = "lm")
# coef(test_iptw)
# # With lm solver
# #      theta_1      theta_2      theta_3      theta_4      theta_5
# # -29.09466494  -0.56446480  -1.96546575   0.12844008   0.01433613
# test_iptw <- m_estimate(stacked_equations = psi_iptw, init = inits_iptw)
# test_iptw@theta
# # test_m@theta
# # > test_m@theta with default solver
# #  theta_1      theta_2      theta_3      theta_4      theta_5
# # -28.09467210  -0.56446480  -1.96546575   0.12844008   0.01433613

# Intermediate step for creating the weight matrix
iptw_weights <- function(theta) {
    # Separating the thetas
    beta_a_lag <- theta[0:1]
    beta_w <- theta[2]
    beta_time <- theta[3:length(theta)]
    # Creating the time varying probability of treatment matrix
    trt_obs_matrix <- sweep(
        x = outer(X = unique_trt_times, Y = t_trt, FUN = "=="),
        MARGIN = 2,
        STATS = trt_obs,
        FUN = "*"
    )
    trt_lag_matrix <- sweep(
        x = outer(X = unique_trt_times, Y = t_trt_lag, FUN = "=="),
        MARGIN = 2,
        STATS = trt_obs_lag,
        FUN = "*"
    )
    time_design_matrix <- diag(n_trt_times)
    time_design_matrix[1:n_trt_times, 1] <- 1
    logodds_a_lag_matrix <- trt_lag_matrix * beta_a_lag
    logodds_w_matrix <- matrix(data = rep(W * beta_w, n_trt_times), ncol = length(W), byrow = TRUE)
    logodds_t <- time_design_matrix %*% matrix(beta_time)
    logodds_t_matrix <- matrix(data = rep(logodds_t, length(trt_obs)), ncol = length(trt_obs))
    pred_trt_matrix <- plogis(logodds_a_lag_matrix + logodds_w_matrix + logodds_t_matrix)
    pred_trt_observed_mtrx <- (trt_obs_matrix * pred_trt_matrix) + (1 - trt_obs_matrix) * (1 - pred_trt_matrix)
    pred_trt_alltimepoints <- matrix(1, nrow = length(times_to_predict), ncol = length(trt_obs))
    pred_trt_alltimepoints[unique_trt_times + 1, ] <- pred_trt_observed_mtrx
    pred_trt_cumprod_mtrx <- apply(X = pred_trt_alltimepoints, MARGIN = 2, FUN = cumprod)
    in_risk_set_all <- 1 * outer(X = times_to_predict, Y = (t_delta - 1), FUN = "<=")

    # Transform the predicted probabilities into the probability of remaining uncensored for each plan
    # Weight matrix for clones to strategy 1, start treatment by the gp
    cens1_t_matrix <- sweep(
        x = outer(X = times_to_predict, Y = t_cens1, FUN = "=="),
        MARGIN = 2,
        STATS = cens1,
        FUN = "*"
    )
    uncens1_t_mtrx <- 1 - apply(X = cens1_t_matrix, MARGIN = 2, FUN = cumsum) # indicates when the observation remains uncensored
    cens1_risk_set_mtrx <- uncens1_t_mtrx * in_risk_set_all
    pred_trt_cumprod_mtrx_gp <- matrix(1, nrow = length(times_to_predict), ncol = length(trt_obs))
    # Identifying those who were treated exactly at the gp as those who are upweighted as the counterfactual
    pred_trt_cumprod_mtrx_gp[gp + 1, ((t_trt == 2) & (trt_obs == 1))] <- pred_trt_cumprod_mtrx[gp + 1, ((t_trt == 2) & (trt_obs == 1))]
    cens1_tmp_mtrx <- cens1_risk_set_mtrx * (pred_trt_cumprod_mtrx_gp**(-1))
    cens1_weight_mtrx <- apply(X = cens1_tmp_mtrx, MARGIN = 2, FUN = cumprod)
    # Weight matrix for clones to strategy 0, no trt within nor at the gp
    cens0_t_mtrx <- sweep(
        x = outer(X = times_to_predict, Y = t_cens0, FUN = "=="),
        MARGIN = 2,
        STATS = cens0,
        FUN = "*"
    )
    uncens0_t_mtrx <- 1 - apply(X = cens0_t_mtrx, MARGIN = 2, FUN = cumsum)
    cens0_risk_set_mtrx <- uncens0_t_mtrx * in_risk_set_all
    cens0_weight_mtrx <- cens0_risk_set_mtrx * (pred_trt_cumprod_mtrx**(-1))

    return(list(
        noweight = 1,
        cens1 = cens1_weight_mtrx,
        cens0 = cens0_weight_mtrx
    ))
}
tmpweights <- iptw_weights(test_m@theta)
str(tmpweights)

# Estimating equation for the weighted MSM. This is the weighted version of the Pooled Logistic Regression estimating equation implementation developed by Paul N Zivich III PhD et al 2025 (https://arxiv.org/abs/2504.13291). See also https://deli.readthedocs.io/en/latest/Examples/Zivich-PLogit.html, https://r-causal.github.io/deli/reference/ee_plogit.html
psi_msm <- function(theta, weight_matrix = tmpweights$cens1) {
    # psi_msm <- function(theta, weight_matrix = 1) {
    beta_t <- theta
    time_design_matrix <- diag(n_times_to_predict)
    time_design_matrix[, 1] <- 1
    lodds_t_to_predict <- time_design_matrix %*% matrix(beta_t)
    delta_pred_mtrx <- matrix(rep(plogis(lodds_t_to_predict), length(delta)), ncol = length(delta))
    delta_obs_mtrx <- sweep(
        x = outer(X = times_to_predict, Y = t_delta - 1, FUN = "=="),
        MARGIN = 2,
        STATS = delta,
        FUN = "*"
    )
    delta_residual_matrix <- (delta_obs_mtrx - delta_pred_mtrx) * weight_matrix
}
# test_msm <- m_estimate(stacked_equations = psi_msm, init = init_risks0)
# summary(test_msm)
# test_msm@theta
# Intermediate step for creating the risk predictions
pred_risk <- function(theta) {
    beta_t <- theta
    time_design_matrix <- diag(n_times_to_predict)
    time_design_matrix[, 1] <- 1
    lodds_t_to_predict <- time_design_matrix %*% matrix(beta_t)
    delta_pred <- plogis(lodds_t_to_predict)
    delta_pred_haz <- matrix(0, nrow = n_times_to_predict, length(delta))
    delta_pred_haz[times_to_predict + 1, ] <- delta_pred
    delta_pred_surv <- apply(X = (1 - delta_pred_haz), MARGIN = 2, FUN = cumprod)
    delta_pred_risk <- 1 - delta_pred_surv
}

# Full psi stack for CCW
# Brings everything together. See https://r-causal.github.io/deli/articles/custom-estimating-equations.html
psi_ccw <- function(theta) {
    theta_rd <- theta[1:n_times_to_predict]
    theta_r1 <- theta[(n_times_to_predict + 1):(n_times_to_predict * 2)]
    theta_r0 <- theta[((n_times_to_predict * 2) + 1):(n_times_to_predict * 3)]
    theta_msm_r1 <- theta[((n_times_to_predict * 3) + 1):(n_times_to_predict * 4)]
    theta_msm_r0 <- theta[((n_times_to_predict * 4) + 1):index_msm]
    theta_iptw <- theta[-seq_len(index_msm)]
    # IPTW model
    ee_iptw <- psi_iptw(theta = theta_iptw)
    # Weight matrices
    weight_matrices <- iptw_weights(theta = theta_iptw)
    # MSM for outcome
    ee_msm_r1 <- psi_msm(theta = theta_msm_r1, weight_matrix = weight_matrices$cens1)
    ee_msm_r0 <- psi_msm(theta = theta_msm_r0, weight_matrix = weight_matrices$cens0)
    # Intermediate risk prediction step
    pred_risk_c1 <- pred_risk(theta = theta_msm_r1)
    pred_risk_c0 <- pred_risk(theta = theta_msm_r0)
    # Risks
    ee_risk_1 <- pred_risk_c1 - matrix(rep(theta_r1, length(delta)), nrow = n_times_to_predict, ncol = length(delta))
    ee_risk_0 <- pred_risk_c0 - matrix(rep(theta_r0, length(delta)), nrow = n_times_to_predict, ncol = length(delta))
    # Risk difference
    ee_rd <- matrix(rep((theta_r1 - theta_r0 - theta_rd), length(delta)), nrow = n_times_to_predict, ncol = length(delta))
    # Psi stack
    ee_stack <- rbind(ee_rd, ee_risk_1, ee_risk_0, ee_msm_r1, ee_msm_r0, ee_iptw)
}

# !SECTION - ee

##################################################
# SECTION - Implementing the CCW EE
##################################################
index_msm <- n_times_to_predict * 5
inits = c(init_rd, init_risks1, init_risks0, init_time_y1, init_time_y0, inits_iptw)
# m_estr <- MEstimator(stacked_equations = psi_ccw, init = inits) |> estimate()
# https://r-causal.github.io/deli/reference/estimate.html
start_time <- Sys.time()
m_estr <- MEstimator(stacked_equations = psi_ccw, init = inits) |> estimate(solver = "lm")
end_time <- Sys.time()
# Using (minpack.lm::nls.lm()), which mirrors the delicatessen default (scipy.optimize.root(method = "lm"))
print(start_time)
print(end_time)
print(c("total time in seconds:", round(end_time - start_time, 4)))
# [1] "total time in seconds:" "50.1006"
str(m_estr)
coef(m_estr)
sqrt(diag(m_estr@variance))
dim(confidence_intervals(m_estr))

# Organizing the R outputs into a table:
estimates <- unname(coef(m_estr)[1:(n_times_to_predict * 3)])
var <- unname(diag(m_estr@variance)[1:(n_times_to_predict * 3)])
ci <- (confidence_intervals(m_estr))[1:(n_times_to_predict * 3), ]
measure <- c(rep("rd", n_times_to_predict), rep("risk1", n_times_to_predict), rep("risk0", n_times_to_predict))
t_in <- rep(1:6 - 1, 3)

out_R <- data.frame(
    t_in = t_in,
    estimate = estimates,
    variance = var,
    stderr = sqrt(var),
    LCL = ci[, 1],
    UCL = ci[, 2],
    measure = measure
)
rownames(out_R) <- NULL
print(out_R)
#    t_in     estimate     variance       stderr           LCL         UCL measure
# 1     0 0.0005073060 1.598508e-07 0.0003998135 -0.0002763141 0.001290926      rd
# 2     1 0.0006280231 4.306959e-07 0.0006562743 -0.0006582509 0.001914297      rd
# 3     2 0.0076312475 1.654381e-05 0.0040674086 -0.0003407269 0.015603222      rd
# 4     3 0.0079366188 2.662432e-05 0.0051598756 -0.0021765516 0.018049789      rd
# 5     4 0.0067415349 3.522681e-05 0.0059352174 -0.0048912775 0.018374347      rd
# 6     5 0.0081413839 5.143338e-05 0.0071717070 -0.0059149035 0.022197671      rd
# 7     0 0.0104000000 1.029181e-06 0.0010144854  0.0084116452 0.012388355   risk1
# 8     1 0.0209000000 2.046315e-06 0.0014304946  0.0180962821 0.023703718   risk1
# 9     2 0.0376929161 1.672414e-05 0.0040895155  0.0296776129 0.045708219   risk1
# 10    3 0.0489126436 2.524527e-05 0.0050244668  0.0390648697 0.058760418   risk1
# 11    4 0.0595242235 3.223839e-05 0.0056778857  0.0483957720 0.070652675   risk1
# 12    5 0.0780582762 4.619126e-05 0.0067964155  0.0647375465 0.091379006   risk1
# 13    0 0.0098926940 1.089592e-06 0.0010438355  0.0078468141 0.011938574   risk0
# 14    1 0.0202719769 2.355546e-06 0.0015347787  0.0172638660 0.023280088   risk0
# 15    2 0.0300616686 3.693550e-06 0.0019218611  0.0262948901 0.033828447   risk0
# 16    3 0.0409760248 5.171309e-06 0.0022740513  0.0365189663 0.045433083   risk0
# 17    4 0.0527826886 6.695661e-06 0.0025875975  0.0477110907 0.057854286   risk0
# 18    5 0.0699168922 8.854523e-06 0.0029756550  0.0640847155 0.075749069   risk0

out_python <- read.csv("./out_eeiptw_Rcompare.csv")
print(out_python)
# Compare to the MEstimator python output, out_eeiptw
#    X     estimate     variance       stderr           LCL         UCL measure
# 1  0 0.0005073060 1.598513e-07 0.0003998141 -0.0002763152 0.001290927      rd
# 2  1 0.0006280231 4.306943e-07 0.0006562730 -0.0006582484 0.001914295      rd
# 3  2 0.0076312475 1.654380e-05 0.0040674074 -0.0003407245 0.015603220      rd
# 4  3 0.0079366188 2.662455e-05 0.0051598978 -0.0021765950 0.018049833      rd
# 5  4 0.0067415349 3.522706e-05 0.0059352392 -0.0048913201 0.018374390      rd
# 6  5 0.0081413839 5.143342e-05 0.0071717097 -0.0059149089 0.022197677      rd
# 7  0 0.0104000000 1.029190e-06 0.0010144901  0.0084116360 0.012388364   risk1
# 8  1 0.0209000000 2.046314e-06 0.0014304944  0.0180962825 0.023703718   risk1
# 9  2 0.0376929161 1.672413e-05 0.0040895143  0.0296776154 0.045708217   risk1
# 10 3 0.0489126436 2.524543e-05 0.0050244833  0.0390648374 0.058760450   risk1
# 11 4 0.0595242235 3.223863e-05 0.0056779071  0.0483957300 0.070652717   risk1
# 12 5 0.0780582762 4.619127e-05 0.0067964158  0.0647375459 0.091379006   risk1
# 13 0 0.0098926940 1.089592e-06 0.0010438354  0.0078468142 0.011938574   risk0
# 14 1 0.0202719769 2.355533e-06 0.0015347746  0.0172638741 0.023280080   risk0
# 15 2 0.0300616686 3.693540e-06 0.0019218584  0.0262948954 0.033828442   risk0
# 16 3 0.0409760248 5.171321e-06 0.0022740538  0.0365189612 0.045433088   risk0
# 17 4 0.0527826886 6.695657e-06 0.0025875968  0.0477110920 0.057854285   risk0
# 18 5 0.0699168922 8.854550e-06 0.0029756596  0.0640847066 0.075749078   risk0

# Exact same up to the 10th sigfig
sum(signif(out_python$estimate, 10) == signif(out_R$estimate, 10))
# 18

# Exact same up to the 4th sigfig
sum(signif(out_python$variance, 4) == signif(out_R$variance, 4))
# 18

# Exact same up to the 5th sigfig
sum(signif(out_python$stderr, 5) == signif(out_R$stderr, 5))
# 18

# Exact same up to the 3rd sigfig
sum(signif(out_python$LCL, 3) == signif(out_R$LCL, 3))
# 18

# Exact same up to the 5th sigfig
sum(signif(out_python$UCL, 5) == signif(out_R$UCL, 5))
# 18

# !SECTION - Implementing the CCW EE

##################################################
# SECTION - Scrap code for testing and creating the ee's -- keeping here to backtrack
##################################################
theta <- c(init_rd, init_risks1, init_risks0, init_time_y1, init_time_y0, inits_iptw)
theta_rd <- theta[1:n_times_to_predict]
theta_r1 <- theta[(n_times_to_predict + 1):(n_times_to_predict * 2)]
theta_r0 <- theta[((n_times_to_predict * 2) + 1):(n_times_to_predict * 3)]
theta_msm_r1 <- theta[((n_times_to_predict * 3) + 1):(n_times_to_predict * 4)]
theta_msm_r0 <- theta[((n_times_to_predict * 4) + 1):index_msm]
theta_iptw <- theta[-seq_len(index_msm)]
# IPTW model
ee_iptw <- psi_iptw(theta = theta_iptw)
# Weight matrices
weight_matrices <- iptw_weights(theta = theta_iptw)
# MSM for outcome
ee_msm_r1 <- psi_msm(theta = theta_msm_r1, weight_matrix = weight_matrices$cens1)
ee_msm_r0 <- psi_msm(theta = theta_msm_r0, weight_matrix = weight_matrices$cens0)
# Intermediate risk prediction step
pred_risk_c1 <- pred_risk(theta = theta_msm_r1)
pred_risk_c0 <- pred_risk(theta = theta_msm_r0)
# Risks
ee_risk_1 <- matrix(rep(theta_r1, length(delta)), nrow = n_times_to_predict, ncol = length(delta)) - pred_risk_c1
ee_risk_0 <- matrix(rep(theta_r0, length(delta)), nrow = n_times_to_predict, ncol = length(delta)) - pred_risk_c0
# Risk difference
ee_rd <- pred_risk_c1 - pred_risk_c0
pred_risk_c1[, 1:10]
pred_risk_c0[, 1:10]
ee_rd[, 1:10]
# Psi stack
ee_stack <- rbind(ee_rd, ee_risk_1, ee_risk_0, ee_msm_r1, ee_msm_r0, ee_iptw)
ee_stack[, 1:2]
# Testing the full psi stack

#############################################
theta <- inits
theta_rd <- theta[1:n_times_to_predict]
theta_r1 <- theta[(n_times_to_predict + 1):(n_times_to_predict * 2)]
theta_r0 <- theta[((n_times_to_predict * 2) + 1):(n_times_to_predict * 3)]
theta_msm_r1 <- theta[((n_times_to_predict * 3) + 1):(n_times_to_predict * 4)]
theta_msm_r0 <- theta[((n_times_to_predict * 4) + 1):index_msm]
theta_iptw <- theta[-seq_len(index_msm)]
test <- pred_risk(theta = test_msm@theta)
# Test for creating risk predictions
beta_t = test_msm@theta
beta_t@theta
time_design_matrix <- diag(n_times_to_predict)
time_design_matrix[, 1] <- 1
lodds_t_to_predict <- time_design_matrix %*% matrix(beta_t)
delta_pred <- plogis(lodds_t_to_predict)
delta_pred_haz <- matrix(0, nrow = n_times_to_predict, length(delta))
delta_pred_haz[times_to_predict + 1, ] <- delta_pred
delta_pred_surv <- apply(X = (1 - delta_pred_haz), MARGIN = 2, FUN = cumprod)
delta_pred_risk <- 1 - delta_pred_surv
ee_risk_1 <- matrix(rep(theta_r1, length(delta)), nrow = n_times_to_predict, ncol = length(delta)) - delta_pred_risk
rep(theta_r1, 3)
tmp <- matrix(rep(theta_r1, length(delta)), nrow = n_times_to_predict, ncol = length(delta))

matrix(rep(theta_r1, 3), ncol = 3)
tmp[1:6, 110:120]
ee_risk_1[1:6, 110:120]
delta_pred_surv[1:6, 110:120]
delta_pred_risk[1:6, 110:120]

#############################################
# Test for the msm
beta_t <- init_risks1
time_design_matrix <- diag(n_times_to_predict)
time_design_matrix[, 1] <- 1
lodds_t_to_predict <- time_design_matrix %*% matrix(beta_t)
delta_pred_mtrx <- matrix(rep(plogis(lodds_t_to_predict), length(delta)), ncol = length(delta))
delta_obs_mtrx <- sweep(
    x = outer(X = times_to_predict, Y = t_delta - 1, FUN = "=="),
    MARGIN = 2,
    STATS = delta,
    FUN = "*"
)
delta_residual_matrix <- (delta_obs_mtrx - delta_pred_mtrx) * weight_matrix

delta_pred_mtrx[1:6, 1:10]
#############################################
# Test for weight matrix

beta_a_lag <- test_m@theta[0:1]
beta_w <- test_m@theta[2]
beta_time <- test_m@theta[3:length(test_m@theta)]
# Creating the time varying probability of treatment matrix
trt_obs_matrx <- sweep(
    x = outer(X = unique_trt_times, Y = t_trt, FUN = "=="),
    MARGIN = 2,
    STATS = trt_obs,
    FUN = "*"
)
trt_lag_matrix <- sweep(
    x = outer(X = unique_trt_times, Y = t_trt_lag, FUN = "=="),
    MARGIN = 2,
    STATS = trt_obs_lag,
    FUN = "*"
)
time_design_matrix <- diag(n_trt_times)
time_design_matrix[1:n_trt_times, 1] <- 1
logodds_a_lag_matrix <- trt_lag_matrix * beta_a_lag
logodds_w_matrix <- matrix(data = rep(W * beta_w, n_trt_times), ncol = length(W), byrow = TRUE)
logodds_t <- time_design_matrix %*% matrix(beta_time)
logodds_t_matrix <- matrix(data = rep(logodds_t, length(trt_obs)), ncol = length(trt_obs))
pred_trt_matrix <- plogis(logodds_a_lag_matrix + logodds_w_matrix + logodds_t_matrix)
pred_trt_observed_mtrx <- (trt_obs_matrix * pred_trt_matrix) + (1 - trt_obs_matrix) * (1 - pred_trt_matrix)
pred_trt_alltimepoints <- matrix(1, nrow = length(times_to_predict), ncol = length(trt_obs))
pred_trt_alltimepoints[unique_trt_times + 1, ] = pred_trt_observed_mtrx
pred_trt_cumprod_mtrx <- apply(X = pred_trt_alltimepoints, MARGIN = 2, FUN = cumprod)
in_risk_set_all <- 1 * outer(X = times_to_predict, Y = (t_delta - 1), FUN = "<=")

# Transform the predicted probabilities into the probability of remaining uncensored for each plan
# Weight matrix for clones to strategy 1, start treatment by the gp
cens1_t_matrix <- sweep(
    x = outer(X = times_to_predict, Y = t_cens1, FUN = "=="),
    MARGIN = 2,
    STATS = cens1,
    FUN = "*"
)
uncens1_t_mtrx <- 1 - apply(X = cens1_t_matrix, MARGIN = 2, FUN = cumsum) # indicates when the observation remains uncensored
cens1_risk_set_mtrx <- uncens1_t_mtrx * in_risk_set_all
pred_trt_cumprod_mtrx_gp <- matrix(1, nrow = length(times_to_predict), ncol = length(trt_obs))
# Identifying those who were treated exactly at the gp as those who are upweighted as the counterfactual
pred_trt_cumprod_mtrx_gp[gp + 1, ((t_trt == 2) & (trt_obs == 1))] <- pred_trt_cumprod_mtrx[gp + 1, ((t_trt == 2) & (trt_obs == 1))]
cens1_tmp_mtrx <- cens1_risk_set_mtrx * (pred_trt_cumprod_mtrx_gp**(-1))
cens1_weight_mtrx <- apply(X = cens1_tmp_mtrx, MARGIN = 2, FUN = cumprod)
# Weight matrix for clones to strategy 0, no trt within nor at the gp
cens0_t_mtrx <- sweep(
    x = outer(X = times_to_predict, Y = t_cens0, FUN = "=="),
    MARGIN = 2,
    STATS = cens0,
    FUN = "*"
)
uncens0_t_mtrx <- 1 - apply(X = cens0_t_mtrx, MARGIN = 2, FUN = cumsum)
cens0_risk_set_mtrx = uncens0_t_mtrx * in_risk_set_all
cens0_weight_mtrx <- cens0_risk_set_mtrx * (pred_trt_cumprod_mtrx**(-1))


unique((cens0_weight_mtrx)[4, ])
unique((pred_trt_cumprod_mtrx**(-1))[gp + 1, ])
unique(cens1_weight_mtrx[gp + 1, ])
cens1_weight_mtrx[1:6, 1:15]
tmp[1:6, 1:15]
t_trt[1:15]
trt_obs[1:15]

# data <- rep(logodds_t, 10)
in_risk_set_all[1:6, 1:33]
(t_delta - 1)[1:33]
(delta)[1:33]
dim(pred_trt_alltimepoints)
test <- outer(X = unique_trt_times, Y = trt_obs_lag, FUN = "==")
trt_lag_matrix[1:3, 1:33]
dim(test)

#############################################
# Test for iptw psi

theta <- inits_iptw
beta_a_lag <- theta[0:1]
beta_w = theta[2]
beta_time <- theta[3:length(x)]
# https://search.r-project.org/R/refmans/base/html/sweep.html
# Observed treatment matrix
trt_obs_matrix <- sweep(
    x = outer(X = unique_trt_times, Y = t_trt, FUN = "=="), # create boolean matrix of outer product of the unique treatment times and the vector of the times of treatment
    MARGIN = 2, # multiple the outer product boolean matrix by the trt_obs, by column
    STATS = trt_obs, # multiplication happens by each column
    FUN = "*"
)
# Risk set matrix
in_risk_set <- 1 * outer(X = unique_trt_times, Y = t_delta - 1, FUN = "<=")
# in_risk_set[1:3, 1:20]
# print(1 * (outer(X = unique_trt_times, Y = t_delta - 1, FUN = "<=")[1:3, 1:20]))
# Treatment lag matrix
print(trt_obs_matrix[1:3, 1:10])
trt_lag_matrix <- sweep(
    x = outer(X = unique_trt_times, Y = t_trt_lag, FUN = "=="), # boolean matrix of outer product of the unique treatment lag times with the vector of the  times of treatment lagged
    MARGIN = 2, # by column-wise operation
    STATS = trt_obs_lag, # apply FUN with trt_obs_lag
    FUN = "*"
)
# print(trt_lag_matrix[1:3, 1:10])
# print(trt_obs_matrix[1:3, 1:10])

# Creating the log-odds for the prediction matrix
logodds_a_lag = (trt_lag_matrix * beta_a_lag) # broadcast multiplication with a scalar
# print(logodds_a_lag[1:3, 1:20])
logodds_a_lag_matrix <- logodds_a_lag * in_risk_set # multiplying by the risk set matrix in case the person was censored or died at the lag treatment time
# print(logodds_a_lag_matrix[1:3, 1:20])
logodds_w <- W * beta_w
logodds_w_matrix <- matrix(data = rep(logodds_w, length(unique_trt_times)), ncol = length(logodds_w), byrow = TRUE)
# print(logodds_w_matrix[1:3, 1:20])

logodds_covariates <- logodds_a_lag_matrix + logodds_w_matrix
# print(logodds_covariates[1:3, 1:20])
# Log odds discrete indicators for time matrix
# diag(n_trt_times)

time_design_matrix <- diag(n_trt_times)
time_design_matrix[0:n_trt_times, 1] <- 1
# print(time_design_matrix)
logodds_t <- time_design_matrix %*% beta_time
# Predicted probability of treatment
logodds_t_matrix = matrix(data = rep(logodds_t, length(trt_obs)), ncol = length(trt_obs))
# logodds_t_matrix[1:3, 1:10]
trt_pred_matrix <- plogis(logodds_covariates + logodds_t_matrix)
trt_residual_matrix = (trt_obs_matrix - trt_pred_matrix) * in_risk_set

a_lag_score <- matrix(rep(1, n_trt_times), nrow = 1) %*% (trt_residual_matrix * trt_lag_matrix)
# dim(a_lag_score)
trt_residuals <- matrix(rep(1, n_trt_times), nrow = 1) %*% (trt_residual_matrix)
# dim(trt_residuals)
w_score <- trt_residuals * matrix(W, nrow = 1)
# dim(w_score)
t_score <- trt_residual_matrix
score <- rbind(a_lag_score, w_score, t_score)
return(score)
dim(score)
# print(trt_obs_matrix, max.print = 1000)
# print(trt_pred_matrix[1:3, -seq_len(9980)])
# print(trt_lag_matrix[1:3, -seq_len(9990)])
# print(trt_residual_matrix[1:3, 1:30])
# print(in_risk_set[1:3, 1:30])
# print(trt_lag_matrix[1:3, 1:10])

# Indicator for risk set
t_delta
unique_trt_times

# Indicator for in the risk set

# Predicted treatment

# tmp <- outer(X = unique_trt_times, Y = t_trt, FUN = "==")
# tmp[, 0:5]
# print(dim(outer(X = unique_trt_times, Y = t_trt, FUN = "==")))

plogis(x)
# !SECTION - Estimating equations

# Estimating equations rd (risk for strategy 1 minus strategy 0) output
#    estimate      variance    stderr       LCL       UCL measure
# 0  0.000507  1.598513e-07  0.000400 -0.000276  0.001291      rd
# 1  0.000628  4.306943e-07  0.000656 -0.000658  0.001914      rd
# 2  0.007631  1.654380e-05  0.004067 -0.000341  0.015603      rd
# 3  0.007937  2.662455e-05  0.005160 -0.002177  0.018050      rd
# 4  0.006742  3.522706e-05  0.005935 -0.004891  0.018374      rd
# 5  0.008141  5.143342e-05  0.007172 -0.005915  0.022198      rd

# Estimating equations r1 (risk for strategy 1) output
#    estimate  variance    stderr       LCL       UCL measure
# 0  0.010400  0.000001  0.001014  0.008412  0.012388   risk1
# 1  0.020900  0.000002  0.001430  0.018096  0.023704   risk1
# 2  0.037693  0.000017  0.004090  0.029678  0.045708   risk1
# 3  0.048913  0.000025  0.005024  0.039065  0.058760   risk1
# 4  0.059524  0.000032  0.005678  0.048396  0.070653   risk1
# 5  0.078058  0.000046  0.006796  0.064738  0.091379   risk1

# Estimating equations r0 (risk for strategy 0) output
#    estimate  variance    stderr       LCL       UCL measure
# 0  0.009893  0.000001  0.001044  0.007847  0.011939   risk0
# 1  0.020272  0.000002  0.001535  0.017264  0.023280   risk0
# 2  0.030062  0.000004  0.001922  0.026295  0.033828   risk0
# 3  0.040976  0.000005  0.002274  0.036519  0.045433   risk0
# 4  0.052783  0.000007  0.002588  0.047711  0.057854   risk0
# 5  0.069917  0.000009  0.002976  0.064085  0.075749   risk0
