#############################################################################
# SImulation for IPTW implementation estimating equations
# JHW (2026-08-05)
# Details: NA
# Making into a draft simulation for IPTW version
#############################################################################
 
##################################################
# Loading Packages
##################################################
# from importlib import reload

import numpy as np
import pandas as pd

# import functions_ee
from delicatessen import MEstimator

# import statsmodels.api as sm
# import statsmodels.formula.api as smf
from dgp import dgp_grace

# from deli_estimation import MEstimator
from functions_ee import clone_and_censor

# The family distribution (binomial) and link function (logit)
# f = sm.families.family.Binomial(link=sm.genmod.families.links.Logit())
# reload(functions_ee)


##################################################
# SECTION - Functions
##################################################
def inverse_logit(logodds):
    lodds = np.asarray(logodds)
    return 1 / (1 + np.exp(-lodds))

def psi_iptw(theta):
    beta_a_lag = np.asarray(theta[:1])
    beta_w = np.asarray(theta[1:2])
    beta_time = np.asarray(theta[2:])
    # * trt_residuals
    # Observed
    trt_obs_matrix = trt_obs * (t_trt == unique_trt_times[:, None].astype(int))
    # Risk Set
    in_risk_set = ((t_delta - 1) >= unique_trt_times[:, None]).astype(int)
    # in_risk_set = ((t_delta) >= unique_trt_times[:, None]).astype(int)

    # Predicted
    trt_lag_matrix = trt_obs_lag.T * ((t_trt_lag) == unique_trt_times[:, None]).astype(int)
    # Time under observation
    # final_time_underobs = ((t_delta - 1) == unique_trt_times[:, None]).astype(int) # indicator for the last point in time they are in the risk set
    # Log odds covariate matrix
    logodds_a_lag = (trt_lag_matrix * beta_a_lag) # broadcast multiplication with a scalar
    logodds_a_lag_matrix = logodds_a_lag * in_risk_set
    logodds_w = np.dot(W, beta_w)
    logodds_w_matrix = np.tile(logodds_w, reps=(n_trt_times, 1))
    logodds_covariates = logodds_a_lag_matrix + logodds_w_matrix
    # Log odds discrete indicators for time matrix
    time_design_matrix = np.identity(n=n_trt_times)
    time_design_matrix[:, 0] = 1
    logodds_t_matrix = np.dot(time_design_matrix, beta_time)
    # Predicted probability of treatment
    trt_pred_matrix = inverse_logit(logodds_covariates + logodds_t_matrix[:, None])
    # Trt residuals matrix
    trt_residual_matrix = (trt_obs_matrix - trt_pred_matrix) * in_risk_set

    # * a_lag_score
    tmp = trt_residual_matrix * trt_lag_matrix
    a_lag_score = np.dot(np.ones(shape=(1, n_trt_times)), tmp)

    # Use residuals then to calculate the score (here is sum of residuals)
    trt_residuals = np.dot(np.ones(shape=(1, n_trt_times)), trt_residual_matrix)
    # * w_score
    w_score = trt_residuals * W.T

    # * t_score (score for the discrete indicators for time matrix )
    t_score = trt_residual_matrix

    # * score
    score = np.vstack([a_lag_score, w_score, t_score])

    return score


# Create the universal weight matrix
def iptw_weights(theta):

    beta_a_lag = np.asarray(theta[:1])
    beta_w = np.asarray(theta[1:2])
    beta_time = np.asarray(theta[2:])

    trt_lag_mtrx = trt_obs_lag * ((t_trt_lag) == unique_trt_times[:, None]).astype(int)
    time_design_mtrx = np.identity(n=n_trt_times)
    time_design_mtrx[:, 0] = 1

    lodds_trt_lag_mtrx = trt_lag_mtrx * beta_a_lag
    lodds_w_mtrx = np.repeat(np.dot(W, beta_w)[None, :], repeats=n_trt_times, axis=0)
    lodds_t_mtrx = np.dot(time_design_mtrx, beta_time[:, None])

    pred_trt_mtrx = inverse_logit(lodds_trt_lag_mtrx + lodds_w_mtrx + lodds_t_mtrx)
    trt_obs_mtrx = trt_obs * (t_trt == unique_trt_times[:, None]).astype(int)
    pred_trt_observed_mtrx = (trt_obs_mtrx * pred_trt_mtrx) + (1 - trt_obs_mtrx) * (1 - pred_trt_mtrx)
    pred_trt_alltimepoints = np.ones(shape=(len(times_to_predict), nobs))
    pred_trt_alltimepoints[unique_trt_times] = pred_trt_observed_mtrx # in case the trt times are not 0, 1, 2...
    pred_trt_cumprod_mtrx = np.cumprod(pred_trt_alltimepoints, axis = 0)
    # Create the universal risk set (from unexpanded dataset)
    in_risk_set_all = ((t_delta - 1) >= times_to_predict[:, None]).astype(int)
    # in_risk_set_all = ((t_delta) >= times_to_predict[:, None]).astype(int)

    # Clones to plan 1 weight matrix
    cens1_t_mtrx = cens1 * (t_cens1 == times_to_predict[:, None]).astype(int)
    uncens1_t_mtrx = 1 - cens1_t_mtrx.cumsum(axis=0)
    cens1_risk_set_mtrx = uncens1_t_mtrx * in_risk_set_all
    # Weights for clones to trt plan 1
    pred_trt_cumprod_mtrx_gp = np.ones(shape=(len(times_to_predict), nobs))

    # * NOTE - USING THE CUMPROD All simulations before version 11
    pred_trt_cumprod_mtrx_gp[gp, :] = np.where( (t_trt == 2) & (trt_obs == 1), pred_trt_cumprod_mtrx[gp, :], 1) # CUMPROD simulations before version 11 have the cumprod version

    # Simulations for version 12 is trying the NOT cumprod version of the pred_trt_observed_mtrx
    # Then version 12 looks at the noncumprod version using the pred_trt_alltimepoints (not sure if this makes difference likely not) 
    # pred_trt_cumprod_mtrx_gp[gp, :] = np.where( (t_trt == 2) & (trt_obs == 1), pred_trt_alltimepoints[gp, :], 1) # NOT CUMPROD
    # NOTE - NOT CUMPROD, version 121, version 47 - just the non-cumprod predicted probability here 
    # pred_trt_cumprod_mtrx_gp[gp, :] = np.where( (t_trt == 2) & (trt_obs == 1), pred_trt_alltimepoints[gp, :], 1) # NOT CUMPROD, version 121, version 47 - just the non-cumprod predicted probability here 
    # pred_trt_cumprod_mtrx_gp[gp, :] =  pred_trt_alltimepoints[gp, :] # NOT CUMPROD, version 122 - just the non-cumprod predicted probability here 

    cens1_tmp_mtrx = cens1_risk_set_mtrx * np.power(pred_trt_cumprod_mtrx_gp, -1)
    cens1_weight_mtrx = np.cumprod(cens1_tmp_mtrx, axis=0)

    # Weights for clones to trt plan 0
    cens0_t_mtrx = cens0 * (t_cens0 == times_to_predict[:, None]).astype(int)
    uncens0_t_mtrx = 1 - cens0_t_mtrx.cumsum(axis=0)
    cens0_risk_set_mtrx = uncens0_t_mtrx * in_risk_set_all
    cens0_weight_mtrx = cens0_risk_set_mtrx * np.power(pred_trt_cumprod_mtrx, -1)
    # cens0_weight_mtrx = np.where(pred_trt_cumprod_mtrx == 0, 0, cens0_risk_set_mtrx * np.power(pred_trt_cumprod_mtrx, -1)) # WORKAROUND to prevent diving by zero?
    return cens1_weight_mtrx, cens0_weight_mtrx

# Create the universal weight matrix
# def iptw_weights_v2(theta):
# 
#     beta_a_lag = np.asarray(theta[:1])
#     beta_w = np.asarray(theta[1:2])
#     beta_time = np.asarray(theta[2:])
# 
#     trt_lag_mtrx = trt_obs_lag * ((t_trt_lag) == unique_trt_times[:, None]).astype(int)
#     time_design_mtrx = np.identity(n=n_trt_times)
#     time_design_mtrx[:, 0] = 1
# 
#     lodds_trt_lag_mtrx = trt_lag_mtrx * beta_a_lag
#     lodds_w_mtrx = np.repeat(np.dot(W, beta_w)[None, :], repeats=n_trt_times, axis=0)
#     lodds_t_mtrx = np.dot(time_design_mtrx, beta_time[:, None])
# 
#     pred_trt_mtrx = inverse_logit(lodds_trt_lag_mtrx + lodds_w_mtrx + lodds_t_mtrx)
#     trt_obs_mtrx = trt_obs * (t_trt == unique_trt_times[:, None]).astype(int)
#     pred_trt_observed_mtrx = (trt_obs_mtrx * pred_trt_mtrx) + (1 - trt_obs_mtrx) * (1 - pred_trt_mtrx)
#     pred_trt_alltimepoints = np.ones(shape=(len(times_to_predict), nobs))
#     pred_trt_alltimepoints[unique_trt_times] = pred_trt_observed_mtrx # in case the trt times are not 0, 1, 2...
#     pred_trt_cumprod_mtrx = np.cumprod(pred_trt_alltimepoints, axis = 0)
#     # Create the universal risk set (from unexpanded dataset)
#     in_risk_set_all = ((t_delta - 1) >= times_to_predict[:, None]).astype(int)
#     # in_risk_set_all = ((t_delta) >= times_to_predict[:, None]).astype(int)
# 
#     # Clones to plan 1 weight matrix
#     cens1_t_mtrx = cens1 * (t_cens1 == times_to_predict[:, None]).astype(int)
#     uncens1_t_mtrx = 1 - cens1_t_mtrx.cumsum(axis=0)
#     cens1_risk_set_mtrx = uncens1_t_mtrx * in_risk_set_all
#     # Weights for clones to trt plan 1
#     pred_trt_mtrx_gp2 = np.ones(shape=(len(times_to_predict), nobs))
#     # USING THE CUMPROD All simulations before version 11
#     pred_trt_mtrx_gp2[0:(gp+1), :] = np.where( (t_trt == 2) & (trt_obs == 1), pred_trt_alltimepoints[0:(gp+1), :], 1) # CUMPROD simulations before version 11 have the cumprod version
#     pred_trt_mtrx_gp2_cumprod = np.cumprod(pred_trt_mtrx_gp2, axis=0)
#     cens1_weight_mtrx2 = cens1_risk_set_mtrx * np.power(pred_trt_mtrx_gp2_cumprod, -1)
# 
#     # Weights for clones to trt plan 0
#     cens0_t_mtrx = cens0 * (t_cens0 == times_to_predict[:, None]).astype(int)
#     uncens0_t_mtrx = 1 - cens0_t_mtrx.cumsum(axis=0)
#     cens0_risk_set_mtrx = uncens0_t_mtrx * in_risk_set_all
#     cens0_weight_mtrx = cens0_risk_set_mtrx * np.power(pred_trt_cumprod_mtrx, -1)
#     # cens0_weight_mtrx = np.where(pred_trt_cumprod_mtrx == 0, 0, cens0_risk_set_mtrx * np.power(pred_trt_cumprod_mtrx, -1)) # WORKAROUND to prevent diving by zero?
#     return cens1_weight_mtrx2, cens0_weight_mtrx


# def psi_msm_v2(theta, weight_matrix):
#     # Initiate parameters
#     beta_time_predict = np.asarray(theta)
#     # No intercept version 
#     time_design_matrix = np.identity(n=n_times_to_predict)
# 
#     # Log odds times to predict 
#     lodds_t_to_predict = np.dot(time_design_matrix, beta_time_predict[:, None])
#     delta_pred_mtrx = inverse_logit(lodds_t_to_predict)
#     delta_obs_mtrx = delta * ((t_delta - 1) == times_to_predict[:, None]).astype(int)
#     delta_residual_matrix = (delta_obs_mtrx - delta_pred_mtrx) * weight_matrix
#     return delta_residual_matrix
# 
# def pred_risk_v2(theta):
#     beta_t = np.asarray(theta)
#     # Matrices
#     time_design_matrix = np.identity(n=n_times_to_predict)
#     # Logodds 
#     lodds_t_to_predict = np.dot(time_design_matrix, beta_t[:, None])
#     # Predictions
#     delta_pred = inverse_logit(lodds_t_to_predict)
#     delta_pred_haz = np.zeros(shape=(n_times_to_predict, delta.shape[0]))
#     delta_pred_haz[times_to_predict, :] = delta_pred
#     delta_pred_surv = np.cumprod(1 - delta_pred_haz, axis=0)
#     delta_pred_risk = 1 - delta_pred_surv
#     return delta_pred_risk
# 
# 
# def psi_ccw_iptw_v2(theta):
#     ### Init Betas for the function
#     index_msm = n_times_to_predict*5
#     theta_rd = theta[:n_times_to_predict]
#     theta_r1 = theta[n_times_to_predict:(n_times_to_predict*2)]
#     theta_r0 = theta[(n_times_to_predict*2):(n_times_to_predict*3)]
#     theta_msm_r1 = theta[(n_times_to_predict*3):(n_times_to_predict*4)]
#     theta_msm_r0 = theta[(n_times_to_predict*4):index_msm]
#     theta_iptw = theta[index_msm:]
# 
#     # Estimating equations and transformed intermediates
#     ee_iptw = psi_iptw(theta=theta_iptw) 
#     # Intermediate step 
#     weight_matrix_c1, weight_matrix_c0 = iptw_weights(theta=theta_iptw)
#     # Modeling step
#     ee_msm_r1 = psi_msm_v2(theta=theta_msm_r1, weight_matrix=weight_matrix_c1)
#     ee_msm_r0 = psi_msm_v2(theta=theta_msm_r0, weight_matrix=weight_matrix_c0)
#     # Intermediate step
#     pred_risk_c1 = pred_risk_v2(theta=theta_msm_r1)  
#     pred_risk_c0 = pred_risk_v2(theta=theta_msm_r0)  
#     # Nonparametric step
#     ee_risk_1 = pred_risk_c1 - np.asarray(theta_r1)[:, None] 
#     ee_risk_0 = pred_risk_c0 - np.asarray(theta_r0)[:, None] 
# 
#     ee_rd = (pred_risk_c1 - pred_risk_c0) - np.asarray(theta_rd)[:, None] 
# 
#     ee_stack = np.vstack([ee_rd, ee_risk_1, ee_risk_0, ee_msm_r1, ee_msm_r0, ee_iptw]) 
# 
#     return ee_stack


#############################################
# OUTCOME USING Y ~ C(T_OUT) + INTERCEPT
#############################################
def psi_msm_v3(theta, weight_matrix):
    # Initiate parameters
    beta_time_predict = np.asarray(theta)
    # WITH intercept version 
    time_design_matrix = np.identity(n=n_times_to_predict)
    time_design_matrix[:, 0] = 1

    # Log odds times to predict 
    lodds_t_to_predict = np.dot(time_design_matrix, beta_time_predict[:, None])
    delta_pred_mtrx = inverse_logit(lodds_t_to_predict)
    delta_obs_mtrx = delta * ((t_delta - 1) == times_to_predict[:, None]).astype(int)
    delta_residual_matrix = (delta_obs_mtrx - delta_pred_mtrx) * weight_matrix
    return delta_residual_matrix   

def pred_risk_v3(theta):
    beta_t = np.asarray(theta)
    # Matrices
    time_design_matrix = np.identity(n=n_times_to_predict)
    time_design_matrix[:, 0] = 1
    # Logodds 
    lodds_t_to_predict = np.dot(time_design_matrix, beta_t[:, None])
    # Predictions
    delta_pred = inverse_logit(lodds_t_to_predict)
    delta_pred_haz = np.zeros(shape=(n_times_to_predict, delta.shape[0]))
    delta_pred_haz[times_to_predict, :] = delta_pred
    delta_pred_surv = np.cumprod(1 - delta_pred_haz, axis=0)
    delta_pred_risk = 1 - delta_pred_surv
    return delta_pred_risk


def psi_ccw_iptw_v3(theta):
    ### Init Betas for the function
    index_msm = n_times_to_predict*5
    theta_rd = theta[:n_times_to_predict]
    theta_r1 = theta[n_times_to_predict:(n_times_to_predict*2)]
    theta_r0 = theta[(n_times_to_predict*2):(n_times_to_predict*3)]
    theta_msm_r1 = theta[(n_times_to_predict*3):(n_times_to_predict*4)]
    theta_msm_r0 = theta[(n_times_to_predict*4):index_msm]
    theta_iptw = theta[index_msm:]

    # Estimating equations and transformed intermediates
    ee_iptw = psi_iptw(theta=theta_iptw) 
    # Intermediate step 
    weight_matrix_c1, weight_matrix_c0 = iptw_weights(theta=theta_iptw)
    # Modeling step
    ee_msm_r1 = psi_msm_v3(theta=theta_msm_r1, weight_matrix=weight_matrix_c1)
    ee_msm_r0 = psi_msm_v3(theta=theta_msm_r0, weight_matrix=weight_matrix_c0)
    # Intermediate step
    pred_risk_c1 = pred_risk_v3(theta=theta_msm_r1)  
    pred_risk_c0 = pred_risk_v3(theta=theta_msm_r0)  
    # Nonparametric step
    ee_risk_1 = pred_risk_c1 - np.asarray(theta_r1)[:, None] 
    ee_risk_0 = pred_risk_c0 - np.asarray(theta_r0)[:, None] 

    # ee_rd = (pred_risk_c1 - pred_risk_c0) - np.asarray(theta_rd)[:, None]  #estimating equation for version 46-- doesn't matter if predicted or thetas 

    ee_rd = np.ones((n_times_to_predict, ee_risk_1.shape[1]))*((np.asarray(theta_r1)[:, None] - np.asarray(theta_r0)[:, None]) - np.asarray(theta_rd)[:, None]) # estimating eq in version 45

    ee_stack = np.vstack([ee_rd, ee_risk_1, ee_risk_0, ee_msm_r1, ee_msm_r0, ee_iptw]) 

    return ee_stack


# !SECTION -

##################################################
# SECTION - Runs
##################################################
# risk1 = pd.read_csv("true_risk1.csv")
risk1 = pd.read_csv("true_risk1_v1.csv")
r11, r12, r13, r14, r15, r16 = np.asarray(risk1['Surv prob'])

risk0 = pd.read_csv("true_risk0.csv")
r01, r02, r03, r04, r05, r06 = np.asarray(risk0['Surv prob'])

truth = risk1 - risk0
# truth.to_csv("true_rd.csv")

#############################################
# Starting the runs
#############################################

t1, t2, t3, t4, t5, t6 = np.asarray(truth["Surv prob"])
r1, r2, r3, r4, r5, r6 = [], [], [], [], [], []

# rng_runs = np.random.default_rng(209_309)
rng_runs = np.random.default_rng(319_031_111)
gp = 2
n_trts = 2
# runs=5_000
runs= 100_000
# runs = 2_000
# runs = 5_000
# nobs = 2_000
# nobs = 10_000
nobs = 3_000
# nobs = 5_000

vers = 101 # Version 46, but with 3000 obs and 100,000 iterations for MC simulation on the longleaf

print(f"version: {vers}")
print(f"nobs: {nobs}")
print(f"runs: {runs}")


for i in range(runs):
# if i == 1986:
    print(f"run: {i+1}")
    r1_row, r2_row, r3_row, r4_row, r5_row, r6_row = [], [], [], [], [], []
    # Create dgp
    random_integer = rng_runs.integers(low=15, high=100_000_000, size=1)
    # Get data ready for ee
    # Get data ready for ee
    # # d_obs = dgp_grace(k=6, grace=gp, nobs=nobs, timevarying=0, rngseed=random_integer + (2 * i), plan=None)
    d_obs = dgp_grace(k=6, grace=gp, nobs=nobs, timevarying=0, rngseed=random_integer + (1 * i), plan=None)
    # Creating the cloned and censored data
    d = clone_and_censor(d_obs, gp=2, type='d_vec_orig')
    d_c0 = clone_and_censor(d_obs, gp=2, type='d_vec_c0')
    d_c1 = clone_and_censor(d_obs, gp=2, type='d_vec_c1')
    del d_obs

    ### Input from the original dataset
    # Covariate
    W = np.asarray(d[['W']])
    # For the observed treatment
    trt_obs = np.asarray(d['trt_obs'])
    # n_trts = len(np.unique(trt_obs))
    t_trt = np.asarray(d['t_trt'])
    trt_times = np.asarray(d.loc[((d['t_trt'] <= gp) & (d['trt_obs'] == 1)), 't_trt'])
    unique_trt_times = np.unique(trt_times)
    n_trt_times = unique_trt_times.shape[0]
    # History of treatment
    trt_obs_lag = np.asarray(d['trt_obs_lag'])
    t_trt_lag = np.asarray(d['t_trt_lag']) # lag time of the treatment lag
    # For the risk set matrix
    delta = np.asarray(d['event'])
    t_delta = np.asarray(d['t_out'])

    ### Input from the censored clones
    trt_clone1 = np.asarray(d_c1[['trt_plan']])
    cens1 = np.asarray(d_c1['censor'])
    t_cens1 = np.asarray(d_c1['t_in_cens'])
    delta1 = np.asarray(d_c1['event'])
    t_delta1 = np.asarray(d_c1['t_out'])

    trt_clone0 = np.asarray(d_c0[['trt_plan']])
    cens0 = np.asarray(d_c0['censor'])
    t_cens0 = np.asarray(d_c0['t_in_cens'])
    delta0 = np.asarray(d_c0['event'])
    t_delta0 = np.asarray(d_c0['t_out'])

    ### All time points to predict
    times_to_predict = np.unique(t_delta - 1)
    n_times_to_predict = len(times_to_predict)

    ### Creat inits
    # For the IPTW from the original data
    init_a_lag = [-24.0,]
    init_w = [-0.1309, ]
    init_time = [-1.8034, -0.2509, -0.2721]
    # init_a_lag = [-22.0,]
    # init_w = [0.1, ]
    # init_time = [0.2, 0.5, 0.3]
    inits_iptw = init_a_lag + init_w + init_time
    # For the predicted risks outcome
    init_trt = [0.348279,]
    init_time_y1 = [-2.3807, -2.206558, -1.1257, -2.748212, -2.855993, -2.816546]
    init_time_y0 = [0.1158363 , 0.12840615, 0.16394126, 0.17089769, 0.17709666, 0.1834949 ]
    # init_trt = [0.4,]
    # init_time_y = [-2, -2.1, -1, -3, -3.2, -3.1]

    # For the risks
    init_risks1 = [0.115, 0.13, 0.17970892, 0.17970892, 0.18355848, 0.18355848]
    # init_risks1 = [0.15, ] * n_times_to_predict
    init_risks0 = [0.0854, 0.092283, 0.099975, 0.114262, 0.121953, 0.135142]
    # init_risks0 = [0.10, ] * n_times_to_predict
    #  For risk differences
    init_rd = [0.0295, 0.0377, 0.0797, 0.0654, 0.0616, 0.0484]
    # init_rd = [0.02] * n_times_to_predict

    inits = init_rd + init_risks1 + init_risks0 + init_time_y1 +init_time_y0 + inits_iptw
    theta = inits

    ### Init Betas for the function
    index_msm = n_times_to_predict*5

    theta_rd = theta[:n_times_to_predict]
    theta_r1 = theta[n_times_to_predict:(n_times_to_predict*2)]
    theta_r0 = theta[(n_times_to_predict*2):(n_times_to_predict*3)]
    theta_msm_r1 = theta[(n_times_to_predict*3):(n_times_to_predict*4)]
    theta_msm_r0 = theta[(n_times_to_predict*4):index_msm]
    theta_iptw = theta[index_msm:]

    # Now estimate
    # ccw_estr = MEstimator(psi_ccw_iptw_v2, init=inits)
    ccw_estr = MEstimator(psi_ccw_iptw_v3, init=inits)
    ccw_estr.estimate()
    estr_theta = ccw_estr.theta
    # Now take the rd outputs

    point_rd = estr_theta[:n_times_to_predict]
    point_r1 = estr_theta[n_times_to_predict:(n_times_to_predict*2)]
    point_r0 = estr_theta[(n_times_to_predict*2):(n_times_to_predict*3)]
    point_msm_r1 = estr_theta[(n_times_to_predict*3):(n_times_to_predict*4)]
    point_msm_r0 = estr_theta[(n_times_to_predict*4):index_msm]
    point_iptw = estr_theta[index_msm:]

    variance_rd = (np.diag(ccw_estr.variance))[:n_times_to_predict]
    variance_r1 = (np.diag(ccw_estr.variance))[n_times_to_predict:(n_times_to_predict*2)]
    variance_r0 = (np.diag(ccw_estr.variance))[(n_times_to_predict*2):(n_times_to_predict*3)]
    variance_msm_r1 = (np.diag(ccw_estr.variance))[(n_times_to_predict*3):(n_times_to_predict*4)]
    variance_msm_r0 = (np.diag(ccw_estr.variance))[(n_times_to_predict*4):index_msm]
    variance_iptw = (np.diag(ccw_estr.variance))[index_msm:]

    lcl_rd = (ccw_estr.confidence_intervals())[:n_times_to_predict, 0]
    lcl_r1 = (ccw_estr.confidence_intervals())[n_times_to_predict:(n_times_to_predict*2), 0]
    lcl_r0 = (ccw_estr.confidence_intervals())[(n_times_to_predict*2):(n_times_to_predict*3), 0]
    lcl_msm_r1 = (ccw_estr.confidence_intervals())[(n_times_to_predict*3):(n_times_to_predict*4), 0]
    lcl_msm_r0 = (ccw_estr.confidence_intervals())[(n_times_to_predict*4):index_msm, 0]
    lcl_iptw = (ccw_estr.confidence_intervals())[index_msm:, 0]

    ucl_rd = (ccw_estr.confidence_intervals())[:n_times_to_predict, 1]
    ucl_r1 = (ccw_estr.confidence_intervals())[n_times_to_predict:(n_times_to_predict*2), 1]
    ucl_r0 = (ccw_estr.confidence_intervals())[(n_times_to_predict*2):(n_times_to_predict*3), 1]
    ucl_msm_r1 = (ccw_estr.confidence_intervals())[(n_times_to_predict*3):(n_times_to_predict*4), 1]
    ucl_msm_r0 = (ccw_estr.confidence_intervals())[(n_times_to_predict*4):index_msm, 1]
    ucl_iptw = (ccw_estr.confidence_intervals())[index_msm:, 1]
# 
#     ## FOR RISKS ONLY 
#     for j in [ point_r1, variance_r1, lcl_r1, ucl_r1,
#               point_r0, variance_r0, lcl_r0, ucl_r0]:
#         r1_row.append(j[0])
#         r2_row.append(j[1])
#         r3_row.append(j[2])
#         r4_row.append(j[3])
#         r5_row.append(j[4])
#         r6_row.append(j[5])
#     # Now store into a table
#     r1.append(r1_row)
#     r2.append(r2_row)
#     r3.append(r3_row)
#     r4.append(r4_row)
#     r5.append(r5_row)
#     r6.append(r6_row)
# 
#     if ((0 < i) & ((i + 1) % 500 == 0)):
#         cols = [ 'point_r1', 'variance_r1', 'lcl_r1', 'ucl_r1',
#               'point_r0', 'variance_r0', 'lcl_r0', 'ucl_r0']
#         for t_out, t_results, truth, risk1, risk0 in zip([1, 2, 3, 4, 5, 6], [r1, r2, r3, r4, r5, r6], [t1, t2, t3, t4, t5, t6], [r11, r12, r13, r14, r15, r16], [r01, r02, r03, r04, r05, r06]):
#             results = pd.DataFrame(t_results, columns=cols)
#             results["t_out"] = t_out
#             results["truth_rd"] = truth
#             results["truth_r1"] = risk1
#             results["truth_r0"] = risk0
#             results.to_csv(f"./out/v{vers}sim_iptw_{t_out}_n{nobs}_{i}.csv")
# 
            # For RISKS and RISK DIFFERENCE 
    for j in [point_rd, variance_rd, lcl_rd, ucl_rd,
              point_r1, variance_r1, lcl_r1, ucl_r1,
              point_r0, variance_r0, lcl_r0, ucl_r0]:
        r1_row.append(j[0])
        r2_row.append(j[1])
        r3_row.append(j[2])
        r4_row.append(j[3])
        r5_row.append(j[4])
        r6_row.append(j[5])
    # Now store into a table
    r1.append(r1_row)
    r2.append(r2_row)
    r3.append(r3_row)
    r4.append(r4_row)
    r5.append(r5_row)
    r6.append(r6_row)

    if ((0 < i) & ((i + 1) % 500 == 0)):
    # if ((0 < i) & (i >= 1)):
        cols = ['point_rd', 'variance_rd', 'lcl_rd', 'ucl_rd',
              'point_r1', 'variance_r1', 'lcl_r1', 'ucl_r1',
              'point_r0', 'variance_r0', 'lcl_r0', 'ucl_r0']
        for t_out, t_results, truth, risk1, risk0 in zip([1, 2, 3, 4, 5, 6], [r1, r2, r3, r4, r5, r6], [t1, t2, t3, t4, t5, t6], [r11, r12, r13, r14, r15, r16], [r01, r02, r03, r04, r05, r06]):
            results = pd.DataFrame(t_results, columns=cols)
            results["t_out"] = t_out
            results["truth_rd"] = truth
            results["truth_r1"] = risk1
            results["truth_r0"] = risk0
            results.to_csv(f"./out/v{vers}sim_tout_{t_out}_n{nobs}_{i}.csv")


##################################################
# SECTION - Now taking the results and evaluating performance
##################################################

# I want to calculate the risks only
# FOR RISKS ONLY 
# cols = [ 'point_r1', 'variance_r1', 'lcl_r1', 'ucl_r1',
#         'point_r0', 'variance_r0', 'lcl_r0', 'ucl_r0']
# for t_out, t_results, truth, risk1, risk0 in zip([1, 2, 3, 4, 5, 6], [r1, r2, r3, r4, r5, r6], [t1, t2, t3, t4, t5, t6], [r11, r12, r13, r14, r15, r16], [r01, r02, r03, r04, r05, r06]):
#     results = pd.DataFrame(t_results, columns=cols)
#     results["t_out"] = t_out
#     results["truth_rd"] = truth
#     results["truth_r1"] = risk1
#     results["truth_r0"] = risk0
#     results.to_csv(f"./out/v{vers}finsim_tout_{t_out}_n{nobs}_{runs}.csv")
# print(f"./out/v{vers}finsim_iptw_{t_out}_n{nobs}_{runs}.csv")

# # FOR RISKS AND RISK DIFFERENCES 
# I want to calculate the rd, r1, r0
cols = ['point_rd', 'variance_rd', 'lcl_rd', 'ucl_rd',
        'point_r1', 'variance_r1', 'lcl_r1', 'ucl_r1',
        'point_r0', 'variance_r0', 'lcl_r0', 'ucl_r0']
for t_out, t_results, truth, risk1, risk0 in zip([1, 2, 3, 4, 5, 6], [r1, r2, r3, r4, r5, r6], [t1, t2, t3, t4, t5, t6], [r11, r12, r13, r14, r15, r16], [r01, r02, r03, r04, r05, r06]):
    results = pd.DataFrame(t_results, columns=cols)
    results["t_out"] = t_out
    results["truth_rd"] = truth
    results["truth_r1"] = risk1
    results["truth_r0"] = risk0
    results.to_csv(f"./out/v{vers}iptwsim_tout_{t_out}_n{nobs}_{runs}.csv")
# V12 tries to see if I put the wrong probability for my plan risk 1 probability of remaining uncensored. 
# V10 is fixing the weight matrix (to see if np.power is less problematic) and repeating the simulation with all risks and risk difference (do not expect to see difference here) 
# V9 IS RISKS ONLY fixing the weight matrix and trying out the np.power so that the weight matrix does not divide by zero 
# V8 IS RISKS ONLY 
# V7 IS RISKS AND RISK DIFFERENCE 
# !SECTION -
##################################################
# SECTION - test
##################################################

# !SECTION - test