#############################################################################
# EE and bootstrap comparison of the IPTW implementation of the CCW 
# JHW (2026-08-06)
# Details: NA
#############################################################################

##################################################
# Loading Packages
##################################################

import time

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from dgp import dgp_grace

from deli_estimation import MEstimator
from functions_ee import clone_and_censor

# Establish the family distribution (binomial) and link function (logit)
f = sm.families.family.Binomial(link=sm.genmod.families.links.Logit())
##################################################
# SECTION - DATA 
##################################################
date = "2026-08-12"

rng_runs = np.random.default_rng(319_031_111)
gp=2
nobs=10_000
i = 1
# Create dgp
random_integer = rng_runs.integers(low=15, high=100_000_000, size=1)
# Get data ready for ee 
d_obs = dgp_grace(k=6, grace=gp, nobs=nobs, timevarying=0, rngseed=random_integer + (2 * i), plan=None)

# trt_observed = d_obs[['id', 'plan_000','plan_100', 'plan_010', 'plan_001', 'plan_NA']].groupby('id').first().value_counts()
# print((trt_observed.array)/nobs)

d_cens1 = clone_and_censor(d_obs, gp=2, type = 'da_long').copy()
d_cens1['trt_clone'] = 1
d_cens1["_idx"] = (d_cens1[["id"]]).apply(tuple)
d_cens1 = d_cens1.set_index("_idx")

d_cens0 = clone_and_censor(d_obs, gp=2, type = 'db_long').copy()
d_cens0['trt_clone'] = 0
d_cens0["_idx"] = (d_cens0[["id"]]).apply(tuple)
d_cens0 = d_cens0.set_index("_idx")

d = clone_and_censor(d_obs, gp=2, type='d_vec_orig')
d_c0 = clone_and_censor(d_obs, gp=2, type='d_vec_c0')
d_c1 = clone_and_censor(d_obs, gp=2, type='d_vec_c1')

# Output interim as csv for R
d_obs.to_csv("./data/d_obs_long.csv")
d_cens1.to_csv("./data/d_cens1_long.csv")
d_cens0.to_csv("./data/d_cens0_long.csv")
d.to_csv("./data/d_obs_wide.csv")
d_c0.to_csv("./data/d_cens0_wide.csv")
d_c1.to_csv("./data/d_cens1_wide.csv")

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
# !SECTION - DATA 

##################################################
# SECTION - EE IPTW FUNCTIONS 
##################################################
#############################################
# OUTCOME USING Y ~ C(T_OUT) + INTERCEPT
#############################################
def inverse_logit(logodds):
    lodds = np.asarray(logodds)
    return 1 / (1 + np.exp(-lodds))

def psi_iptw(theta):
    beta_a_lag = np.asarray(theta[:1])
    beta_w = np.asarray(theta[1:2])
    beta_time = np.asarray(theta[2:])
    # * trt_residuals
    # Observed
    trt_obs_matrix = trt_obs * (t_trt == unique_trt_times[:, None]).astype(int)
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

def psi_msm_v3(theta, weight_matrix):
    # Initiate parameters
    beta_t = np.asarray(theta)
    # WITH intercept version 
    time_design_matrix = np.identity(n=n_times_to_predict)
    time_design_matrix[:, 0] = 1

    # Log odds times to predict 
    lodds_t_to_predict = np.dot(time_design_matrix, beta_t[:, None])
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

# !SECTION - EE IPTW FUNCTIONS 

##################################################
# SECTION - ESTIMATING EQUATIONS
##################################################
# Now estimate
# ccw_estr = MEstimator(psi_ccw_iptw_v2, init=inits)
t0_ee = time.perf_counter()
ccw_estr = MEstimator(psi_ccw_iptw_v3, init=inits)
ccw_estr.estimate()
t1_ee = time.perf_counter()
print(t1_ee - t0_ee)
# 3.7499153000098886 seconds
with open(f"./time_ee_{date}.txt", "w", encoding="utf-8") as file:
    print(f"For size = {nobs} study pop, Elapsed for estimating equations B=1 iters: {((t1_ee - t0_ee)/60):.3f} mins OR {(t1_ee - t0_ee):.3f} secs", file=file)
estr_theta = ccw_estr.theta
print(ccw_estr.theta)
# MEstimator python output
# [ 5.07305976e-04  6.28023085e-04  7.63124750e-03  7.93661880e-03
#   6.74153490e-03  8.14138394e-03  1.04000000e-02  2.09000000e-02
#   3.76929161e-02  4.89126436e-02  5.95242235e-02  7.80582762e-02
#   9.89269402e-03  2.02719769e-02  3.00616686e-02  4.09760248e-02
#   5.27826886e-02  6.99168922e-02 -4.55549501e+00  2.02364897e-02
#   5.07118536e-01  1.15562719e-01  7.10549623e-02  6.48622726e-01
#  -4.60601682e+00  5.85536038e-02  1.01142858e-02  1.30179758e-01
#   2.21152373e-01  6.11819554e-01 -3.30946845e+01 -5.64464801e-01
#  -1.96546575e+00  1.28440083e-01  1.43361337e-02]
# Now take the rd outputs
estr_theta[index_msm:]
# Output 
point_rd = estr_theta[:n_times_to_predict]
point_r1 = estr_theta[n_times_to_predict:(n_times_to_predict*2)]
point_r0 = estr_theta[(n_times_to_predict*2):(n_times_to_predict*3)]
variance_rd = (np.diag(ccw_estr.variance))[:n_times_to_predict]
print(np.sqrt(variance_rd))
variance_r1 = (np.diag(ccw_estr.variance))[n_times_to_predict:(n_times_to_predict*2)]
print(np.sqrt(variance_r1))
variance_r0 = (np.diag(ccw_estr.variance))[(n_times_to_predict*2):(n_times_to_predict*3)]
print(np.sqrt(variance_r0))
lcl_rd = (ccw_estr.confidence_intervals())[:n_times_to_predict, 0]
lcl_r1 = (ccw_estr.confidence_intervals())[n_times_to_predict:(n_times_to_predict*2), 0]
lcl_r0 = (ccw_estr.confidence_intervals())[(n_times_to_predict*2):(n_times_to_predict*3), 0]
ucl_rd = (ccw_estr.confidence_intervals())[:n_times_to_predict, 1]
ucl_r1 = (ccw_estr.confidence_intervals())[n_times_to_predict:(n_times_to_predict*2), 1]
ucl_r0 = (ccw_estr.confidence_intervals())[(n_times_to_predict*2):(n_times_to_predict*3), 1]

# Risk differences 
ee_rd_out = pd.DataFrame({"estimate": np.asarray(point_rd)})
ee_rd_out['variance'] = np.asarray(variance_rd)
ee_rd_out['stderr'] = np.asarray(np.sqrt(variance_rd))
ee_rd_out['LCL'] = np.asarray(lcl_rd)
ee_rd_out['UCL'] = np.asarray(ucl_rd)
ee_rd_out['measure'] = "rd"
print(ee_rd_out)
#    estimate      variance    stderr       LCL       UCL measure
# 0  0.000507  1.598513e-07  0.000400 -0.000276  0.001291      rd
# 1  0.000628  4.306943e-07  0.000656 -0.000658  0.001914      rd
# 2  0.007631  1.654380e-05  0.004067 -0.000341  0.015603      rd
# 3  0.007937  2.662455e-05  0.005160 -0.002177  0.018050      rd
# 4  0.006742  3.522706e-05  0.005935 -0.004891  0.018374      rd
# 5  0.008141  5.143342e-05  0.007172 -0.005915  0.022198      rd
# Risk 1 
ee_r1_out = pd.DataFrame({"estimate": np.asarray(point_r1)})
ee_r1_out['variance'] = np.asarray(variance_r1)
ee_r1_out['stderr'] = np.asarray(np.sqrt(variance_r1))
ee_r1_out['LCL'] = np.asarray(lcl_r1)
ee_r1_out['UCL'] = np.asarray(ucl_r1)
ee_r1_out['measure'] = "risk1"
print(ee_r1_out)
#    estimate  variance    stderr       LCL       UCL measure
# 0  0.121000  0.000011  0.003261  0.114608  0.127392   risk1
# 1  0.130400  0.000011  0.003367  0.123800  0.137000   risk1
# 2  0.145939  0.000023  0.004810  0.136511  0.155367   risk1
# 3  0.154989  0.000029  0.005420  0.144366  0.165612   risk1
# 4  0.165317  0.000036  0.006003  0.153552  0.177082   risk1
# 5  0.181853  0.000047  0.006836  0.168456  0.195251   risk1
# Risk 0
ee_r0_out = pd.DataFrame({"estimate": np.asarray(point_r0)})
ee_r0_out['variance'] = np.asarray(variance_r0)
ee_r0_out['stderr'] = np.asarray(np.sqrt(variance_r0))
ee_r0_out['LCL'] = np.asarray(lcl_r0)
ee_r0_out['UCL'] = np.asarray(ucl_r0)
ee_r0_out['measure'] = "risk0"
print(ee_r0_out)
#    estimate  variance    stderr       LCL       UCL measure
# 0  0.009893  0.000001  0.001044  0.007847  0.011939   risk0
# 1  0.020272  0.000002  0.001535  0.017264  0.023280   risk0
# 2  0.030062  0.000004  0.001922  0.026295  0.033828   risk0
# 3  0.040976  0.000005  0.002274  0.036519  0.045433   risk0
# 4  0.052783  0.000007  0.002588  0.047711  0.057854   risk0
# 5  0.069917  0.000009  0.002976  0.064085  0.075749   risk0
print([point_rd, np.sqrt(variance_rd), point_r1, np.sqrt(variance_r1), point_r0, np.sqrt(variance_r0)])

# !SECTION - ESTIMATING EQUATIONS

##################################################
# SECTION - TRADITIONAL IMPLEMENTATION
##################################################

# Estimating IPTW 
m_iptw_cut2 = smf.glm(formula='A ~  A_lag + W + C(t_in)', data=d_obs.loc[d_obs['t_in'] <= gp], family=f)
# m_iptw_cut2 = smf.glm(formula='A ~  A_lag2 + W + C(t_in)', data=d_obs, family=f)
logit_iptw_cut2 = m_iptw_cut2.fit()
logit_iptw_cut2.summary()
print(logit_iptw_cut2.params)
# Intercept       -1.965466
# C(t_in)[T.1]     0.128440
# C(t_in)[T.2]     0.014336
# A_lag          -25.573781
# W               -0.564465
# dtype: float64
# Compare with the estimating equations: 
with np.printoptions(formatter={"float_kind": lambda value: f"{value:4f}"}):
    print(estr_theta[index_msm:])
# [-33.094684 -0.564465 -1.965466 0.128440 0.014336]

# Version where A_lag is always 0 for t_in=0
d_iptw = d_obs.copy()
d_iptw['pred_trt'] = logit_iptw_cut2.predict(d_iptw.loc[d_iptw['t_in'] <= gp])
d_iptw['pred_trt'] = d_iptw['pred_trt'].fillna(0)
d_iptw['pred_notrt']  = 1 - d_iptw['pred_trt']
d_iptw['pred_observedtrt'] = np.where((d_iptw['A'] == 1) & (d_iptw['t_in'] <= gp), d_iptw['pred_trt'], d_iptw['pred_notrt'] )
d_iptw['pred_trt_denom'] = d_iptw.groupby('id')['pred_observedtrt'].cumprod()
d_iptw['iptw_wgt'] = 1 / d_iptw['pred_trt_denom']
d_iptw['iptw_wgt'].value_counts()
d_cens1_i = d_cens1
d_cens1_i['trt_clone'] = 1
d_cens1_i = pd.merge(left=d_cens1_i, right=d_iptw[['id', 't_in', 'pred_observedtrt', 'pred_trt_denom']])
d_cens1_i['censor_1'] = d_cens1_i['censor']
# Adjusting the weights for each one 
d_cens1_i['pred_trt_plan1'] = np.where((d_cens1_i['t_in'] == gp), d_cens1_i['pred_trt_denom'], 1)
d_cens1_i['pr_trt1_denom'] = d_cens1_i.groupby('id')['pred_trt_plan1'].cumprod()
d_cens1_i['iptw_plan1_set1'] = np.where((d_cens1_i['plan_100'] == 1) | (d_cens1_i['plan_010'] == 1), 1, 1/d_cens1_i['pr_trt1_denom'])
d_cens1_i['iptw_clones'] = d_cens1_i['iptw_plan1_set1'] 
d_cens1_i.loc[d_cens1_i['censor'] == 0,:]['iptw_clones'].value_counts()
d_cens0_i = d_cens0
d_cens0_i['trt_clone'] = 0
d_cens0_i = pd.merge(left = d_cens0_i, right = d_iptw[['id', 't_in', 'pred_trt', 'pred_observedtrt','pred_trt_denom']], how='left', left_on=['id', 't_in'], right_on=['id', 't_in'])
d_cens0_i['censor_0'] = d_cens0_i['censor']
d_cens0_i['iptw_plan0'] = 1/d_cens0_i['pred_trt_denom']
d_cens0_i['iptw_clones'] = d_cens0_i['iptw_plan0']
d_cens0_i.loc[d_cens0_i['censor'] == 0,:]['iptw_clones'].value_counts()


##################################################
# Point Estimates  
##################################################
d_long = pd.concat([d_cens0_i, d_cens1_i], axis=0)
d_plan_long = d_long.loc[d_long['censor'] == 0].copy()

# Risk 1

# Or regression separately - there is discrepancy in aggreement likely because there is less information to borrow from without stacked observations. 
# Outcome model 
m_out = smf.glm(formula='Y ~ C(t_out)', family=f, data=d_plan_long.loc[d_plan_long['trt_clone'] == 1], freq_weights=d_plan_long.loc[d_plan_long['trt_clone'] == 1,'iptw_clones'])
logit_out = m_out.fit()
print(logit_out.params)
# Intercept       -4.555495
# C(t_out)[T.2]    0.020236
# C(t_out)[T.3]    0.507119
# C(t_out)[T.4]    0.115563
# C(t_out)[T.5]    0.071055
# C(t_out)[T.6]    0.648623
# dtype: float64
trt_1 = d_plan_long.loc[d_plan_long['trt_clone'] == 1].copy()
trt_1['predY'] = logit_out.predict(trt_1)
trt_1['prednoY'] = 1 - trt_1['predY']
trt_1['predY_cumprod'] = trt_1.groupby('id')['prednoY'].cumprod()
trt_1['riskY'] = 1 - trt_1['predY_cumprod']
print(np.unique(trt_1['riskY']))
bs_iptw_risk1 = np.unique(trt_1['riskY'])
# [0.0104     0.0209     0.03769292 0.04891264 0.05952422 0.07805828]


# Outcome model just among clones 0
m_out = smf.glm(formula='Y ~ C(t_out) ', family=f, data=d_plan_long.loc[d_plan_long['trt_clone'] == 0], freq_weights=d_plan_long.loc[d_plan_long['trt_clone'] == 0,'iptw_clones'])
logit_out = m_out.fit()
print(logit_out.params)
# Intercept       -4.606017
# C(t_out)[T.2]    0.058554
# C(t_out)[T.3]    0.010114
# C(t_out)[T.4]    0.130180
# C(t_out)[T.5]    0.221152
# C(t_out)[T.6]    0.611820
# dtype: float64
trt_0 = d_plan_long.loc[d_plan_long['trt_clone'] == 0].copy()
trt_0['predY'] = logit_out.predict(trt_0)
trt_0['prednoY'] = 1 - trt_0['predY']
trt_0['predY_cumprod'] = trt_0.groupby('id')['prednoY'].cumprod()
trt_0['riskY'] = 1 - trt_0['predY_cumprod']
print(np.unique(trt_0['riskY']))
bs_iptw_risk0 = np.unique(trt_0['riskY'])
# [0.00989269 0.02027198 0.03006167 0.04097602 0.05278269 0.06991689]

##################################################
# BOOTSTRAP FOR CONFIDENCE INTERVALS 
##################################################
def ccw_iptw_weights(indat):
    d_iptw = indat.copy()
    m_iptw_cut2 = smf.glm(formula='A ~  A_lag + W + C(t_in)', data=d_iptw.loc[d_iptw['t_in'] <= gp], family=f)
    logit_iptw_cut2 = m_iptw_cut2.fit()# Version where A_lag is always 0 for t_in=0
    d_iptw['pred_trt'] = logit_iptw_cut2.predict(d_iptw.loc[d_iptw['t_in'] <= gp])
    d_iptw['pred_trt'] = d_iptw['pred_trt'].fillna(0)
    d_iptw['pred_notrt']  = 1 - d_iptw['pred_trt']
    d_iptw['pred_observedtrt'] = np.where((d_iptw['A'] == 1) & (d_iptw['t_in'] <= gp), d_iptw['pred_trt'], d_iptw['pred_notrt'] )
    d_iptw['pred_trt_denom'] = d_iptw.groupby('id')['pred_observedtrt'].cumprod()
    d_iptw['iptw_wgt'] = 1 / d_iptw['pred_trt_denom']
    return d_iptw[['id', 't_in', 'pred_trt', 'pred_observedtrt', 'pred_trt_denom']]

def ccw_iptw_plan1(indat, weights):
    # Merging in and setting the weights
    d_cens1_i = indat
    d_cens1_i['trt_clone'] = 1
    d_cens1_i = pd.merge(left=d_cens1_i, right=weights[['id', 't_in', 'pred_observedtrt', 'pred_trt_denom']])
    # Adjusting the weights for each one 
    d_cens1_i['pred_trt_plan1'] = np.where((d_cens1_i['t_in'] == gp), d_cens1_i['pred_trt_denom'], 1)
    d_cens1_i['pr_trt1_denom'] = d_cens1_i.groupby('id')['pred_trt_plan1'].cumprod()
    d_cens1_i['iptw_plan1_set1'] = np.where((d_cens1_i['plan_100'] == 1) | (d_cens1_i['plan_010'] == 1), 1, 1/d_cens1_i['pr_trt1_denom'])

    # Modeling the outcome 
    d1 = d_cens1_i.loc[(d_cens1_i["censor"] != 1), :].copy()
    cens1_mod = smf.glm(formula='Y ~ C(t_out)', family=f, data=d1, freq_weights=d1['iptw_plan1_set1'])
    cens1_fit = cens1_mod.fit()
    d1['predY'] = cens1_fit.predict(d1)
    d1['prednoY'] = 1 - d1['predY']
    d1['predY_cumprod'] = d1.groupby('id')['prednoY'].cumprod()
    d1['riskY'] = 1 - d1['predY_cumprod']
    iptw_risk1 = np.unique(d1['riskY'])
    return np.asarray(iptw_risk1)

def ccw_iptw_plan0(indat, weights):
    # Merging in and setting the weights
    d_cens0_i = indat
    d_cens0_i['trt_clone'] = 0
    d_cens0_i = pd.merge(left = d_cens0_i, right = weights[['id', 't_in', 'pred_trt', 'pred_observedtrt','pred_trt_denom']], how='left', left_on=['id', 't_in'], right_on=['id', 't_in'])
    d_cens0_i['iptw_plan0'] = 1/d_cens0_i['pred_trt_denom']
    # Modeling the outcome
    d0 = d_cens0_i.loc[(d_cens0_i["censor"] != 1), :].copy()
    cens0_mod = smf.glm(formula='Y ~ C(t_out)', family=f, data=d0, freq_weights=d0['iptw_plan0'])
    cens0_fit = cens0_mod.fit()
    d0['predY'] = cens0_fit.predict(d0)
    d0['prednoY'] = 1 - d0['predY']
    d0['predY_cumprod'] = d0.groupby('id')['prednoY'].cumprod()
    d0['riskY'] = 1 - d0['predY_cumprod']
    iptw_risk0 = np.unique(d0['riskY'])
    return np.asarray(iptw_risk0)

# Verifying the bootstrap does work in a sense
iptw_weight_dat = ccw_iptw_weights(indat=d_obs)
risk1 = ccw_iptw_plan1(indat=d_cens1, weights=iptw_weight_dat)
# array([0.115     , 0.13      , 0.17970892, 0.17970892, 0.18355848, 0.18355848])
risk0 = ccw_iptw_plan0(indat=d_cens0, weights=iptw_weight_dat)
# array([0.085488  , 0.09228369, 0.09997561, 0.11426203, 0.12195395, 0.13514295])


# Boot strap prep
seed = 1982283
print(seed)
n = d_obs["id"].nunique()
rng = np.random.default_rng(seed)
ids = d_obs['id'].unique()


# B=100_000
B = 10_000
print("Bootstrap:", B)
# B=5
iters1 = []
iters0 = []
t0_boot = time.perf_counter()
for i in range(B):
    if (i != 0) & (i % 10 == 0):
        print(i)
    id_samp = rng.choice(ids, size=len(ids), replace=True)

    # For the weights
    groupsobs = d_obs.groupby("id").indices
    row_indexobs = np.hstack([groupsobs[orig] for orig in id_samp])
    d_sampobs = d_obs.iloc[row_indexobs].reset_index(drop=True)
    sizesobs_perid = np.array([len(groupsobs[orig]) for orig in id_samp])
    d_sampobs["id"] = np.repeat(np.arange(len(id_samp)), sizesobs_perid) 
    d_weights = ccw_iptw_weights(d_sampobs)

    # For the clones to plan 1
    groups1 = d_cens1.groupby("id").indices  # creates a dictionary of for each id, the row index that belongs to the id
    row_index1 = np.hstack([groups1[orig] for orig in id_samp])
    d_samp1 = d_cens1.iloc[row_index1].reset_index(drop=True)
    sizes1_perid = np.array([len(groups1[orig]) for orig in id_samp])
    d_samp1["id"] = np.repeat(np.arange(len(id_samp)), sizes1_perid)
    # parts1 = [da_long[da_long['id']==orig].assign(id=new) for new, orig in enumerate(id_samp)]
    # d_samp1 = pd.concat(parts1, ignore_index=True)
    param_out1 = ccw_iptw_plan1(d_samp1, weights=d_weights)
    iters1.append(param_out1)

    # For the clones for plan 0
    groups0 = d_cens0.groupby("id").indices  # creates a dictionary of for each id, the row index that belongs to the id
    row_index0 = np.hstack([groups0[orig] for orig in id_samp])
    d_samp0 = d_cens0.iloc[row_index0].reset_index(drop=True)
    sizes0_perid = np.array([len(groups0[orig]) for orig in id_samp])
    d_samp0["id"] = np.repeat(np.arange(len(id_samp)), sizes0_perid)
    param_out0 = ccw_iptw_plan0(d_samp0, weights=d_weights)
    iters0.append(param_out0)
    # Storing interim iters every 10000 iters
    if (i != 0) & (i % 10000 == 0):
        print(f"{i} modulo 100 is 0!")
        pd.DataFrame(iters0).to_csv(f"./iptw_iters1_{i}of{B}.csv", index=False)
        pd.DataFrame(iters1).to_csv(f"./iptw_iters0_{i}of{B}.csv", index=False)

    # # Storing interim iters every 100 iters
    # if (i != 0) & (i % 100 == 0):
    #     print(f"{i} modulo 100 is 0!")
    #     pd.DataFrame(iters0).to_csv(f"./iptw_iters1_{i}of{B}.csv", index=False)
    #     pd.DataFrame(iters1).to_csv(f"./iptw_iters0_{i}of{B}.csv", index=False)

t1_boot = time.perf_counter()

with open(f"./time_{B}bootstrap_{date}.txt", "w", encoding="utf-8") as file:
    print(f"For size = {nobs} study pop, Elapsed for B={B} iters: {((t1_boot - t0_boot)/60):.3f} mins or  OR {(t1_ee - t0_ee):.3f} secs", file=file)

# Elapsed for B=1000 iters: 10.055 mins

stacked_iters1 = np.vstack(iters1)
stacked_iters0 = np.vstack(iters0)
np.mean(stacked_iters0, axis = 0)
# array([0.10901008, 0.11668609, 0.12775744, 0.13670338, 0.15439118,
#        0.16784082])

np.mean(stacked_iters1, axis = 0)

pd.DataFrame(stacked_iters1).to_csv(f"./stacked_iptw_{B}iters1.csv", index=False)
pd.DataFrame(stacked_iters0).to_csv(f"./stacked_iptw_{B}iters0.csv", index=False)

np.var(stacked_iters1, axis=0, ddof=1)
np.sqrt(np.var(stacked_iters1, axis=0, ddof=1))


out1 = pd.DataFrame({"estimate": np.asarray(bs_iptw_risk1)})
out1["variance"] = np.var(stacked_iters1, axis=0, ddof=1)
out1["stderr"] = np.sqrt(out1["variance"])
out1["LCL"] = out1["estimate"] - 1.96 * out1["stderr"]
out1["UCL"] = out1["estimate"] + 1.96 * out1["stderr"]
out1["measure"] = "risk1"

print(out1)

# RISK 1 BOOTSTRAP 


print(ee_r1_out)
# RISK 1 ESTIMATING EQUATIONS


np.var(stacked_iters1, axis=0, ddof=1)
np.sqrt(np.var(stacked_iters1, axis=0, ddof=1))
np.var(stacked_iters0, axis=0, ddof=1)
np.sqrt(np.var(stacked_iters0, axis=0, ddof=1))

out0 = pd.DataFrame({"estimate": np.asarray(bs_iptw_risk0)})
out0["variance"] = np.var(stacked_iters0, axis=0, ddof=1)
out0["stderr"] = np.sqrt(out0["variance"])
out0["LCL"] = out0["estimate"] - 1.96 * out0["stderr"]
out0["UCL"] = out0["estimate"] + 1.96 * out0["stderr"]
out0["measure"] = "risk0"

# print(out0)
# RISK 0 BOOTSTRAP

# print(ee_r0_out)
# RISK 0 ESTIMATING EQUATIONS

np.var(stacked_iters1 - stacked_iters0, axis=0, ddof=1)
np.sqrt(np.var(stacked_iters1 - stacked_iters0, axis=0, ddof=1))


outrd = pd.DataFrame({"estimate": np.asarray(bs_iptw_risk1 - bs_iptw_risk0)})
outrd["variance"] = np.var(stacked_iters1 - stacked_iters0, axis=0, ddof=1)
outrd["stderr"] = np.sqrt(outrd["variance"])
outrd["LCL"] = outrd["estimate"] - 1.96 * outrd["stderr"]
outrd["UCL"] = outrd["estimate"] + 1.96 * outrd["stderr"]
outrd["measure"] = "rd"

# print(outrd)
# RISK DIFFERENCE BOOTSTRAP

# print(ee_rd_out)
out_eeiptw_Rcompare = pd.concat([ee_rd_out, ee_r1_out, ee_r0_out], axis=0)
out_eeiptw_Rcompare.to_csv('out_eeiptw_Rcompare.csv', float_format="%.17g")
# compared to estimating equations
out_boot = pd.concat([out0, out1, outrd], axis=0)
out_boot['implementation'] = f"{B} bootstrap"
out_boot['time_secs'] = (t1_boot - t0_boot)
out_eeiptw = pd.concat([ee_r0_out, ee_r1_out, ee_rd_out], axis=0)
out_eeiptw['implementation'] = "ee"
out_eeiptw['time_secs'] = (t1_ee - t0_ee)
out_csv = pd.concat([ out_boot ,out_eeiptw], axis=0)

out_csv.to_csv(f'time_{B}NPBSvsEE_{date}_{nobs}.csv')

# Results with Bootstrap 100,000 Iterations-- expect Std Err is similar to the 4th decimal point



# # Results with Bootstrap 10,000 Iterations-- StdErr is similar to the 3rd decimal point
#  	t_out-1	estimate 	variance 	    stderr 	    LCL 	    UCL 	    measure  implementation 	time_secs
# 0 	0 	0.009893 	1.108293e-06 	0.001053 	0.007829 	0.011956 	risk0 	10000 bootstrap 	6639.645044
# 1 	1 	0.020272 	2.350866e-06 	0.001533 	0.017267 	0.023277 	risk0 	10000 bootstrap 	6639.645044
# 2 	2 	0.030062 	3.725678e-06 	0.001930 	0.026278 	0.033845 	risk0 	10000 bootstrap 	6639.645044
# 3 	3 	0.040976 	5.252466e-06 	0.002292 	0.036484 	0.045468 	risk0 	10000 bootstrap 	6639.645044
# 4 	4 	0.052783 	6.781399e-06 	0.002604 	0.047679 	0.057887 	risk0 	10000 bootstrap 	6639.645044
# 5 	5 	0.069917 	9.013581e-06 	0.003002 	0.064032 	0.075801 	risk0 	10000 bootstrap 	6639.645044
# 6 	0 	0.010400 	1.036630e-06 	0.001018 	0.008404 	0.012396 	risk1 	10000 bootstrap 	6639.645044
# 7 	1 	0.020900 	2.044986e-06 	0.001430 	0.018097 	0.023703 	risk1 	10000 bootstrap 	6639.645044
# 8 	2 	0.037693 	1.676500e-05 	0.004095 	0.029668 	0.045718 	risk1 	10000 bootstrap 	6639.645044
# 9 	3 	0.048913 	2.602180e-05 	0.005101 	0.038914 	0.058911 	risk1 	10000 bootstrap 	6639.645044
# 10 	4 	0.059524 	3.335632e-05 	0.005775 	0.048204 	0.070844 	risk1 	10000 bootstrap 	6639.645044
# 11 	5 	0.078058 	4.753149e-05 	0.006894 	0.064545 	0.091571 	risk1 	10000 bootstrap 	6639.645044
# 12 	0 	0.000507 	1.569433e-07 	0.000396 	-0.000269 	0.001284 	rd 	    10000 bootstrap 	6639.645044
# 13 	1 	0.000628 	4.257657e-07 	0.000653 	-0.000651 	0.001907 	rd 	    10000 bootstrap 	6639.645044
# 14 	2 	0.007631 	1.666868e-05 	0.004083 	-0.000371 	0.015633 	rd 	    10000 bootstrap 	6639.645044
# 15 	3 	0.007937 	2.741881e-05 	0.005236 	-0.002327 	0.018200 	rd 	    10000 bootstrap 	6639.645044
# 16 	4 	0.006742 	3.635557e-05 	0.006030 	-0.005076 	0.018559 	rd 	    10000 bootstrap 	6639.645044
# 17 	5 	0.008141 	5.330201e-05 	0.007301 	-0.006168 	0.022451 	rd 	    10000 bootstrap 	6639.645044
# 18 	0 	0.009893 	1.089592e-06 	0.001044 	0.007847 	0.011939 	risk0 	        ee 	        1.742570
# 19 	1 	0.020272 	2.355561e-06 	0.001535 	0.017264 	0.023280 	risk0 	        ee 	        1.742570
# 20 	2 	0.030062 	3.693567e-06 	0.001922 	0.026295 	0.033828 	risk0 	        ee 	        1.742570
# 21 	3 	0.040976 	5.171314e-06 	0.002274 	0.036519 	0.045433 	risk0 	        ee 	        1.742570
# 22 	4 	0.052783 	6.695651e-06 	0.002588 	0.047711 	0.057854 	risk0 	        ee 	        1.742570
# 23 	5 	0.069917 	8.854544e-06 	0.002976 	0.064085 	0.075749 	risk0 	        ee 	        1.742570
# 24 	0 	0.010400 	1.029190e-06 	0.001014 	0.008412 	0.012388 	risk1 	        ee 	        1.742570
# 25 	1 	0.020900 	2.046314e-06 	0.001430 	0.018096 	0.023704 	risk1 	        ee 	        1.742570
# 26 	2 	0.037693 	1.672415e-05 	0.004090 	0.029678 	0.045708 	risk1 	        ee 	        1.742570
# 27 	3 	0.048913 	2.524546e-05 	0.005024 	0.039065 	0.058760 	risk1 	        ee 	        1.742570
# 28 	4 	0.059524 	3.223866e-05 	0.005678 	0.048396 	0.070653 	risk1 	        ee 	        1.742570
# 29 	5 	0.078058 	4.619130e-05 	0.006796 	0.064738 	0.091379 	risk1 	        ee 	        1.742570
# 30 	0 	0.000507 	1.598513e-07 	0.000400 	-0.000276 	0.001291 	rd 	            ee 	        1.742570
# 31 	1 	0.000628 	4.306998e-07 	0.000656 	-0.000658 	0.001914 	rd 	            ee 	        1.742570
# 32 	2 	0.007631 	1.654384e-05 	0.004067 	-0.000341 	0.015603 	rd 	            ee 	        1.742570
# 33 	3 	0.007937 	2.662455e-05 	0.005160 	-0.002177 	0.018050 	rd 	            ee 	        1.742570
# 34 	4 	0.006742 	3.522707e-05 	0.005935 	-0.004891 	0.018374 	rd 	            ee 	        1.742570
# 35 	5 	0.008141 	5.143342e-05 	0.007172 	-0.005915 	0.022198 	rd 	            ee 	        1.742570


# !SECTION - 