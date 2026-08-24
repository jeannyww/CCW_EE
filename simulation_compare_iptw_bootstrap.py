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
from delicatessen import MEstimator
from dgp import dgp_grace

# from deli_estimation import MEstimator
from functions_ee import clone_and_censor

# Establish the family distribution (binomial) and link function (logit)
f = sm.families.family.Binomial(link=sm.genmod.families.links.Logit())
##################################################
# SECTION - DATA 
##################################################
date = "2026-08-12"

rng_runs = np.random.default_rng(319_031_111)
gp=2
nobs=3000
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

with open(f"./time_ee_{date}.txt", "w", encoding="utf-8") as file:
    print(f"For size = {nobs} study pop, Elapsed for estimating equations B=1 iters: {((t1_ee - t0_ee)/60):.3f} mins OR {(t1_ee - t0_ee):.3f} secs", file=file)
estr_theta = ccw_estr.theta
# Now take the rd outputs

# Output 
point_rd = estr_theta[:n_times_to_predict]
point_r1 = estr_theta[n_times_to_predict:(n_times_to_predict*2)]
point_r0 = estr_theta[(n_times_to_predict*2):(n_times_to_predict*3)]
variance_rd = (np.diag(ccw_estr.variance))[:n_times_to_predict]
# array([0.00017243, 0.00021711, 0.00258787, 0.00268472, 0.00274975,
    #    0.00282905])
print(np.sqrt(variance_rd))
# [0.01313138 0.01473475 0.05087111 0.05181431 0.0524381  0.05318879]
variance_r1 = (np.diag(ccw_estr.variance))[n_times_to_predict:(n_times_to_predict*2)]
print(np.sqrt(variance_r1))
#[0.02255827 0.02378024 0.05322323 0.05322323 0.05329367 0.05329367]
variance_r0 = (np.diag(ccw_estr.variance))[(n_times_to_predict*2):(n_times_to_predict*3)]
print(np.sqrt(variance_r0))
# [0.02110527 0.02200509 0.02310635 0.02487155 0.02579815 0.02709598]
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

# Risk 1 
ee_r1_out = pd.DataFrame({"estimate": np.asarray(point_r1)})
ee_r1_out['variance'] = np.asarray(variance_r1)
ee_r1_out['stderr'] = np.asarray(np.sqrt(variance_r1))
ee_r1_out['LCL'] = np.asarray(lcl_r1)
ee_r1_out['UCL'] = np.asarray(ucl_r1)
ee_r1_out['measure'] = "risk1"
print(ee_r1_out)

# Risk 0
ee_r0_out = pd.DataFrame({"estimate": np.asarray(point_r0)})
ee_r0_out['variance'] = np.asarray(variance_r0)
ee_r0_out['stderr'] = np.asarray(np.sqrt(variance_r0))
ee_r0_out['LCL'] = np.asarray(lcl_r0)
ee_r0_out['UCL'] = np.asarray(ucl_r0)
ee_r0_out['measure'] = "risk0"
print(ee_r0_out)

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

# Version where A_lag is always 0 for t_in=0
d_iptw = d_obs.copy()
d_iptw['pred_trt'] = logit_iptw_cut2.predict(d_iptw.loc[d_iptw['t_in'] <= gp])
d_iptw['pred_trt'] = d_iptw['pred_trt'].fillna(0)
d_iptw['pred_notrt']  = 1 - d_iptw['pred_trt']
d_iptw['pred_observedtrt'] = np.where((d_iptw['A'] == 1) & (d_iptw['t_in'] <= gp), d_iptw['pred_trt'], d_iptw['pred_notrt'] )
d_iptw['pred_trt_denom'] = d_iptw.groupby('id')['pred_observedtrt'].cumprod()
d_iptw['iptw_wgt'] = 1 / d_iptw['pred_trt_denom']

d_cens1_i = d_cens1
d_cens1_i['trt_clone'] = 1
d_cens1_i = pd.merge(left=d_cens1_i, right=d_iptw[['id', 't_in', 'pred_observedtrt', 'pred_trt_denom']])
d_cens1_i['censor_1'] = d_cens1_i['censor']
# Adjusting the weights for each one 
d_cens1_i['pred_trt_plan1'] = np.where((d_cens1_i['t_in'] == gp), d_cens1_i['pred_trt_denom'], 1)
d_cens1_i['pr_trt1_denom'] = d_cens1_i.groupby('id')['pred_trt_plan1'].cumprod()
d_cens1_i['iptw_plan1_set1'] = np.where((d_cens1_i['plan_100'] == 1) | (d_cens1_i['plan_010'] == 1), 1, 1/d_cens1_i['pr_trt1_denom'])
d_cens1_i['iptw_clones'] = d_cens1_i['iptw_plan1_set1'] 

d_cens0_i = d_cens0
d_cens0_i['trt_clone'] = 0
d_cens0_i = pd.merge(left = d_cens0_i, right = d_iptw[['id', 't_in', 'pred_trt', 'pred_observedtrt','pred_trt_denom']], how='left', left_on=['id', 't_in'], right_on=['id', 't_in'])
d_cens0_i['censor_0'] = d_cens0_i['censor']
d_cens0_i['iptw_plan0'] = 1/d_cens0_i['pred_trt_denom']
d_cens0_i['iptw_clones'] = d_cens0_i['iptw_plan0']


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
print(logit_out.summary())
#                  Generalized Linear Model Regression Results                  
# ==============================================================================
# Dep. Variable:                      Y   No. Observations:                  569
# Model:                            GLM   Df Residuals:                  1235.26
# Model Family:                Binomial   Df Model:                            5
# Link Function:                  Logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -142.43
# Date:                Sat, 08 Aug 2026   Deviance:                       284.86
# Time:                        14:47:38   Pearson chi2:                     816.
# No. Iterations:                    25   Pseudo R-squ. (CS):             0.1127
# Covariance Type:            nonrobust                                         
# =================================================================================
#                     coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# Intercept        -2.0407      0.222     -9.207      0.000      -2.475      -1.606
# C(t_out)[T.2]    -2.0198      0.623     -3.242      0.001      -3.241      -0.799
# C(t_out)[T.3]    -0.7628      0.362     -2.105      0.035      -1.473      -0.053
# C(t_out)[T.4]   -24.5254   2.44e+04     -0.001      0.999   -4.78e+04    4.78e+04
# C(t_out)[T.5]    -3.3163      1.027     -3.231      0.001      -5.328      -1.304
# C(t_out)[T.6]   -24.5254   2.45e+04     -0.001      0.999    -4.8e+04    4.79e+04
# =================================================================================
trt_1 = d_plan_long.loc[d_plan_long['trt_clone'] == 1].copy()
trt_1['predY'] = logit_out.predict(trt_1)
trt_1['prednoY'] = 1 - trt_1['predY']
trt_1['predY_cumprod'] = trt_1.groupby('id')['prednoY'].cumprod()
trt_1['riskY'] = 1 - trt_1['predY_cumprod']
print(np.unique(trt_1['riskY']))
bs_iptw_risk1 = np.unique(trt_1['riskY'])
# [0.115      0.13       0.17970892 0.17970892 0.18355848 0.18355848]

# Outcome model just among clones 0
m_out = smf.glm(formula='Y ~ C(t_out) ', family=f, data=d_plan_long.loc[d_plan_long['trt_clone'] == 0], freq_weights=d_plan_long.loc[d_plan_long['trt_clone'] == 0,'iptw_clones'])
logit_out = m_out.fit()
print(logit_out.summary())
#                  Generalized Linear Model Regression Results                  
# ==============================================================================
# Dep. Variable:                      Y   No. Observations:                  808
# Model:                            GLM   Df Residuals:                  1081.63
# Model Family:                Binomial   Df Model:                            5
# Link Function:                  Logit   Scale:                          1.0000
# Method:                          IRLS   Log-Likelihood:                -111.94
# Date:                Sat, 08 Aug 2026   Deviance:                       223.87
# Time:                        14:48:17   Pearson chi2:                 1.09e+03
# No. Iterations:                     8   Pseudo R-squ. (CS):            0.03474
# Covariance Type:            nonrobust                                         
# =================================================================================
#                     coef    std err          z      P>|z|      [0.025      0.975]
# ---------------------------------------------------------------------------------
# Intercept        -2.3700      0.253     -9.379      0.000      -2.865      -1.875
# C(t_out)[T.2]    -2.5246      0.898     -2.812      0.005      -4.284      -0.765
# C(t_out)[T.3]    -2.3922      0.854     -2.803      0.005      -4.065      -0.719
# C(t_out)[T.4]    -1.7571      0.651     -2.697      0.007      -3.034      -0.480
# C(t_out)[T.5]    -2.3675      0.854     -2.774      0.006      -4.041      -0.694
# C(t_out)[T.6]    -1.8132      0.674     -2.691      0.007      -3.134      -0.492
# =================================================================================
trt_0 = d_plan_long.loc[d_plan_long['trt_clone'] == 0].copy()
trt_0['predY'] = logit_out.predict(trt_0)
trt_0['prednoY'] = 1 - trt_0['predY']
trt_0['predY_cumprod'] = trt_0.groupby('id')['prednoY'].cumprod()
trt_0['riskY'] = 1 - trt_0['predY_cumprod']
print(np.unique(trt_0['riskY']))
bs_iptw_risk0 = np.unique(trt_0['riskY'])
# [0.085488   0.09228369 0.09997561 0.11426203 0.12195395 0.13514295]

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
B = 1_000
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
# array([0.1193435 , 0.126775  , 0.14518164, 0.14589186, 0.15561684,
#        0.17689964])

pd.DataFrame(stacked_iters1).to_csv(f"./stacked_iptw_{B}iters1.csv", index=False)
pd.DataFrame(stacked_iters0).to_csv(f"./stacked_iptw_{B}iters0.csv", index=False)
# pd.DataFrame(stacked_itersRD).to_csv("./stacked_itersRD.csv", index=False)

np.var(stacked_iters1, axis=0, ddof=1)
np.sqrt(np.var(stacked_iters1, axis=0, ddof=1))
# array([0.00697552, 0.00715406, 0.01056829, 0.01055145, 0.01217187,
#        0.01494049])

out1 = pd.DataFrame({"estimate": np.asarray(bs_iptw_risk1)})
out1["variance"] = np.var(stacked_iters1, axis=0, ddof=1)
out1["stderr"] = np.sqrt(out1["variance"])
out1["LCL"] = out1["estimate"] - 1.96 * out1["stderr"]
out1["UCL"] = out1["estimate"] + 1.96 * out1["stderr"]
out1["measure"] = "risk1"

print(out1)

# RISK 1 BOOTSTRAP 
#    estimate  variance       std       LCL       UCL
# 0  0.119004  0.000049  0.006976  0.105332  0.132676
# 1  0.126504  0.000051  0.007154  0.112482  0.140526
# 2  0.144871  0.000112  0.010568  0.124157  0.165584
# 3  0.145588  0.000111  0.010551  0.124907  0.166269
# 4  0.155499  0.000148  0.012172  0.131642  0.179356
# 5  0.177018  0.000223  0.014940  0.147735  0.206301
print(ee_r1_out)
# RISK 1 ESTIMATING EQUATIONS
#    estimate  variance    stderr       LCL       UCL
# 0  0.119000  0.000052  0.007240  0.104810  0.133190
# 1  0.126500  0.000055  0.007433  0.111932  0.141068
# 2  0.144868  0.000126  0.011245  0.122828  0.166907
# 3  0.145585  0.000127  0.011252  0.123533  0.167638
# 4  0.155496  0.000162  0.012747  0.130513  0.180479
# 5  0.177015  0.000243  0.015599  0.146441  0.207590

# Compared to eestimating equations
# array([0.119     , 0.1265    , 0.14486753, 0.14558512, 0.15549596,
# 0.17701546]), 
# array([0.00724013, 0.00743296, 0.01124461, 0.0112515 , 0.01274656,
# 0.01559935]), 

np.var(stacked_iters1, axis=0, ddof=1)
np.sqrt(np.var(stacked_iters1, axis=0, ddof=1))
# array([0.00697552, 0.00715406, 0.01056829, 0.01055145, 0.01217187,
#        0.01494049])
np.var(stacked_iters0, axis=0, ddof=1)
np.sqrt(np.var(stacked_iters0, axis=0, ddof=1))
# array([0.00708384, 0.00733122, 0.00759468, 0.00799304, 0.00861501,
    #    0.00915675])

out0 = pd.DataFrame({"estimate": np.asarray(bs_iptw_risk0)})
out0["variance"] = np.var(stacked_iters0, axis=0, ddof=1)
out0["stderr"] = np.sqrt(out0["variance"])
out0["LCL"] = out0["estimate"] - 1.96 * out0["stderr"]
out0["UCL"] = out0["estimate"] + 1.96 * out0["stderr"]
out0["measure"] = "risk0"

# print(out0)
# RISK 0 BOOTSTRAP
#    estimate  variance       std       LCL       UCL
# 0  0.108612  0.000050  0.007084  0.094727  0.122496
# 1  0.116394  0.000054  0.007331  0.102025  0.130763
# 2  0.127445  0.000058  0.007595  0.112559  0.142330
# 3  0.136298  0.000064  0.007993  0.120632  0.151965
# 4  0.154006  0.000074  0.008615  0.137121  0.170892
# 5  0.167220  0.000084  0.009157  0.149273  0.185168
# print(ee_r0_out)
# RISK 0 ESTIMATING EQUATIONS
#    estimate  variance    stderr       LCL       UCL
# 0  0.108612  0.000056  0.007493  0.093927  0.123297
# 1  0.116394  0.000060  0.007757  0.101190  0.131599
# 2  0.127445  0.000067  0.008171  0.111430  0.143461
# 3  0.136299  0.000072  0.008480  0.119678  0.152920
# 4  0.154007  0.000082  0.009046  0.136277  0.171738
# 5  0.167222  0.000089  0.009427  0.148745  0.185698

# Compared to estimating equations
# array([0.10861206, 0.11639432, 0.12744517, 0.13629923, 0.15400734,
# 0.16722157]), 
# array([0.00749262, 0.00775748, 0.00817132, 0.00848035, 0.0090463 ,
# 0.00942692])

np.var(stacked_iters1 - stacked_iters0, axis=0, ddof=1)
np.sqrt(np.var(stacked_iters1 - stacked_iters0, axis=0, ddof=1))
# array([0.00343632, 0.00354108, 0.0091942 , 0.00958359, 0.01187489,
    #    0.01493929])


outrd = pd.DataFrame({"estimate": np.asarray(bs_iptw_risk1 - bs_iptw_risk0)})
outrd["variance"] = np.var(stacked_iters1 - stacked_iters0, axis=0, ddof=1)
outrd["stderr"] = np.sqrt(outrd["variance"])
outrd["LCL"] = outrd["estimate"] - 1.96 * outrd["stderr"]
outrd["UCL"] = outrd["estimate"] + 1.96 * outrd["stderr"]
outrd["measure"] = "rd"

# print(outrd)
# RISK DIFFERENCE BOOTSTRAP
#    estimate  variance       std       LCL       UCL
# 0  0.010392  0.000012  0.003436  0.003657  0.017127
# 1  0.010110  0.000013  0.003541  0.003169  0.017050
# 2  0.017426  0.000085  0.009194 -0.000595  0.035446
# 3  0.009290  0.000092  0.009584 -0.009494  0.028073
# 4  0.001493  0.000141  0.011875 -0.021782  0.024768
# 5  0.009798  0.000223  0.014939 -0.019483  0.039079
# print(ee_rd_out)
# RISK DIFFERENCE ESTIMATING EQUATIONS
#    estimate  variance    stderr       LCL       UCL
# 0  0.010388  0.000012  0.003430  0.003665  0.017111
# 1  0.010106  0.000013  0.003554  0.003139  0.017072
# 2  0.017422  0.000094  0.009692 -0.001574  0.036418
# 3  0.009286  0.000100  0.010012 -0.010337  0.028909
# 4  0.001489  0.000149  0.012212 -0.022446  0.025423
# 5  0.009794  0.000241  0.015526 -0.020637  0.040224

# compared to estimating equations
# array([0.01038794, 0.01010568, 0.01742236, 0.00928589, 0.00148862,
# 0.0097939 ]), 
# array([0.00343007, 0.0035543 , 0.00969207, 0.01001198, 0.01221188,
# 0.01552608]), 
out_boot = pd.concat([out0, out1, outrd], axis=0)
out_boot['implementation'] = f"{B} bootstrap"
out_boot['time_secs'] = (t1_boot - t0_boot)
out_eeiptw = pd.concat([ee_r0_out, ee_r1_out, ee_rd_out], axis=0)
out_eeiptw['implementation'] = "ee"
out_eeiptw['time_secs'] = (t1_ee - t0_ee)
out_csv = pd.concat([ out_boot ,out_eeiptw], axis=0)

out_csv.to_csv(f'time_{B}NPBSvsEE_{date}_{nobs}.csv')
# !SECTION - 