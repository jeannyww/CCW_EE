#############################################################################
# Simulation to test CCW 
# JHW (2026-01-19)
# Details: simulation runs for IPCW -- need to look into any mistakes in my simulation and ees that result in minute bias and over-estimates confidence interval coverage for the "treat within the grace period" arm 
#############################################################################

##################################################
# Loading Packages
##################################################
import numpy as np
import pandas as pd
from deli_estimation import MEstimator
from dgp import dgp_grace
from functions_ee import ee_plr_censoring, ee_plr_outcome, pred_plr_censor, adjust_ccw_weights, pred_plr_outcome

runs=2_000
nobs=5_000
##################################################
# SECTION - draft functions 
##################################################
def clone_and_censor(data):
    # 1) Clone
    # Plan B (no treatment at all or treatment after the grace period)
    db = data.copy()
    db["plan"] = 0

    da = data.copy()
    da["plan"] = 1

    # 2) Plan B censor when deviate off trt plan Plan B - No A != 1 for any t_in == 0, 1, 2
    db["stop_plan"] = np.where(db["A"] == 1, 1, 0)
    db["n_stopped"] = db.groupby("id")["stop_plan"].cumsum()
    db["n_cumtrt"] = db.groupby("id")["A"].cumsum()
    db["censor"] = np.where(((db["t_in"] <= gp) & (db["n_stopped"] >= 1) & (db["n_cumtrt"] >= 1)), 1, 0)
    db["cum_censor"] = db.groupby("id")["censor"].cumsum()
    db_tcensored = db.loc[(db["censor"] == 1) & (db["cum_censor"] == 1)].copy()
    db_tnocensor = db.loc[db["cum_censor"] == 0].copy()
    db_long = pd.concat([db_tcensored, db_tnocensor], ignore_index=True)
    db_long = db_long.sort_values(["id", "t_in"]).reset_index(drop=True)
    db_long["intercept"] = 1  # Creating intercept for estimating equation implementation
    db_long['t_trt_lagged'] = np.where(db_long['A'] == 1, db_long['t_out'] + 1, 0)
    db_long['t_trt'] = np.where(db_long['A'] == 1, db_long['t_in'], 0)
    # Creating an indexing variable for bootstrap sampling
    db_long["_idx"] = (db_long[["id"]]).apply(tuple)
    db_long = db_long.set_index("_idx")

    # 2) Plan A censor when deviate off of trt Plan A - A ==1 for any t_in == 0, 1, 2
    da["stop_plan"] = np.where(da["A"] == 1, 0, 1)  # if A==1 then you got trt at some t_in
    da["n_stopped"] = da.groupby("id")["stop_plan"].cumsum()
    da["n_cumtrt"] = da.groupby("id")["A"].cumsum()
    da["censor"] = np.where(
        ((da["t_in"] >= gp) & (da["n_stopped"] >= (gp + 1)) & (da["n_cumtrt"] == 0)), 1, 0
    )
    da["id"].nunique()
    da["cum_censor"] = da.groupby("id")["censor"].cumsum()
    da["cum_censor"].value_counts()

    da_tcensored = da.loc[(da["censor"] == 1) & (da["cum_censor"] == 1)].copy()
    # da_tcensored[da_tcensored["id"]==7]
    da_tcensored["id"].nunique()
    # da_tcensored.shape
    # da_tcensored.head()
    # da_tcensored["t_in"].value_counts()  # check censor point for going off of plan A is at gp value
    da_tnocensor = da.loc[da["cum_censor"] == 0].copy()
    da_tnocensor["t_in"].value_counts()
    da_tnocensor["id"].nunique()
    # da_tnocensor.shape
    # dac_tmp = pd.concat([da_tcensored, da_tnocensor], ignore_index=True)
    da_long = pd.concat([da_tcensored, da_tnocensor], ignore_index=True)
    da_long = da_long.sort_values(["id", "t_in"]).reset_index(drop=True)
    da_long["intercept"] = 1
    da_long['t_trt_lagged'] = np.where(da_long['A'] == 1, da_long['t_out'] + 1, 0)
    da_long['t_trt'] = np.where(da_long['A'] == 1, da_long['t_in'], 0)
    da_long["_idx"] = (da_long[["id"]]).apply(tuple)
    da_long = da_long.set_index("_idx")
        
    # Vectorized version of data 
    # Preparing the vectorized data
    db_vec = (
        db_long.sort_values(["id", "t_in"])
        .groupby("id")
        .aggregate(
            plan=("plan", "first"),
            t_in=("t_in", "max"),
            t_out=("t_out", "max"),        
            t_trt=("t_trt", "max"),
            trt_obs=("A", "max"),
            censor=("censor", "last"),
            event=("Y", "last"),
            intercept=("intercept", "first"),
            W=("W", "first"),
        )
        .reset_index()
    )

    # Input arrays for Plan A
    da_vec = (
        da_long.sort_values(["id", "t_in"])
        .groupby("id")
        .aggregate(
            plan=("plan", "first"),
            t_in=("t_in", "max"),
            t_out=("t_out", "max"),
            t_trt=("t_trt", "max"),
            trt_obs=("A", "max"),
            censor=("censor", "last"),
            event=("Y", "last"),
            intercept=("intercept", "first"),
            W=("W", "first"),
        )
        .reset_index()
    )
    da_vec.head()

    d_vec = pd.concat([da_vec, db_vec], axis = 0)
    return d_vec

def psi_censor_plr0(theta): 
    ee_censor0 = ee_plr_censoring(theta=theta, c=c0, t=t_cens0, X=W0, unique_censor_times=censor_times0)
    return ee_censor0
def psi_censor_plr1(theta):
    ee_censor1 = ee_plr_censoring(theta=theta, c=c1, t=t_cens1, X=W1, unique_censor_times=censor_times1)
    return ee_censor1
def psi_ccw_plr(theta):
    # init betas
    init_rds = theta[:n_timepoints]
    init_risks1 = theta[n_timepoints:(n_timepoints*2)]
    init_risks0 = theta[(n_timepoints*2):(n_timepoints*3)]
    init_plr_outcome1 = theta[(n_timepoints*3):(n_timepoints*4)]
    init_plr_outcome0 = theta[(n_timepoints*4):(n_timepoints*5)]
    index_plr = (n_timepoints*5) + W1.shape[1] + n_censor_times1
    init_plr_censor1 = theta[(n_timepoints*5):index_plr]
    init_plr_censor0 = theta[index_plr:]

    # Estimating functions for plan a=1
    # Model
    ee_censor1 = ee_plr_censoring(theta=init_plr_censor1, c=c1, t=t_cens1, X=W1, unique_censor_times=censor_times1)
    # Predict 
    censoring_weights1 = pred_plr_censor(theta=init_plr_censor1, t=t_cens1, c=c1, X=W1,unique_censor_times=censor_times1, times_to_predict=tot_timepoints)
    actual_weights1 = adjust_ccw_weights(plan=1, weights=censoring_weights1, trt=trt1, trt_times=t_trt1, gp=gp)
    # Model
    ee_outcome1 = ee_plr_outcome(theta=init_plr_outcome1, t=t_delta1, delta=delta1, t_cens=t_cens1, censor=c1, weights=actual_weights1)
    # Predict 
    pred_risk1 = pred_plr_outcome(theta=init_plr_outcome1, t=t_delta1, delta=delta1, times_to_predict=tot_timepoints, unique_times=unique_delta_times1)
    # Nonparametric transformation
    ee_risk1 = pred_risk1 - np.asarray(init_risks1)[:, None]

    # Estimating functions for plan a=0
    ee_censor0 = ee_plr_censoring(theta=init_plr_censor0, c=c0, t=t_cens0, X=W0, unique_censor_times=censor_times0)
    censoring_weights0 = pred_plr_censor(theta=init_plr_censor0, t=t_cens0, c=c0, X=W0,unique_censor_times=censor_times0, times_to_predict=tot_timepoints)
    ee_outcome0 = ee_plr_outcome(theta=init_plr_outcome0, t=t_delta0, delta=delta0, t_cens=t_cens0, censor=c0, weights=censoring_weights0)
    pred_risk0 = pred_plr_outcome(theta=init_plr_outcome0, t=t_delta0, delta=delta0, times_to_predict=tot_timepoints, unique_times=unique_delta_times0)
    ee_risk0 = pred_risk0 - np.asarray(init_risks0)[:, None]

    # ee_rd_outcome = ee_rd(theta=init_rds, risk1=pred_risk1, risk0=pred_risk0)
    # V2
    # ee_rd_outcome = (pred_risk1 - pred_risk0) - np.asarray(init_rds)[:, None] #v2 output,
    ee_rd_outcome = (pred_risk1 - pred_risk0) - np.asarray(init_rds)[:, None] #v5 output,

    # ee_rd_outcome = np.ones((tot_timepoints.shape[0], ee_risk1.shape[1]))*((np.asarray(init_risks1)[:, None] - np.asarray(init_risks0)[:, None]) - np.asarray(init_rds)[:, None]) # v3, and v4 

    # ee_rd_outcome = (np.zeros((tot_timepoints.shape[0], ee_risk1.shape[1])) + (np.asarray(init_risks1)[:, None] - np.asarray(init_risks0)[:, None])) - np.asarray(init_rds)[:, None] 

    ee_stack = np.vstack([ee_rd_outcome, ee_risk1, ee_risk0, ee_outcome1, ee_outcome0, ee_censor1, ee_censor0])
    return ee_stack

# !SECTION - kkkk 
##################################################
# SECTION - Runs
##################################################
vers = 0
# risk1 = pd.read_csv("true_risk1.csv")
# risk1 = pd.read_csv("true_risk1_v1.csv")
# r11, r12, r13, r14, r15, r16 = np.asarray(risk1['Surv prob'])
# 
# risk0 = pd.read_csv("true_risk0.csv")
# r01, r02, r03, r04, r05, r06 = np.asarray(risk0['Surv prob'])
# 
# 
# truth = risk1 - risk0 
# truth.to_csv("true_rd.csv")

risk1 = pd.read_csv("true2_risk1_method0.csv")
r11, r12, r13, r14, r15, r16 = np.asarray(risk1['Surv prob'])

risk0 = pd.read_csv("true2_risk0.csv")
r01, r02, r03, r04, r05, r06 = np.asarray(risk0['Surv prob'])

truth = risk1 - risk0

#############################################
# Starting the runs
#############################################

t1, t2, t3, t4, t5, t6 = np.asarray(truth["Surv prob"])
r1, r2, r3, r4, r5, r6 = [], [], [], [], [], []

# rng_runs = np.random.default_rng(209_309)
rng_runs = np.random.default_rng(319_031_111)
gp=2
# runs=1_000
# runs=5
# runs=5_000
# runs=10_000


for i in range(runs):
    print(f"run: {i+1}")
    r1_row, r2_row, r3_row, r4_row, r5_row, r6_row = [], [], [], [], [], []
    # Create dgp
    random_integer = rng_runs.integers(low=15, high=100_000_000, size=1)
    # Get data ready for ee 
    d = dgp_grace(k=6, grace=gp, nobs=nobs, timevarying=0, rngseed=random_integer + (2 * i), plan=None)
    # Creating the cloned and censored data 
    d_vec = clone_and_censor(d)

    ### Input for the data in 
    # for a=1, censoring model
    t_cens1 = np.asarray(d_vec.loc[(d_vec["plan"] == 1), "t_in"])
    c1 = np.asarray(d_vec.loc[(d_vec["plan"] == 1), 'censor'])
    # W1 = np.asarray(d_vec.loc[(d_vec["plan"] == 1)][['W']])
    W1 = np.asarray(d_vec.loc[(d_vec["plan"] == 1)][['W']])
    Wtmp = np.asarray(d_vec.loc[(d_vec["plan"] == 1)][['W', 'plan']])
    censor_times1 = list(np.unique(d_vec.loc[(d_vec["plan"] == 1) & (d_vec["censor"] == 1), "t_in"]))
    n_censor_times1 = len(censor_times1)
    init_plr_censor1 = [ 0.2, ] * W1.shape[1] + [0.1] + [0.5]*(n_censor_times1 - 1)
    t_trt1 = np.asarray(d_vec.loc[(d_vec["plan"] == 1), "t_trt"])
    trt1 = np.asarray(d_vec.loc[(d_vec["plan"] == 1), "trt_obs"])

    # Inputs for a=0, censoring model
    t_cens0 = np.asarray(d_vec.loc[(d_vec["plan"] == 0), "t_in"])
    c0 = np.asarray(d_vec.loc[d_vec['plan'] == 0, 'censor'])
    W0 = np.asarray(d_vec.loc[d_vec['plan'] == 0, ['W']])
    censor_times0 = list(np.unique(d_vec.loc[(d_vec['plan'] == 0) & (d_vec['censor']==1), 't_in']))
    n_censor_times0 = len(censor_times0)
    init_plr_censor0 = [0.15, ] * W0.shape[1] + [0.12] + [0.55]*(n_censor_times0 - 1)
    t_trt0 = np.asarray(d_vec.loc[(d_vec["plan"] == 0), 't_trt'])
    trt0 = np.asarray(d_vec.loc[(d_vec["plan"] == 0), 'trt_obs'])

    ### Inputs for outcome model  
    t_delta1 = np.asarray(d_vec.loc[(d_vec['plan']==1), "t_out"])
    t_delta0 = np.asarray(d_vec.loc[(d_vec['plan']==0), "t_out"])
    delta1 = np.asarray(d_vec.loc[(d_vec['plan']==1), "event"])
    delta0 = np.asarray(d_vec.loc[(d_vec['plan']==0), "event"])
    unique_delta_times1 = list(np.unique(d_vec.loc[(d_vec["plan"] == 1) & (d_vec["event"] == 1), "t_out"]))
    unique_delta_times0 = list(np.unique(d_vec.loc[(d_vec["plan"] == 0) & (d_vec["event"] == 1), "t_out"]))
    n_delta_times1 = len(unique_delta_times1)
    n_delta_times0 = len(unique_delta_times0)

    # Inits for the root finder to start with 
    tot_timepoints = np.unique(d_vec['t_in'])
    n_timepoints = len(tot_timepoints)
    init_rd = [0.011] * n_timepoints
    # init_rd = [0.011193, 0.011111, 0.016134, 0.014702, 0.010743, 0.008038]

    # init_risks1 = [0.2, ] * n_timepoints
    init_risks1 = [0.1206, 0.1306, 0.143251, 0.153401, 0.162823, 0.169993]
    # init_risks0 = [0.25, ] * n_timepoints
    init_risks0 = [0.109407, 0.119489, 0.127117, 0.138699, 0.15208, 0.161955]

    init_plr_outcome1 = [0.6,] + [0.2,]*(n_timepoints-1) # create a new one now with a different candidtate init coefficient for the intercept 
    init_plr_outcome0 = [0.5,] + [0.1,]*(n_timepoints-1) # create a new one now with a different candidtate init coefficient for the intercept 
    init_plr_censor1 = [ 0.2, ] * W1.shape[1] + [0.1] + [0.5]*(n_censor_times1 - 1)
    init_plr_censor0 = [0.15, ] * W0.shape[1] + [0.12] + [0.55]*(n_censor_times0 - 1)

    inits = init_rd + init_risks1 + init_risks0 + init_plr_outcome1 + init_plr_outcome0 + init_plr_censor1 + init_plr_censor0

    # Now estimate
    estr = MEstimator(psi_ccw_plr, init=inits) 
    estr.estimate()
    # Now take the rd outputs
    point_rd = estr.theta[:n_timepoints]
    variance_rd = (np.diag(estr.variance))[:n_timepoints]
    lcl_rd = estr.confidence_intervals()[:n_timepoints,0]
    ucl_rd = estr.confidence_intervals()[:n_timepoints,1]
    # Now take the risk 1 outputs
    point_r1 = estr.theta[n_timepoints:(n_timepoints*2)]
    variance_r1 = (np.diag(estr.variance))[n_timepoints:(n_timepoints*2)]
    lcl_r1 = estr.confidence_intervals()[n_timepoints:(n_timepoints*2),0]
    ucl_r1 = estr.confidence_intervals()[n_timepoints:(n_timepoints*2),1]
    # Now take the risk 0 outputs
    point_r0 = estr.theta[(n_timepoints*2):(n_timepoints*3)]
    variance_r0 = (np.diag(estr.variance))[(n_timepoints*2):(n_timepoints*3)]
    lcl_r0 = estr.confidence_intervals()[(n_timepoints*2):(n_timepoints*3),0]
    ucl_r0 = estr.confidence_intervals()[(n_timepoints*2):(n_timepoints*3),1]
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
        cols = ['point_rd', 'variance_rd', 'lcl_rd', 'ucl_rd',
              'point_r1', 'variance_r1', 'lcl_r1', 'ucl_r1',
              'point_r0', 'variance_r0', 'lcl_r0', 'ucl_r0']
        for t_out, t_results, truth, risk1, risk0 in zip([1, 2, 3, 4, 5, 6], [r1, r2, r3, r4, r5, r6], [t1, t2, t3, t4, t5, t6], [r11, r12, r13, r14, r15, r16], [r01, r02, r03, r04, r05, r06]):
            results = pd.DataFrame(t_results, columns=cols)
            results["t_out"] = t_out
            results["truth_rd"] = truth
            results["truth_r1"] = risk1
            results["truth_r0"] = risk0
            results.to_csv(f"./out/v{vers}simIPCW_tout_{t_out}_n{nobs}_{i}.csv")


##################################################
# SECTION - Now taking the results and evaluating performance 
##################################################
# I want to calculate the 
cols = ['point_rd', 'variance_rd', 'lcl_rd', 'ucl_rd',
        'point_r1', 'variance_r1', 'lcl_r1', 'ucl_r1',
        'point_r0', 'variance_r0', 'lcl_r0', 'ucl_r0']
for t_out, t_results, truth, risk1, risk0 in zip([1, 2, 3, 4, 5, 6], [r1, r2, r3, r4, r5, r6], [t1, t2, t3, t4, t5, t6], [r11, r12, r13, r14, r15, r16], [r01, r02, r03, r04, r05, r06]):
    results = pd.DataFrame(t_results, columns=cols)
    results["t_out"] = t_out
    results["truth_rd"] = truth
    results["truth_r1"] = risk1
    results["truth_r0"] = risk0
    results.to_csv(f"./out/v{vers}finIPCW_tout_{t_out}_n{nobs}_{runs}.csv")

# !SECTION - 
# 
# ##################################################
# #***** SECTION - Calculating the numbers 
# ##################################################
# 
# # risk1 v4
# # Plan A4 pseudopop:
# #  Time
# # Plan A pseudopop:
# #  Time
# # 1    0.115845
# # 2    0.125019
# # 3    0.139782
# # 4    0.149750
# # 5    0.160729
# # 6    0.172827
# # Name: Surv prob, dtype: float64
# 
# # risk1 v orig
# # Time	Surv prob
# # 0	1	0.116354
# # 1	2	0.125568
# # 2	3	0.139805
# # 3	4	0.149432
# # 4	5	0.159938
# # 5	6	0.171359
# # risk0
# # Time	Surv prob
# # 0	1	0.104188
# # 1	2	0.112864
# # 2	3	0.122362
# # 3	4	0.133758
# # 4	5	0.146095
# # 5	6	0.159402
# 
# # # The statistics
# # results = pd.read_csv('sim_tout_2_n3000.csv')
# # results = pd.read_csv('sim_tout_5_n3000.csv')
# # results = pd.read_csv('sim_tout_4_n3000.csv')
# # results = pd.read_csv('sim_tout_3_n3000.csv')
# # results = pd.read_csv('sim_tout_2_n3000.csv')
# # results = pd.read_csv('sim_tout_1_n3000.csv')
# # num = 3 # this one has CI coverage of 87.2, check this , bias eliminated coverage was 92.5 ish
# # results = pd.read_csv(f'fsim_tout_{num}_n3000_500.csv')
# 
# num = 3 # this one has CI coverage of 87.2, check this , bias eliminated coverage was 92.5 ish
# results = pd.read_csv(f'zsim_tout_{num}_n3000_2000.csv')
# ['truth_risk1', 'truth,risk0'] in (results.columns)
# results.columns
# # Try a logic lop 
# n_timepoints = 6
# # Logic loop
# for i in range(1, n_timepoints+1):
#     print(i)
#     results = pd.read_csv(f'zsim_tout_{i}_n3000_2000.csv')
# 
# out = pd.DataFrame(columns=['Measure', 't_out', 'Bias', 'ESE', 'SER', 'RMSE', 'Coverage', 'MISS-Bias'])
# results = pd.read_csv(f'zsim_tout_{i}_n3000_2000.csv')
# print(i)
# # Empty set = F, nonempty set = T
# missing = set(['truth_risk1', 'truth_risk0']) - set(results.columns)
# 
# # Logic loop
# out_rows = []
# for i in range(1, n_timepoints+1):
#     print(i)
#     results = pd.read_csv(f'zsim_tout_{i}_n3000_2000.csv')
#     t_out = i 
#     if not missing:
#         print('columns do not exist. creating cols')
#         results['truth_r1'] = results['truth_risk1']    
#         results['truth_r0'] = results['truth_risk0']     
#     else: 
#         print("columns already exist")
# 
#     for est in ['r1', 'r0', 'rd']:
#         bias_vec =  results[f'point_{est}'] - results[f'truth_{est}']  
# 
#         bias = bias_vec.sum() / results.shape[0]
# 
#         emp_se = np.sqrt(((bias_vec)**2).sum() / (results.shape[0] - 1))
# 
#         ser = np.average(np.sqrt(results[f'variance_{est}'])) / emp_se
# 
#         rmse = np.sqrt(((bias_vec**2).sum()) / results.shape[0])
# 
#         cov_vec = np.where(
#             (results[f'lcl_{est}'] <= results[f'truth_{est}']) &
#             (results[f'truth_{est}'] <= results[f'ucl_{est}']),
#             1, 0
#         )
#         coverage = cov_vec.sum() / results.shape[0]
# 
#         b_e_c_vec = np.where((results[f'lcl_{est}'] <= results[f'point_{est}'].mean()) & (results[f'point_{est}'].mean() <= results[f'ucl_{est}'] ), 1, 0)
#         b_e_c = b_e_c_vec.sum() / results.shape[0]
# 
#         out_rows.append({
#             'Measure': est,
#             't_out': t_out,
#             'Bias': bias,
#             'Emperical_SE': emp_se,
#             'SE_ratio': ser,
#             'RMSE': rmse,
#             'CI_Coverage': coverage,
#             'Bias_elim_coverage': b_e_c
#         })
#     out = pd.DataFrame(out_rows)
# 
# out_sorted = out.sort_values(by=['Measure', 't_out']).reset_index(drop=True)
# 
# out_sorted.to_csv('out_sorted_n3k_runs2k.csv')
# # Measure	t_out	Bias	Emperical_SE	SE_ratio	RMSE	CI_Coverage	Bias_elim_coverage
# # r0	1	0.000878	0.00593	1.016	0.00593	0.955	0.955
# # r0	2	0.000871	0.00616	1.019	0.00616	0.959	0.958
# # r0	3	0.000900	0.00654	1.006	0.00654	0.953	0.952
# # r0	4	0.000844	0.00681	1.015	0.00681	0.958	0.950
# # r0	5	0.000844	0.00719	1.008	0.00719	0.957	0.957
# # r0	6	0.000833	0.00765	0.991	0.00764	0.951	0.952
# # 							
# # r1	1	-0.000902	0.00571	1.021	0.00571	0.947	0.952
# # r1	2	-0.000865	0.00591	1.020	0.00591	0.953	0.955
# # r1	3	-0.001942	0.00744	0.995	0.00744	0.938	0.951
# # r1	4	-0.001776	0.00817	0.999	0.00817	0.940	0.952
# # r1	5	-0.001454	0.00882	1.009	0.00882	0.943	0.947
# # r1	6	-0.001191	0.00967	0.995	0.00967	0.943	0.949
# # 							
# # rd	1	-0.001780	0.00334	0.831	0.00334	0.886	0.944
# # rd	2	-0.001736	0.00346	0.850	0.00346	0.898	0.944
# # rd	3	-0.002842	0.00658	0.867	0.00658	0.877	0.931
# # rd	4	-0.002620	0.00771	0.922	0.00771	0.914	0.941
# # rd	5	-0.002297	0.00883	0.945	0.00882	0.926	0.947
# # rd	6	-0.002024	0.00993	0.953	0.00993	0.935	0.942
# 
# coverage = np.where((results[f'lcl_{est}'] <= results[f'truth_{est}']) & (results[f'truth_{est}'] <= results[f'ucl_{est}'] ), 1, 0)
# print(f't{num} {est} CI coverage: {results['coverage'].sum() / results.shape[0]}')
# np.where((results[f'lcl_{est}'] <= results[f'truth_{est}']) & (results[f'truth_{est}'] <= results[f'ucl_{est}'] ), 1, 0)
# print(f't{num} {est} CI coverage: {results['coverage'].sum() / results.shape[0]}')
# # results.dtypes
# print(results.columns)
# results.shape
# 
# # results.head()
# results['truth_r1'] = results['truth_risk1']    
# results['truth_r0'] = results['truth_risk0']   
# # # results['truth_r1'] - results['truth_r0']
# est = 'rd'
# est = 'r0'
# est = 'r1'
# results['coverage'] = np.where((results[f'lcl_{est}'] <= results[f'truth_{est}']) & (results[f'truth_{est}'] <= results[f'ucl_{est}'] ), 1, 0)
# print(f't{num} {est} CI coverage: {results['coverage'].sum() / results.shape[0]}')
# # 
# # They probably already know the questions beforehand.
# # test_truth =  0.017443000000000014
# # # test_truth =   0.011705573473359453
# # results['coverage'] = np.where((results[f'lcl_{est}'] <= test_truth) & (test_truth <= results[f'ucl_{est}'] ), 1, 0)
# # print(f'CI coverage: {results['coverage'].sum() / results.shape[0]}')
# # 
# results['avg_theta'] = results[f'point_{est}'].mean() #0.14640871005853215
# results['bias_elim_coverage'] = np.where((results[f'lcl_{est}'] <= results['avg_theta']) & (results['avg_theta'] <= results[f'ucl_{est}'] ), 1, 0)
# print(f'Bias Elminated CI coverage: {results['bias_elim_coverage'].sum() / results.shape[0]}')
# results[['truth_rd', 'avg_theta', 'truth_r1', 'truth_r0' ]]
# # 
# results['bias'] = results[f'point_{est}'] - results[f'truth_{est}']
# print(f"Bias: {results['bias'].sum() / results.shape[0]}")
# print(f"MSE: {(results['bias']**2).sum() / results.shape[0]}")
# print(f'EmpSE: {np.sqrt(((results['bias'])**2).sum() / (results.shape[0] - 1))}')
# 
# results['coverage'] = np.where((results['lcl'] <= results['truth']) & (results['truth'] <= results['ucl'] ), 1, 0)
# print(f'CI coverage: {results['coverage'].sum() / results.shape[0]}')
# results['avg_theta'] = results['point_rd'].mean()
# results['bias_elim_coverage'] = np.where((results['lcl'] <= results['avg_theta']) & (results['avg_theta'] <= results['ucl'] ), 1, 0)
# print(f'Bias Elminated CI coverage: {results['bias_elim_coverage'].sum() / results.shape[0]}')
# results['bias'] = results['point_rd'] - results['truth']
# print(f"Bias: {results['bias'].sum() / results.shape[0]}")
# print(f"MSE: {(results['bias']**2).sum() / results.shape[0]}")
# print(f'EmpSE: {np.sqrt(((results['bias'])**2).sum() / (results.shape[0] - 1))}')
# # ESE
# 
# # SER
# 
# # RMSE
# 
# # Coverage
# 
# # MISS-Bias
# 
# #**** !SECTION - 
# 
# ##################################################
# # SECTION - Troubleshooting against a different truth (not sure if this is relevant)
# ##################################################
# 
# num = 3 # this one has CI coverage of 87.2, check this , bias eliminated coverage was 92.5 ish
# results = pd.read_csv(f'zsim_tout_{num}_n3000_2000.csv')
# results.columns
# results['truth_r1'] = results['truth_risk1']    
# results['truth_r0'] = results['truth_risk0']  
# # pretending truth is" 
# # # Plan A pseudopop:
# # #  Time
# # # 1    0.115482
# # # 2    0.124627
# # # 3    0.139587
# # # 4    0.149635
# # # 5    0.160562
# # # 6    0.172459
# est = 'rd'
# results['coverage'] = np.where((results[f'lcl_{est}'] <= results[f'truth_{est}']) & (results[f'truth_{est}'] <= results[f'ucl_{est}'] ), 1, 0)
# print(f't{num} {est} CI coverage: {results['coverage'].sum() / results.shape[0]}')
# 
# est = 'rd'
# results['coverage'] = np.where((results[f'lcl_{est}'] <= results[f'truth_{est}']) & (results[f'truth_{est}'] <= results[f'ucl_{est}'] ), 1, 0)
# print(f't{num} {est} CI coverage: {results['coverage'].sum() / results.shape[0]}')
# 
# tmp_tru=0.139805-0.12236199
# results['coverage']=np.where((results[f'lcl_{est}'] <= tmp_tru) & (tmp_tru <= results[f'ucl_{est}'] ), 1, 0)
# print(f't{num} {est} CI coverage: {results['coverage'].sum() / results.shape[0]}')
# 
# 
# # !SECTION - 
# 
# ##################################################
# # SECTION - thetas output into a table from just one run example 
# ##################################################
# # estr.theta 
# # out_sandwich = pd.DataFrame({
# #     "theta":estr.theta[:n_timepoints],
# #     "variance": (np.diag(estr.variance))[:n_timepoints],
# #     "lcl":estr.confidence_intervals()[:n_timepoints,0],
# #     "ucl":estr.confidence_intervals()[:n_timepoints,1]
# # })
# # 
# # 
# # estr.theta[:n_timepoints] # RD
# # estr.theta[n_timepoints:(n_timepoints*2)] # risk1
# # estr.theta[(n_timepoints*2):(n_timepoints*3)] # risk 0
# # estr.theta[(n_timepoints*3):(n_timepoints*4)] # plr outcome model 1
# # # array([-1.98676057, -4.46521823, -4.21541662, -4.42371384, -4.48702714,
# # #        -4.75154766])
# # estr.theta[(n_timepoints*4):(n_timepoints*5)] # plr outcome model 0
# # # array([-2.09681374, -4.46973284, -4.74001774, -4.30894518, -4.14895103,
# # #        -4.44104518])
# # index_plr = (n_timepoints*5) + W1.shape[1] + n_censor_times1 # the index 
# # estr.theta[(n_timepoints*5):index_plr] # plr censor model 1
# # estr.theta[index_plr:] # plr censor model 0
# # 
# # out_sandwich = pd.DataFrame({
# #     "theta":estr.theta,
# #     "stderr": np.sqrt(np.diag(estr.variance)),
# #     "lcl":estr.confidence_intervals()[:,0],
# #     "ucl":estr.confidence_intervals()[:,1]
# # })
# # 
# # estr.theta[:n_timepoints] # RD
# # estr.theta[n_timepoints:(n_timepoints*2)] # risk1
# # estr.theta[(n_timepoints*2):(n_timepoints*3)] # risk 0
# # estr.theta[(n_timepoints*3):(n_timepoints*4)] # plr outcome model 1
# # # array([-1.98676057, -4.46521823, -4.21541662, -4.42371384, -4.48702714,
# # #        -4.75154766])
# # estr.theta[(n_timepoints*4):(n_timepoints*5)] # plr outcome model 0
# # # array([-2.09681374, -4.46973284, -4.74001774, -4.30894518, -4.14895103,
# # #        -4.44104518])
# # index_plr = (n_timepoints*5) + W1.shape[1] + n_censor_times1 # the index 
# # estr.theta[(n_timepoints*5):index_plr] # plr censor model 1
# # estr.theta[index_plr:] # plr censor model 0
# # 
# # out_thetas = pd.DataFrame({'theta1_plrcensor':estr.theta[(n_timepoints*3):(n_timepoints*4)],
# #               'theta0_plrcensor': estr.theta[(n_timepoints*4):(n_timepoints*5)],
# #               'risk1' : estr.theta[n_timepoints:(n_timepoints*2)], # risk1,
# #               'risk0' : estr.theta[(n_timepoints*2):(n_timepoints*3)], # risk 0
# #               'RD' : estr.theta[:n_timepoints]
# #               })
# # print(out_thetas)
# # 
# # print(out_sandwich)
# # 
# # # Getting the parameters for the nuisance models 
# # rows = []
# # rows.append({"var":"intercept", "ee_theta0":estr.theta[index_plr:][1], "ee_theta1":estr.theta[(n_timepoints*5):index_plr][1]})
# # rows.append({"var":"T_1", "ee_theta0":estr.theta[index_plr:][2]})
# # rows.append({"var":"T_2", "ee_theta0":estr.theta[index_plr:][3] })
# # rows.append({"var":"T_3", "ee_theta0": None})
# # rows.append({"var":"T_4", "ee_theta0": None})
# # rows.append({"var":"T_5", "ee_theta0": None})
# # rows.append({"var":"W", "ee_theta0": estr.theta[index_plr:][0],"ee_theta1": estr.theta[(n_timepoints*5):index_plr][0] })
# # out_plrcensor = pd.DataFrame(rows)
# # d_vec.shape
# # d_vec.dtypes
# # d_vec.head()
# # !SECTION - 
# 
# 
# # !SECTION - 