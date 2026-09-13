#############################################################################
# Processing the results from simulation_iptw.py    
# JHW (2026-08-05)
# Details: NA
#############################################################################

##################################################
# Loading Packages
##################################################

import numpy as np
import pandas as pd

##################################################
# SECTION - R1 TRUTH VERSIONS
##################################################
# RISKS TRUTH 

true_risk0 = pd.read_csv("true2_risk0.csv")
truth_r0 = np.asarray(true_risk0['Surv prob'])
print(true_risk0)

#    Time  Surv prob
# 0     1   0.008699
# 1     2   0.018218
# 2     3   0.028613
# 3     4   0.041090
# 4     5   0.054616
# 5     6   0.069225

true_risk1_method1 = pd.read_csv('true2_risk1_method1.csv')
truth_r1_meth1 = np.asarray(true_risk1_method1['Surv prob'])
print(true_risk1_method1)
#    Time  Surv prob
# 0     1   0.009385
# 1     2   0.019566
# 2     3   0.036287
# 3     4   0.047308
# 4     5   0.059333
# 5     6   0.072440

# Method 2 is the true counterfactual that I had evaluated
true_risk1_method2 = pd.read_csv('true2_risk1_method2.csv')
truth_r1_meth2 = np.asarray(true_risk1_method2['Surv prob'])
print(true_risk1_method2)
#    Time  Surv prob
# 0     1   0.009385
# 1     2   0.019566
# 2     3   0.035631
# 3     4   0.046306
# 4     5   0.057921
# 5     6   0.070601

true_risk1_method0 = pd.read_csv('true2_risk1_method0.csv')
truth_r1_meth0 = np.asarray(true_risk1_method0['Surv prob'])
print(true_risk1_method0)

#    Time  Surv prob
# 0     1   0.009418
# 1     2   0.019653
# 2     3   0.036706
# 3     4   0.047904
# 4     5   0.060136
# 5     6   0.073413

# !SECTION - R1 TRUTH VERSIONS

##################################################
# SECTION - Intermediate step in case I had to split runs for large simulations
##################################################
# cols = ['point_rd', 'variance_rd', 'lcl_rd', 'ucl_rd',
#         'point_r1', 'variance_r1', 'lcl_r1', 'ucl_r1',
#         'point_r0', 'variance_r0', 'lcl_r0', 'ucl_r0']

# results = pd.DataFrame()
# res_tmp = pd.read_csv("./out/v2sim_tout_1_n50000_499.csv")
# results = pd.concat([results, res_tmp])
# res_tmp = pd.read_csv(f"./out/v2sim_tout_{t_in + 1}_n50000_{run + 99}.csv")
# results = pd.DataFrame()
# for t_in in range(0, 6):
#     print(t_in + 1)
#     results = pd.DataFrame()
#     for run in [400, 900, 1400, 1900, 2900 ]:
#         print(run + 99)
#         res_tmp = pd.read_csv(f"./out/v2sim_tout_{t_in + 1}_n50000_{run + 99}.csv")
#         results = pd.concat([results, res_tmp])
#     results.to_csv(f"./out/v2iptwsim_tout_{t_in + 1}_n50000_3000.csv")

    # results = pd.DataFrame(t_results, columns=cols)
    # results["t_out"] = t_out
    # results["truth_rd"] = truth
    # results["truth_r1"] = risk1
    # results["truth_r0"] = risk0
    # results.to_csv(f"./out/v{vers}iptwsim_tout_{t_out}_n{nobs}_{runs}.csv")
# results.shape
# out/v2sim_tout_6_n50000_2999.csv
# !SECTION - 

##################################################
# SECTION - RISKS & RDs Compiling results from iptw simulation for risks and risk differences 
##################################################

n_timepoints = 6
out_rows = []

for i in range(1, n_timepoints + 1):
    print(i)
    t_out = i 
    # res1 = pd.read_csv(f"./out/v2sim_tout_{i}_n50000_2999.csv") 
    # res2 = pd.read_csv(f"./out/v2sim_tout_{i}_n50000_4999.csv") 
    # results = pd.concat([res1, res2])
    # use this when the simulations stop halfway 
    res0 = pd.read_csv(f"./out/v2iptwsim_tout_{i}_n100000_1000.csv") 
    res1 = pd.read_csv(f"./out/v21iptwsim_tout_{i}_n100000_1000.csv")
    res2 = pd.read_csv(f"./out/v22iptwsim_tout_{i}_n100000_1000.csv")
    res3 = pd.read_csv(f"./out/v23iptwsim_tout_{i}_n100000_1000.csv")
    res4 = pd.read_csv(f"./out/v24iptwsim_tout_{i}_n100000_1000.csv")
    results = pd.concat([res0, res1, res2, res3, res4])
    # results = pd.read_csv(f"./out/v2iptwsim_tout_{i}_n20000_5000.csv") # tweaked the first A ~ model here at time ==0 to match those time >0
    # results = pd.read_csv(f"./out/v2iptwsim_tout_{i}_n3000_5000.csv") # tweaked the first A ~ model here at time ==0 to match those time >0
    # results = pd.read_csv(f"./out/v121iptwsim_tout_{i}_n3000_2000.csv") # tweaked the first A ~ model here at time ==0 to match those time >0
    
    # # # Method 0 is following the natural plan until the grace period, then everyone is set to 1 (ie, dgp_grace(....plan=1))    
    # results['truth_r1'] = truth_r1_meth0[i - 1]
    # results['truth_r0'] = truth_r0[i - 1]
    # results['truth_rd'] = truth_r1_meth0[i - 1] - truth_r0[i - 1]

# #     # Method 2 is iterative dgp_grace_set_plan
    # results['truth_r1'] = truth_r1_meth2[i - 1]
    # results['truth_r0'] = truth_r0[i - 1]
    # results['truth_rd'] = truth_r1_meth2[i - 1] - truth_r0[i - 1]

# #     #     # Method 1 is dgp_grace_planA4 (non iterative)
    # results['truth_r1'] = truth_r1_meth1[i - 1]
    # results['truth_r0'] = truth_r0[i - 1]
    # results['truth_rd'] = truth_r1_meth1[i - 1] - truth_r0[i - 1]


    for est in ['r1', 'r0', 'rd']:
        bias_vec =  results[f'point_{est}'] - results[f'truth_{est}']  

        bias = bias_vec.sum() / results.shape[0]

        emp_se = np.sqrt(((bias_vec)**2).sum() / (results.shape[0] - 1))

        ser = np.average(np.sqrt(results[f'variance_{est}'])) / emp_se

        rmse = np.sqrt(((bias_vec**2).sum()) / results.shape[0])

        cov_vec = np.where(
            (results[f'lcl_{est}'] <= results[f'truth_{est}']) &
            (results[f'truth_{est}'] <= results[f'ucl_{est}']),
            1, 0
        )
        coverage = cov_vec.sum() / results.shape[0]

        b_e_c_vec = np.where((results[f'lcl_{est}'] <= results[f'point_{est}'].mean()) & (results[f'point_{est}'].mean() <= results[f'ucl_{est}'] ), 1, 0)
        b_e_c = b_e_c_vec.sum() / results.shape[0]

        out_rows.append({
            't_out': t_out,
            'Measure': est,
            'Bias': bias,
            'Emperical_SE': emp_se,
            'SE_ratio': ser,
            'RMSE': rmse,
            'CI_Coverage': coverage
            # 'Bias_elim_coverage': b_e_c,
            # 'CI_Coverage0': coverage
        })
    out = pd.DataFrame(out_rows)

out_sorted = out.sort_values(by=['Measure', 't_out']).reset_index(drop=True)    
print(out_sorted)

out_sorted.to_csv(f"./out/_iptw_simout_tmp.csv")

# !SECTION - Compiling results from iptw simulation








































