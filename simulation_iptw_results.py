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
# truth_rd	truth_r1	truth_r0
true_risk0_method3 = pd.read_csv("true_risk0_method3.csv")
truth_r0_meth3 = np.asarray(true_risk0_method3['Surv prob'])
print(true_risk0_method3)

# Time	Surv prob
# 0	1	0.131668
# 1	2	0.141218
# 2	3	0.150412
# 3	4	0.161534
# 4	5	0.173563
# 5	6	0.186493

true_risk0_method2 = pd.read_csv("true_risk0_method2.csv")
truth_r0_meth2 = np.asarray(true_risk0_method2['Surv prob'])
print(true_risk0_method2)

# Time	Surv prob
# 0	1	0.090515
# 1	2	0.097058
# 2	3	0.106709
# 3	4	0.118428
# 4	5	0.131052
# 5	6	0.144650

true_risk0 = pd.read_csv("true_risk0.csv")
truth_r0 = np.asarray(true_risk0['Surv prob'])
print(true_risk0)
#    Time  Surv prob
# 0     1   0.104188
# 1     2   0.112864
# 2     3   0.122362
# 3     4   0.133758
# 4     5   0.146095
# 5     6   0.159402

true_risk1_method1 = pd.read_csv('true_risk1_method1.csv')
truth_r1_meth1 = np.asarray(true_risk1_method1['Surv prob'])
print(true_risk1_method1)
#    Time  Surv prob
# 0     1   0.115482
# 1     2   0.124627
# 2     3   0.139587
# 3     4   0.149635
# 4     5   0.160562
# 5     6   0.172459
true_risk1_method2 = pd.read_csv('true_risk1_method2.csv')
truth_r1_meth2 = np.asarray(true_risk1_method2['Surv prob'])
print(true_risk1_method2)
#    Time  Surv prob
# 0     1   0.115482
# 1     2   0.124627
# 2     3   0.138890
# 3     4   0.148521
# 4     5   0.159037
# 5     6   0.170469
true_risk1_method0 = pd.read_csv('true_risk1_method0.csv')
truth_r1_meth0 = np.asarray(true_risk1_method0['Surv prob'])
print(true_risk1_method0)
#    Time  Surv prob
# 0     1    0.11422
# 1     2    0.12312
# 2     3    0.13754
# 3     4    0.14752
# 4     5    0.15821
# 5     6    0.17083

truth_rd_meth2 = truth_r1_meth2 - truth_r0
print(truth_rd_meth2)
# [0.0112937 0.011763  0.0165282 0.0147624 0.0129423 0.0110668]
truth_rd_meth0 = truth_r1_meth0 - truth_r0
print(truth_rd_meth0)
# [0.0100318 0.0102564 0.015178  0.0137616 0.012115  0.0114276]
truth_rd_meth1 = truth_r1_meth1 - truth_r0
print(truth_rd_meth1)
# [0.0112937 0.011763  0.0172249 0.0158767 0.0144672 0.0130567]



# !SECTION - R1 TRUTH VERSIONS

##################################################
# SECTION - RISKS & RDs Compiling results from iptw simulation for risks and risk differences 
##################################################

n_timepoints = 6
out_rows = []
vers = 2
for i in range(1, n_timepoints + 1):
    print(i)
    t_out = i 

    results = pd.read_csv(f"./out/v2iptwsim_tout_{i}_n3000_5000.csv")
    # results = pd.read_csv(f"./out/v101iptwsim_tout_{i}_n10000_10000.csv") # Regular simulation implementation, now with n=100,000 and 100_000 iterations # Version where I am modeling outcome with Y ~ C(T_IN) + intercept, weight matrix has non cumprod version
    # results = pd.read_csv(f"./out/v47iptwsim_tout_{i}_n10000_10000.csv") # Regular simulation implementation, now with n=100,000 and 100_000 iterations # Version where I am modeling outcome with Y ~ C(T_IN) + intercept, weight matrix has non cumprod version
    # results = pd.read_csv(f"./out/v47iptwsim_tout_{i}_n3000_10000.csv") # Regular simulation implementation, now with n=100,000 and 5000 iterations # Version where I am modeling outcome with Y ~ C(T_IN) + intercept, weight matrix has non cumprod version
    # results = pd.read_csv(f"./out/v47iptwsim_tout_{i}_n10000_5000.csv") # Regular simulation implementation, now with n=100,000 and 5000 iterations # Version where I am modeling outcome with Y ~ C(T_IN) + intercept, weight matrix has non cumprod version
    # results = pd.read_csv(f"./out/v46iptwsim_tout_{i}_n10000_5000.csv") # Regular simulation implementation, now with n=100,000 and 5000 iterations # Version where I am modeling outcome with Y ~ C(T_IN) + intercept, weight matrix remains the same, but RD EE is using the predicted risks rather than the thetas
    # results = pd.read_csv(f"./out/v45iptwsim_tout_{i}_n10000_5000.csv") # Regular simulation implementation, now with n=100,000 and 5000 iterations 
    # results = pd.read_csv(f"./out/v82iptwsim_tout_{i}_n2000_2000.csv") # Only putting the full weights for people on plan #001, everyone else has a weight of 1 
    # results = pd.read_csv(f"./out/v83iptwsim_tout_{i}_n2000_2000.csv") # Only putting the full weights for people on plan #001, everyone else has a weight of 1 
    # results = pd.read_csv(f"./out/v81iptwsim_tout_{i}_n2000_2000.csv") # WRONG VERSION Only putting the full weights for people on plan #001, everyone else has a weight of 1 
    # results = pd.read_csv(f"./out/v80iptwsim_tout_{i}_n2000_2000.csv") # doing absolutely nothing for the plan 1 (everyone to 001) weight matrix 
    # results = pd.read_csv(f"./out/v40iptwsim_tout_{i}_n2000_2000.csv")
    # results = pd.read_csv(f"./out/v30iptwsim_tout_{i}_n2000_2000.csv")
    # results = pd.read_csv(f"./out/v15iptwsim_tout_{i}_n2000_2000.csv")
    # results = pd.read_csv(f"./out/v17iptwsim_tout_{i}_n10000_2000.csv")
    # results = pd.read_csv(f"./out/v14iptwsim_tout_{i}_n10000_2000.csv")
    # results = pd.read_csv(f"./out/v13iptwsim_tout_{i}_n2000_2000.csv")
    # results = pd.read_csv(f"./out/v12iptwsim_tout_{i}_n2000_2000.csv")
    # results = pd.read_csv(f"./out/v11iptwsim_tout_{i}_n2000_5000.csv")
    # results = pd.read_csv(f"./out/v11iptwsim_tout_{i}_n2000_2000.csv")
    # results = pd.read_csv(f"./out/v10iptwsim_tout_{i}_n2000_3000.csv")
    # results = pd.read_csv(f"./out/prev/v6iptwsim_tout_{i}_n2000_2000.csv")
    # res1 = pd.read_csv(f"./out/prev/v6iptwsim_tout_{i}_n2000_2000.csv")
    # res2 = pd.read_csv(f"./out/v7iptwsim_tout_{i}_n2000_3000.csv")
    # results = pd.concat([res1, res2])
    # del res1, res2

#     # Method 1 is dgp_grace_planA4 (non iterative)
#     results['truth_r1'] = truth_r1_meth1[i - 1]
#     results['truth_r0'] = truth_r0[i - 1]
#     results['truth_rd'] = truth_rd_meth1[i - 1]   
# 
#     # Method 2 is iterative dgp_grace_set_plan
    # results['truth_r1'] = truth_r1_meth2[i - 1]
    # results['truth_r0'] = truth_r0[i - 1]
    # results['truth_rd'] = truth_r1_meth2[i - 1] - truth_r0[i - 1]
    # # # Method 0 is following the natural plan until the grace period, then everyone is set to 1 (ie, dgp_grace(....plan=1))    
    # results['truth_r1'] = truth_r1_meth0[i - 1]
    # results['truth_r0'] = truth_r0[i - 1]
    # results['truth_rd'] = truth_r1_meth0[i - 1] - truth_r0[i - 1]

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
out_sorted.to_csv(f"./out/tmpout.csv")

# !SECTION - Compiling results from iptw simulation

##################################################
# SECTION - RISKS ONLY Compiling results for risks only 
##################################################
# n_timepoints = 6
# # sim_tout_6_n3000_1500
# # Logic loop
# out_rows = []
# for i in range(1, n_timepoints+1):
#     print(i)
#     # Only output RD, psi_ccw_plr_rdonly()
#     #     ee_rd_outcome = (pred_risk1 - pred_risk0) - np.asarray(init_rds)[:, None]
#     # results = pd.read_csv(f"./out/v8finsim_tout_{i}_n2000_2000.csv")
    
#     # results = pd.read_csv(f"./out/v20finsim_tout_{i}_n2000_2000.csv")
#     # testing to ensure this is unchanged from v20 before I look at v22
#     # results = pd.read_csv(f"./out/v21finsim_tout_{i}_n2000_2000.csv")
#     # results = pd.read_csv(f"./out/v40finsim_tout_{i}_n2000_2000.csv")
#     results = pd.read_csv(f"./out/v50finsim_tout_{i}_n2000_2000.csv")
#     # results = pd.read_csv(f"./out/v60finsim_tout_{i}_n2000_2000.csv")
#     # results = pd.read_csv(f"./out/v70finsim_tout_{i}_n2000_2000.csv")
#     # if not missing:
#     #     # print('columns do not exist. creating cols')
#     # results['truth_r1'] = risk11[i - 1]    
#     # results['truth_r0'] = results['truth_risk0']     
#     # results['truth_rd'] = rd[i - 1]
#     # else: 
#         # print("columns already exist")
#     t_out=i
#     #     # Method 2 is iterative dgp_grace_set_plan
#     # results['truth_r1'] = truth_r1_meth2[i - 1]
#     # results['truth_r0'] = truth_r0[i - 1]
#     # results['truth_rd'] = truth_r1_meth2[i - 1] - truth_r0[i - 1]
#     # # # Method 0 is following the natural plan until the grace period, then everyone is set to 1 (ie, dgp_grace(....plan=1))    
#     results['truth_r1'] = truth_r1_meth0[i - 1]
#     results['truth_r0'] = truth_r0[i - 1]
#     results['truth_rd'] = truth_r1_meth0[i - 1] - truth_r0[i - 1]

#     for est in ['r1', 'r0']:
#         bias_vec =  results[f'point_{est}'] - results[f'truth_{est}']  

#         bias = bias_vec.sum() / results.shape[0]

#         emp_se = np.sqrt(((bias_vec)**2).sum() / (results.shape[0] - 1))

#         ser = np.average(np.sqrt(results[f'variance_{est}'])) / emp_se

#         rmse = np.sqrt(((bias_vec**2).sum()) / results.shape[0])

#         cov_vec = np.where(
#             (results[f'lcl_{est}'] <= results[f'truth_{est}']) &
#             (results[f'truth_{est}'] <= results[f'ucl_{est}']),
#             1, 0
#         )
#         coverage = cov_vec.sum() / results.shape[0]

#         b_e_c_vec = np.where((results[f'lcl_{est}'] <= results[f'point_{est}'].mean()) & (results[f'point_{est}'].mean() <= results[f'ucl_{est}'] ), 1, 0)
#         b_e_c = b_e_c_vec.sum() / results.shape[0]

#         out_rows.append({
#             't_out': t_out,
#             'Measure': est,
#             'Bias': bias,
#             'Emperical_SE': emp_se,
#             'SE_ratio': ser,
#             'RMSE': rmse,
#             'CI_Coverage': coverage
#             # 'Bias_elim_coverage': b_e_c
#         })
#     out = pd.DataFrame(out_rows)

# Risks_out_sorted = out.sort_values(by=['Measure', 't_out']).reset_index(drop=True)

# # !SECTION - Compiling results for risks only 






































