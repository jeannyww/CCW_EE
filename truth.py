#############################################################################
# Getting the true values for the simulations
# JHW (2026-01-19)
# Details: NA
#############################################################################

##################################################
# Loading Packages
##################################################

import numpy as np
import pandas as pd
import statsmodels.api as sm
from dgp import dgp_grace, dgp_grace_planA4, dgp_grace_set_plan

##################################################
# SECTION - Observed data
##################################################

gp = 2 
nobs=10_000_000
i = 1
# Create dgp - this is the observed, 'natural' course

# d_obs = dgp_grace(k=6, grace=gp, nobs=10_000_000, timevarying=0, rngseed=1_394, plan=None)
# km_obs = sm.SurvfuncRight(d_obs["t_out"], d_obs["Y"], d_obs["t_in"])
# print(km_obs.summary())
# risk_obs =1- km_obs.summary()["Surv prob"]
# # risk_obs.to_csv('./true_risk1_method0.csv')
# print("Plan B pseudopop 10_000_000:\n", risk_obs)
# #       Surv prob  Surv prob SE  num at risk  num events
# # Time                                                  
# # 1      0.990582      0.000031     10000000     94178.0
# # 2      0.980347      0.000044      9905822    102348.0
# # 3      0.969181      0.000055      9803474    111669.0
# # 4      0.957077      0.000064      9691805    121034.0
# # 5      0.943909      0.000073      9570771    131680.0
# # 6      0.929752      0.000081      9439091    141573.0
# # Plan B pseudopop 10_000_000:
# #  Time
# # 1    0.009418
# # 2    0.019653
# # 3    0.030819
# # 4    0.042923
# # 5    0.056091
# # 6    0.070248
# # Name: Surv prob, dtype: float64

# trt_observed = d_obs[['id', 'plan_000','plan_100', 'plan_010', 'plan_001', 'plan_NA']].groupby('id').first().value_counts()
# plan_000  plan_100  plan_010  plan_001  plan_NA
# 1         0         0         0         0          6782981
# 0         1         0         0         0          1062236
#           0         1         0         0          1024941
#                     0         1         0           975245
#                               0         1           154597
# Name: count, dtype: int64
# [0.6782981, 0.1062236, 0.1024941, 0.0975245, 0.0154597]
# 
# print((trt_observed.array)/nobs)

# !SECTION - Observed data

##################################################
# SECTION - PLAN 1 METHOD 0 - simulate such that natural course followed until right before the grace period, then set anyone who has not had A==1 at t==gp to 1 so that force 001
##################################################

# d1 = dgp_grace(k=6, grace=gp, nobs=100_000, timevarying=0, rngseed=1_394, plan=1)
# d1.to_csv('./out/test.csv')
d1 = dgp_grace(k=6, grace=gp, nobs=10_000_000, timevarying=0, rngseed=1_394, plan=1)
# d0 = dgp_grace_set_plan(k=6, grace=gp, nobs_target=10_000_000, timevarying=0,plan=0, rngseed=1_394, max_rounds=1_000_000) # WRONG?
# d0 = dgp_grace(k=6, grace=gp, nobs=10_000, timevarying=0, rngseed=1_394, plan=0)
km1_v0 = sm.SurvfuncRight(d1["t_out"], d1["Y"], d1["t_in"])
print(km1_v0.summary())
risk1_v0 =1- km1_v0.summary()["Surv prob"]
print("Plan B pseudopop 10_000_000:\n", risk1_v0)
#       Surv prob  Surv prob SE  num at risk  num events
# Time                                                  
# 1      0.990582      0.000031     10000000     94178.0
# 2      0.980347      0.000044      9905822    102348.0
# 3      0.963294      0.000059      9803474    170532.0
# 4      0.952096      0.000068      9632942    111979.0
# 5      0.939864      0.000075      9520963    122322.0
# 6      0.926587      0.000082      9398641    132770.0
# Plan B pseudopop 10_000_000:
#  Time
# 1    0.009418
# 2    0.019653
# 3    0.036706
# 4    0.047904
# 5    0.060136
# 6    0.073413
# Name: Surv prob, dtype: float64

risk1_v0.to_csv('./true2_risk1_method0.csv')
# On 10_000_000 pseudopop


# !SECTION - PLAN 1 METHOD 0 

##################################################
# SECTION - Plan 1 METHOD 1 FOR PLAN 1 SIMULATE TRUTH One sweep one go method for simulating Plan A
##################################################

da_tru = dgp_grace_planA4(k=6, grace=gp, nobs_target=10_000_000, timevarying=0, rngseed=1_394)
km_tru = sm.SurvfuncRight(da_tru["t_out"], da_tru["Y"], da_tru["t_in"])
print(km_tru.summary())
risk_tru= km_tru.summary()

riska4 = 1 - risk_tru["Surv prob"]
print("Plan A pseudopop 10_000_000:\n", riska4)
riska4.to_csv('./true2_risk1_method1.csv')
# first round: 0
# round 1: simulated 10000000, adhered found 3061249, n_y_within_gp found 154392, kept = 3215641 or 3215641
# nonadhered 6784359
# remaining 6784359
# now calculating the remaining
#       Surv prob  Surv prob SE  num at risk  num events
# Time                                                  
# 1      0.990615      0.000030     10000000     93852.0
# 2      0.980434      0.000044      9906148    101807.0
# 3      0.963713      0.000059      9804341    167209.0
# 4      0.952692      0.000067      9637132    110214.0
# 5      0.940666      0.000075      9526918    120253.0
# 6      0.927560      0.000082      9406665    131063.0
# Plan A pseudopop 10_000_000:
#  Time
# 1    0.009385
# 2    0.019566
# 3    0.036287
# 4    0.047308
# 5    0.059334
# 6    0.072440
# Name: Surv prob, dtype: float64

# !SECTION - 


##################################################
# SECTION - Plan 1 METHOD 2  Iterative Simulation method for plan A
##################################################

# Plan A using the iterative simulation method 

# Here we force everyone who had a 000 by time=grace period to have 001. This might not be the correct counterfactual.  but using an iterative simulation where (maybe I don't have the same gaurenteed distribution of the type of 001? as in the confounder distribution might not be the same?)
d1_v2 = dgp_grace_set_plan(k=6, grace=gp, nobs_target=10_000_000, timevarying=0, plan=1, rngseed=1_394,max_rounds=1_000_000) 

km1_v2 = sm.SurvfuncRight(d1_v2["t_out"], d1_v2["Y"], d1_v2["t_in"])
risk1_v2 =1- km1_v2.summary()["Surv prob"]
print(km1_v2.summary())

print("METHOD 2 Plan A pseudopop 10_000_000:\n", risk1_v2)

risk1_v2.to_csv("./true2_risk1_method2.csv")

#       Surv prob  Surv prob SE  num at risk  num events
# Time                                                  
# 1      0.990615      0.000030     10000000     93852.0
# 2      0.980434      0.000044      9906148    101807.0
# 3      0.964369      0.000059      9804341    160650.0
# 4      0.953694      0.000066      9643691    106747.0
# 5      0.942079      0.000074      9536944    116151.0
# 6      0.929399      0.000081      9420793    126807.0
# METHOD 2 Plan A pseudopop 10_000_000:
#  Time
# 1    0.009385
# 2    0.019566
# 3    0.035631
# 4    0.046306
# 5    0.057921
# 6    0.070601
# Name: Surv prob, dtype: float64
# !SECTION - PLAN 1 METHOD 2

##################################################
# SECTION -  PLAN 0 METHOD 1
##################################################

d0 = dgp_grace(k=6, grace=gp, nobs=10_000_000, timevarying=0, rngseed=1_394, plan=0)

km0 = sm.SurvfuncRight(d0["t_out"], d0["Y"], d0["t_in"])
risk0 =1- km0.summary()["Surv prob"]
print(km0.summary())

print("Plan B pseudopop 10_000_000:\n", risk0)

#       Surv prob  Surv prob SE  num at risk  num events
# Time                                                  
# 1      0.991301      0.000029     10000000     86986.0
# 2      0.981782      0.000042      9913014     95195.0
# 3      0.971387      0.000053      9817819    103950.0
# 4      0.958910      0.000063      9713869    124773.0
# 5      0.945384      0.000072      9589096    135258.0
# 6      0.930775      0.000080      9453838    146090.0
# Plan B pseudopop 10_000_000:
#  Time
# 1    0.008699
# 2    0.018218
# 3    0.028613
# 4    0.041090
# 5    0.054616
# 6    0.069225
# Name: Surv prob, dtype: float64
risk0.to_csv("./true2_risk0.csv")
# risk0.to_csv("./true_risk0_setplan.csv")
print(risk0)

print(risk1_v2 - risk0)

# 1    0.000687
# 2    0.001348
# 3    0.007018
# 4    0.005215
# 5    0.003304
# 6    0.001376
# Name: Surv prob, dtype: float64
# !SECTION - 

