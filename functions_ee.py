#############################################################################
# Functions to assist with the estimating equations for CCW stack
# JHW (2026-01-11)
# Details: Here it is for the combined using PLR to estimate the hazard functions
# 2026-07-28: conducted a nother simulation using the weighting scheme for plan0-- CI coverage adequate is ok 
#############################################################################

##################################################
# Loading Packages
##################################################

import numpy as np
import pandas as pd

def inverse_logit(logodds): 
    lodds = np.asarray(logodds)
    return 1 / (1 + np.exp(-lodds))

# Helper function to broadcast down the column So that the weights will carry over? Or maybe no longer needed-- need to look into this
def colwise_broadcast(matrix):
    mask = matrix != 0
    idy = np.where(mask, np.arange(matrix.shape[0])[:, None], 0)
    idy = np.maximum.accumulate(idy, axis=0)
    return matrix[idy, np.arange(matrix.shape[1])[None, :]]


def ee_plr_censoring(theta, c, t, X, unique_censor_times):
    t = np.asarray(t)
    censor = np.asarray(c)
    X = np.asarray(X)
    xp = X.shape[1]
    beta_x = theta[:xp]
    beta_s = theta[xp:]

    ###* Residual matrix for the discrete time indicators
    # Assume disjoint 
    unique_censor_times = np.asarray(unique_censor_times)
    n_time_steps = unique_censor_times.shape[0]
    time_design_matrix = np.identity(n=len(unique_censor_times))
    time_design_matrix[:, 0] = 1
    log_odds_w = np.dot(X, beta_x) # n x len(beta_x) matrix
    log_odds_t = np.dot(time_design_matrix, beta_s) # 1 dim array
    log_odds_w_matrix = np.tile(log_odds_w, (n_time_steps, 1))

    c_obs = censor * (t == unique_censor_times[:, None]).astype(int)
    c_pred = inverse_logit(log_odds_w_matrix + log_odds_t[:, None]) # broadcast
    in_risk_set = (t >= unique_censor_times[:, None]).astype(int)

    ###* Residual matrix for Time
    t_residual_matrix = (c_obs - c_pred) * in_risk_set
    
    ###* Residual matrix for covariates
    n_ones = np.ones(shape=(1, n_time_steps))
    c_resid = np.dot(n_ones, t_residual_matrix)[0]
    x_score = c_resid[:, None] * X

    ###* Return censoring psi
    score_plogit = np.vstack([x_score.T, t_residual_matrix])
    return score_plogit

# Predicted inverse probability of being uncensored weights
def pred_plr_censor(theta, t, c, X, unique_censor_times, times_to_predict):
    # prep arrays
    t = np.asarray(t)
    c = np.asarray(c)
    X = np.asarray(X)
    xs = X.shape[1]
    beta = np.asarray(theta)
    beta_x = np.array(beta[:xs])
    beta_s = np.asarray(beta[xs:])
    n_time_steps = len(unique_censor_times)

    # S is None (figure out the parametric function form specifications for time after the disjoint)
    unique_censor_times = np.array(
        unique_censor_times
    )  # manually inserting all time points from 0 to max followup for now
    n_time_steps = len(unique_censor_times)

    # Time design matrix
    time_design_matrix = np.identity(n=len(unique_censor_times))
    time_design_matrix[:, 0] = 1  # intercept

    # Log-odds matrix for covariates
    log_odds_w = np.dot(X, beta_x)  # covariates
    log_odds_w_matrix = np.tile(log_odds_w, (n_time_steps, 1))
 
    # Log-odds matrix for Time
    log_odds_t_matrix = np.dot(time_design_matrix, beta_s)

    # Predicted probability of delta
    delta_pred_eventtimes = inverse_logit(
        log_odds_w_matrix + log_odds_t_matrix[:, None]
    )  # This is non-matrix addition: via broadcasting where 3x1 is added to the nth column of 3xn matrix

    # Makes time indexing more dynamic (for future dynamic treatment regimes potential)
    # Creating an empty array that is the same shape as the desired output
    delta_pred = np.zeros(shape=(len(times_to_predict), t.shape[0]))
    delta_pred[unique_censor_times] = delta_pred_eventtimes
    # NOTE: y_pred_furthertimes draft that attempted to patch the indexing- this is no longer needed and only works for plan B
    # n_furthertimes = len(times_to_predict) - delta_pred_eventtimes.shape[0]
    # delta_pred_furthertimes = np.zeros(shape=(n_furthertimes, t.shape[0]))
    # delta_pred = np.vstack([delta_pred_eventtimes, delta_pred_furthertimes])
    delta_survival_pred = np.cumprod(1 - delta_pred, axis=0)
    time_points_col = np.asarray(times_to_predict)[:, None]
    risk_set_cens = (t >= time_points_col).astype(int)
    weight_matrix = (np.power(delta_survival_pred, -1)) * risk_set_cens
    # weight_matrix = (np.power(delta_survival_pred, -1))
    # This makes sure that the weights are carried over for the hajek cumulative risk nonparametric estimator
    # weight_matrix = colwise_broadcast(weight_matrix)
    return weight_matrix


# Adjusting the weights so that anyone on plan 100 or 010 get all weights of 1, anyone who had the observed plan 001 will keep their weights 
def adjust_ccw_weights(plan, weights, trt, trt_times, gp):
    trt_times = np.asarray(trt_times)
    ccw_weights = weights.copy()
    if plan == 1:
        # Anyone who got treated before the gp (vs treated at the last possible moment of at the gp, in this case an observation would have trt==1 and trt_times==2), will get weights of 1
        # Same as: 
        # dtmp['ipcw_set1'] = np.where(((dtmp['plan_100'] == 1) | (dtmp['plan_010'] == 1)), 1, dtmp['ipcw'])
        ccw_weights[:,  (((trt + trt_times) <= gp) & ((trt + trt_times) > 0))] = 1
    elif plan == 0: 
        # Same as if we did: 
        # db_long['ipcw_set1'] = np.where(((db_long['plan_000'] == 1) & (db_long['t_in'] > gp) ), 1, db_long['ipcw'])
        # Whereas anyone who were not censored gets a weight of 1 for their entire rest of the time (aka anyone who got past gp for followup, but default are never ever censored, have probability of 1 for remaining uncensored), anyone with plan 000 get weight of 1 for t_in == 3, 4, 5 (t_out == 4, 5, 6)
        # ie, for rows gp+1 (ie, gp=2, so for rows index 3,4,5), and for trt_times > gp and where they were untreated)
        # ccw_weights[(gp + 1):, ((trt_times > gp) | (trt == 0))] = 1 # I think this version was the one for ispe abstract 
        pass
    return ccw_weights
# def adjust_ccw_weightsv2(plan, weights, trt, trt_times, gp):
#     trt_times = np.asarray(trt_times)
#     ccw_weights = weights.copy()
#     if plan == 1:
#         # Anyone who got treated before the gp (vs treated at the last possible moment of at the gp, in this case an observation would have trt==1 and trt_times==2), will get weights of 1
#         ccw_weights[:,  (((trt + trt_times) <= gp) & ((trt + trt_times) > 0))] = 1
#     elif plan == 0: 
#         # Whereas anyone who were not censored gets a weight of 1 for their entire rest of the time (aka anyone who got past gp for followup, but default are never ever censored, have probability of 1 for remaining uncensored)
#         # ccw_weights[:, ((trt_times > gp) | (trt == 0))] = 1 # ORIGINAL (BIASED)
#         ccw_weights[(gp + 1):, ((trt_times > gp) | (trt == 0))] = 1
#     return ccw_weights

# Estimating equations for the Outcome model
def ee_plr_outcome(theta, t, delta, t_cens, censor, weights):
    t = np.asarray(t)
    delta = np.asarray(delta)
    beta_s = np.asarray(theta)

    event_times = t[delta == 1]
    unique_event_times = np.unique(event_times)
    # n_unique_event_times = event_times.shape[0]
    time_design_matrix = np.identity(n=len(unique_event_times))
    # time_design_matrix[:, 0] = 1

    log_odds_t = np.dot(time_design_matrix, beta_s)
    y_pred = inverse_logit(log_odds_t[:, None])
    y_obs = delta * (t == unique_event_times[:, None]).astype(int)

    time_points_col = np.asarray(unique_event_times)[:, None] - 1
    censor_t_matrix = censor * (t_cens == time_points_col).astype(int)
    in_risk_set = (t >= unique_event_times[:, None]).astype(int)
    # residual_matrix = (y_obs * in_risk_set *(1 - censor_t_matrix)) - y_pred
    # residual_matrix = ((y_obs.cumsum(axis=0)) - y_pred) * in_risk_set
    # residual_matrix = (y_obs - y_pred) * in_risk_set
    residual_matrix = (y_obs - y_pred) * in_risk_set

    t_score = residual_matrix * weights * (1-censor_t_matrix.cumsum(axis=0))
    return t_score 

# Predicted risk for the outcome
# def pred_plr_outcome(theta, t, delta, censor, t_cens , times_to_predict, unique_times):
#     beta_s = np.asarray(theta)
#     t = np.asarray(t)
#     delta = np.asarray(delta)
#     times_to_predict = times_to_predict
#     # times_to_predict = list(np.unique(t))
#     unique_times = np.unique(unique_times)
# 
#     # For Predicting Risks from the Model 
#     time_design_matrix = np.identity(n=len(unique_times))
#     # time_design_matrix[:, 0] = 1
#     log_odds_t = np.dot(time_design_matrix, beta_s)
#     delta_pred_eventtimes = inverse_logit( log_odds_t[:, None])
#     delta_pred = np.zeros(shape=(len(times_to_predict), t.shape[0]))
#     delta_pred[unique_times-1, :] = delta_pred_eventtimes
#     delta_survival_pred = np.cumprod(1 - delta_pred, axis=0)
# 
#     # For masking the risks where the individuals were censored or died (no risk if they no longer contribute to data)
#     censor_t_matrix = censor * (t_cens == (unique_times[:, None] - 1)).astype(int) # indicator for exact time of censoring
#     uncensored_t_matrix = (1 - censor_t_matrix.cumsum(axis=0)) # indicator for the observed times where someone was uncensored
# 
#     # Finally, return the predicted risks, with the unobserved times masked
#     delta_risk_pred = ((1 - delta_survival_pred) * uncensored_t_matrix)
#     return delta_risk_pred
def pred_plr_outcome(theta, t, delta, times_to_predict, unique_times):
    beta_s = np.asarray(theta)
    t = np.asarray(t)
    delta = np.asarray(delta)
    # times_to_predict = list(np.unique(t))
    unique_times = np.unique(unique_times)
    time_design_matrix = np.identity(n=len(unique_times))
    # time_design_matrix[:, 0] = 1
    log_odds_t = np.dot(time_design_matrix, beta_s)
    delta_pred_eventtimes = inverse_logit( log_odds_t[:, None])
    delta_pred = np.zeros(shape=(len(times_to_predict), t.shape[0]))
    delta_pred[unique_times-1, :] = delta_pred_eventtimes
    delta_survival_pred = np.cumprod(1 - delta_pred, axis=0)

    
    delta_risk_pred = 1 - delta_survival_pred
    return delta_risk_pred

# Estimating equations for the risk (makes sure that those who are censored do not contribute to the next timepoints for risk)
def ee_risk(theta, pred_risk, censor, t_cens, unique_times):
    unique_event_times = np.asarray(unique_times)
    censor_t_matrix = censor * (t_cens == (unique_event_times[:, None] - 1)).astype(int)
    uncensored_t_matrix = (1 - censor_t_matrix.cumsum(axis=0))
    ee_risk = (pred_risk - np.asarray(theta)[:, None]) * uncensored_t_matrix
    return ee_risk

# Estimating equation for the risk difference
# FIXME - look into this 
def ee_rd(theta, risk1, risk0):
    ee_rd = (np.asarray(risk1) - np.asarray(risk0)) - np.asarray(theta)[:, None]
#     (1)
#     ee_rd_outcome = np.ones((tot_timepoints.shape[0], ee_risk1.shape[1]))*((np.asarray(init_risks1)[:, None] - np.asarray(init_risks0)[:, None]) - np.asarray(init_rds)[:, None]) # v3, and v4
# 
#     (2)
#     ee_rd_outcome = (np.zeros((tot_timepoints.shape[0], ee_risk1.shape[1])) + (np.asarray(init_risks1)[:, None] - np.asarray(init_risks0)[:, None])) - np.asarray(init_rds)[:, None]
    return ee_rd 

def clone_and_censor(data, gp, type):
    """
        data= a pd dataframe from the dgp_grace output
        gp= integer 
        type = 'd_vec', 'd_vec_censor', 'da_long', 'db_long', 'd_vec_orig', 'd_vec_c0', 'd_vec_c1'
        plan a = 
        plan b = 

    """
    data['A_lag'] =  data.groupby('id')['A'].shift(1)
    # data['A_lag'] = data['A_lag'].fillna(data.groupby('id')['A'].transform('first')).astype(int)
    data['A_lag'] = np.where(data['t_in'] == 0, 0, data['A_lag'])
    data['t_trt_lag'] = np.where(data['A_lag'] == 1, data['t_in'], 0)
    data['id_counts'] = data.groupby('id')['id'].transform('count') # static count of all rows per that id
        
    anyYgp = data[data['t_in'].between(0, gp - 1)].groupby("id")["Y"].sum() #Series.between(left, right, inclusive='both')
    anyAgp = data[data["t_in"].between(0, gp - 1)].groupby("id")["A"].sum()
    anyAYgp = pd.concat([anyAgp, anyYgp], axis=1)
    ids_y_within_gp = anyYgp[(anyAYgp["A"] == 0) & (anyAYgp["Y"] == 1)].index.to_numpy()

    ids_100 = data[(data['t_in'] == 0) & (data['A'] == 1)]['id']
    ids_010 = data[(data['t_in'] == 1) & (data['A'] == 1)]['id']
    ids_001 = data[(data['t_in'] == 2) & (data['A'] == 1)]['id']
    strictly_000 = 1- data[(data['t_in'] <= gp) & (data['id_counts'] >= 3)].groupby('id')['A'].sum()
    ids_000 = strictly_000[strictly_000 == 1].index.tolist()
    
    data['plan_100'] = np.where(data['id'].isin(ids_100), 1, 0)
    data['plan_010'] = np.where(data['id'].isin(ids_010), 1, 0)
    data['plan_001'] = np.where(data['id'].isin(ids_001), 1, 0)
    data['plan_000'] = np.where(data['id'].isin(ids_000), 1, 0)
    data['plan_NA'] = np.where(data['id'].isin(ids_y_within_gp), 1, 0) 
    data['t_trt'] = np.where(data['A'] == 1, data['t_in'], 0)
    data['intercept'] = 1

    if (type == 'd_vec_nocensor'): 
        pass
    
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
    db_long["trt_clone"] = 0 
    db_long["intercept"] = 1  # Creating intercept for estimating equation implementation
    db_long['t_trt_lagged'] = np.where(db_long['A'] == 1, db_long['t_out'] + 1, 0)
    # db_long['t_trt'] = np.where(db_long['A'] == 1, db_long['t_in'], 0)
    # Creating an indexing variable for bootstrap sampling
    # db_long["_idx"] = (db_long[["id"]]).apply(tuple)
    # db_long = db_long.set_index("_idx")

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
    da_long["trt_clone"] = 1
    da_long["intercept"] = 1
    da_long['t_trt_lagged'] = np.where(da_long['A'] == 1, da_long['t_out'] + 1, 0)

    # da_long["_idx"] = (da_long[["id"]]).apply(tuple)
    # da_long = da_long.set_index("_idx")
    
    if (type == 'd_vec_censor'):
        # Vectorized version of data 
        # Preparing the vectorized data
        db_vec = (
            db_long.sort_values(["id", "t_in"])
            .groupby("id")
            .aggregate(
                plan=("plan", "first"),
                t_in=("t_in", "min"),
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
                t_in=("t_in", "min"),
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
        d_vec = pd.concat([da_vec, db_vec], axis = 0)


    if type == 'd_long':
        return data
    if type == 'd_vec_nocensor':
        pass
        # return d_vec_all
    if type == 'd_vec_censor':
        return d_vec
    if type == 'd_vec_orig':
        d_vec_orig = (
            data.sort_values(['id', 't_in'])
            .groupby('id')
            .aggregate(
                plan_000=('plan_000', 'first'), #new_column = ('source_column', 'function')
                plan_100=('plan_100', 'first'),
                plan_010=('plan_010', 'first'), 
                plan_001=('plan_001', 'first'),
                plan_NA=('plan_NA', 'first'),
                t_in=('t_in', 'max'), # maybe not needed?
                t_out=('t_out', 'max'), # time of the event
                t_trt=('t_trt', 'max'),
                trt_obs=('A', 'max'),
                trt_obs_lag=('A_lag', 'max'),
                t_trt_lag=('t_trt_lag', 'max'), # look into why t_trt_lag is for the next row's t_out 
                event=('Y', 'last'),
                intercept=('intercept', 'first'),
                W=('W', 'first')
            ).reset_index()
        )

        return d_vec_orig
    if type == 'd_vec_c1':
        d_vec_c1 = (
            da_long.sort_values(['id', 't_in'])
            .groupby('id')
            .aggregate(
                trt_plan=('trt_clone', 'first'),
                plan_000=('plan_000', 'first'), #new_column = ('source_column', 'function')
                plan_100=('plan_100', 'first'),
                plan_010=('plan_010', 'first'), 
                plan_001=('plan_001', 'first'),
                plan_NA=('plan_NA', 'first'),
                t_in=('t_in', 'max'), # maybe not needed?
                t_in_cens=('t_in', 'max'),
                t_out=('t_out', 'max'), # time of the event
                t_trt=('t_trt', 'max'),
                trt_obs=('A', 'max'),
                trt_obs_lag=('A_lag', 'max'),
                t_trt_lag=('t_trt_lag', 'max'), # look into why t_trt_lag is for the next row's t_out 
                censor=('censor', 'max'),
                event=('Y', 'last'),
                intercept=('intercept', 'first'),
                W=('W', 'first')
            ).reset_index()
        )
        return d_vec_c1
    if type == 'd_vec_c0':
        d_vec_c0 = (
            db_long.sort_values(['id', 't_in'])
            .groupby('id')
            .aggregate(
                trt_plan=('trt_clone', 'first'),
                plan_000=('plan_000', 'first'), #new_column = ('source_column', 'function')
                plan_100=('plan_100', 'first'),
                plan_010=('plan_010', 'first'), 
                plan_001=('plan_001', 'first'),
                plan_NA=('plan_NA', 'first'),
                t_in=('t_in', 'max'), # maybe not needed?
                t_in_cens=('t_in', 'max'),
                t_out=('t_out', 'max'), # time of the event
                t_trt=('t_trt', 'max'),
                trt_obs=('A', 'max'),
                trt_obs_lag=('A_lag', 'max'),
                t_trt_lag=('t_trt_lag', 'max'), # look into why t_trt_lag is for the next row's t_out 
                censor=('censor', 'last'),
                event=('Y', 'last'),
                intercept=('intercept', 'first'),
                W=('W', 'first')
            ).reset_index()
        )
        return d_vec_c0
    if type == 'da_long': 
        return da_long
    if type == 'db_long':
        return db_long

