 #############################################################################
# DGP for CCW 
# JHW (2026-01-19)
# Details: for simulation purposes 
#############################################################################

##################################################
# Loading Packages
##################################################

import numpy as np
import pandas as pd

#############################################
# Functions to import 
#############################################

def inverse_logit(logodds): 
    lodds = np.asarray(logodds)
    return 1 / (1 + np.exp(-lodds))

# Use this function to generate the natural course
# Use this function to generate the counterfactual population where everyone is assigned to no treatment by grace period timepoint 
# This is actually the natural course, so only accepts plan=None
def dgp_grace(k, grace, nobs, timevarying, rngseed, plan=None):
    """
    k= max t_out (ie, 12 mo)
    grace= length of grace period inclusive (ie, 3 mo)
    nobs= total number of observations (ie, n # of older adults)
    timevarying= 1/0 (ie, no)
    rngseed= integer
    plan= None (observed)
    """

    rng = np.random.default_rng(
        seed=rngseed
    )  # constructs a new Generator with the default BitGenerator
    # pass
    datasets = []
    times = np.arange(start=1, stop=k)
    # * Baseline variables, t_in==0
    # w_value = rng.uniform(low=65, high=90, size=nobs)
    w_value = rng.binomial(n=1, p=0.3, size=nobs)
    # print(f"w_value = rng.binomial(n=1, p=0.3, size= f{nobs})")

    # Plan 1 or plan 0
    if (plan is None)|(plan == 1):
        a_value = rng.binomial(n=1, p=inverse_logit(-2 + 0.5 * w_value))
    elif plan == 0:
        a_value = np.zeros(shape=nobs, dtype=int)
    # Outcome variable at t_in==0
    y_value = rng.binomial(n=1, p=inverse_logit(-2 + 0.7 * a_value - 0.6 * w_value))
    # Dataframe of nobs rows for the t_in ==0, first time point
    d = pd.DataFrame()
    d["W"] = w_value  # baseline W at t_in==0
    d["A"] = a_value  # A at t_in==0 order matters st a_value is the same length of d.shape
    d["p_AmidW"] = inverse_logit(-2 + 0.5 * d["W"])  # P(A|W)
    d["Y"] = y_value  # Y at t_out==1 (t_out is created later)
    d["p_YmidAW"] = inverse_logit(-2 + 0.7 * d["A"] - 0.6 * d["W"])  # P(Y|A,W)
    d["id"] = d.index + 1
    d["t_in"] = 0
    ds = d.copy()
    datasets.append(ds)
    # Record whether the outcome or the trt already happened at t_in==0
    trt_already = a_value
    outcome_already = y_value

    # * Each subsequent t_in until k-1
    # Test just one timepoint for the logic of the compound statement in creating the variables
    # Pretend this is an indent to Incrementally test for each t in times
    for t in times:
        # Only baseline for now
        if timevarying == 0:
            pass  # already established w_value at t_in==0, so it w will carry over
        elif timevarying == 1:
            w_value = rng.binomial(
                n=1, p=inverse_logit(0.3 - 0.7 * a_value + 0.2 * w_value), size=nobs
            )

        # NOTE - DO NOT use plan == 1
        # if A already happened at t==0 or any previous timepoint for an id, then fill rest of the timepoints for that id with zeros.
        if (plan is None):
            # If A is already 1 in the prior timepoints, then put 0 for all rest of them
            a_value = np.where(
                trt_already == 1,
                0,
                rng.binomial(
                    n=1,
                    p=inverse_logit(-2 - (0.8 * a_value) - (0.5 * w_value) + (0.1 * t)),
                    size=nobs,
                ),
            )
        if (plan == 1):
            if t < grace: 
                # If A is already 1 in the prior timepoints, then put 0 for all rest of them
                a_value = np.where(
                    trt_already == 1,
                    0,
                    rng.binomial(
                        n=1,
                        p=inverse_logit(-2 - (0.8 * a_value) - (0.5 * w_value) + (0.1 * t)),
                        size=nobs,
                    ),
                )
            elif (t == grace):
                a_value = np.where(trt_already == 1, 0, np.ones(shape=nobs))
                                   
            elif t > grace: 
                a_value = np.where(
                trt_already == 1,
                0,
                rng.binomial(
                    n=1,
                    p=inverse_logit(-2 - (0.8 * a_value) - (0.5 * w_value) + (0.1 * t)),
                    size=nobs,
                ),
            )
        # Everyone assigned to plan A=000
        elif plan == 0:
            if t <= grace:  # strictly not Plan A
                a_value = np.zeros(shape=nobs)
            elif t > grace:
                # if A is already 1 in t_in>grace then rest are 0
                a_value = np.where(
                    trt_already == 1,
                    0,
                    rng.binomial(
                        n=1,
                        p=inverse_logit(-2 - (0.8 * a_value) - (0.5 * w_value) + (0.1 * t)),
                        size=nobs,
                    ),
                )

        # if Y already happened at t==0 or any previous timepoint for an id, they would have been removed from the ds in the previous t_in cycle
        y_value = rng.binomial(
            n=1,
            p=inverse_logit(-5 + (0.6 * a_value) + (0.7 * w_value) + (0.1 * t)),
            size=nobs,
        )

        # Fill in d from above
        ds = d.copy()  # rewrite d from the prior timepoint
        ds["t_in"] = t
        ds["W"] = w_value
        ds["A"] = a_value
        # ds["p_AmidW"] = inverse_logit(
        #     -2 - 0.8 * ds["A"] - 0.5 * ds["W"] + 0.1 * ds["t_in"]
        # )  # P(A|W)
        ds["Y"] = y_value
        # ds["p_YmidAW"] = inverse_logit(
        #     -5 + 0.6 * ds["W"] + 0.7 * ds["A"] + 0.1 * ds["t_in"]
        # )  # P(Y|A,W)
        ds = ds.loc[
            outcome_already == 0
        ].copy()  # now in this cycle if you already have the outcome, you are now not in the next t_in ds loop
        # Add timepoint t to list
        datasets.append(ds)

        # Update for the next t_in t value loop
        trt_already = np.where(
            trt_already == 1, 1, a_value
        )  # update on whether trt happened for this cycle of t_in
        outcome_already = np.where(
            outcome_already == 1, 1, y_value
        )  # update update on outcome_already for this cycle of t_out (list of length==nobs, it will update itself every t_in iteration for the Y value of t_out )

    # Finally: return d
    # Concat all t_in's
    d = pd.concat(datasets, ignore_index=True)
    d = d.sort_values(by=["id", "t_in"], ignore_index=True)  # sort for viewing
    # t_out until k
    d["t_out"] = d["t_in"] + 1
    d['id_counts'] = d.groupby('id')['id'].transform('count')
        # Identify ids in this batch that adhered (any A in 0..grace)
    # anyA = d[d["t_in"].between(0, grace)].groupby("id")["A"].sum()
    anyYgp = d[d["t_in"].between(0, grace-1)].groupby("id")["Y"].sum()
    anyAgp = d[d["t_in"].between(0, grace-1)].groupby("id")["A"].sum()
    anyAYgp = pd.concat([anyAgp, anyYgp], axis=1)
    ids_y_within_gp = anyAYgp[(anyAYgp["A"] == 0) & (anyAYgp["Y"] == 1)].index.to_numpy()
    # n_y_within_gp = ids_y_within_gp.size
    # Identify the specific plans that people's observed data are consistent with
    ids_100 = d[(d['t_in'] == 0 ) & (d['A'] == 1)]['id']
    ids_010 = d[(d['t_in'] == 1 ) & (d['A'] == 1)]['id']
    ids_001 = d[(d['t_in'] == 2 ) & (d['A'] == 1)]['id']
    strictly_000 = 1 - d.loc[(d['t_in'] <= grace) & (d['id_counts'] >= (grace + 1))].groupby('id')['A'].sum()
    ids_000 = strictly_000[strictly_000 == 1].index.tolist()

    d['plan_100'] = np.where(d['id'].isin(ids_100), 1, 0)
    d['plan_010'] = np.where(d['id'].isin(ids_010), 1, 0)
    d['plan_001'] = np.where(d['id'].isin(ids_001), 1, 0)
    d['plan_000'] = np.where(d['id'].isin(ids_000), 1, 0)
    d['plan_NA'] = np.where(d['id'].isin(ids_y_within_gp), 1, 0)

    d = d[["id", "t_in", "t_out", "W", "A", "Y", "id_counts", 'plan_000', 'plan_100', 'plan_010', 'plan_001', 'plan_NA']].copy()
    return d


# This function will create a population "everyone assigned to plan A" but will iteratively search select plan 001 after the first 'draw' 
# This is a different counterfactual than in the CCW currently where anyone censored for A=000 is upweighted to look like plan=001
# Only use for plan == 1, for plan == 0 this is the WRONG counterfactual
def dgp_grace_set_plan(k, grace, nobs_target, timevarying, plan, rngseed, max_rounds=200):
    """
    k=tot timepoints
    grace=length of grace period 
    nobs_target=the total nobs that you want 
    Iteratively draw batches from dgp_grace and keep only ids who adhere to Plan 1 (have any A==1 in t_in in 0..grace) or plan 0 (have no A in t_in in 0...grace). Builds pseudopop where everyone adheres to plan 1 or plan 0
    """
    rng_base = int(rngseed)
    remaining = nobs_target
    next_global_id = 1
    kept_frames = []
    round_idx = 0
    # Repeat until no remaining ids, but stop if max rounds reached 
    if round_idx == 0:
        print("first round:", round_idx)
        round_idx += 1
        seed = rng_base * 5 + round_idx + 1
        # Simulate batch of observed population
        batch_df = dgp_grace(k=k, grace=grace, nobs=remaining, timevarying=timevarying, rngseed=seed, plan=None)
        # Identify ids in this batch that adhered (any A in 0..grace)
        anyA = batch_df[batch_df["t_in"].between(0, grace)].groupby("id")["A"].sum()
        any_000 = batch_df[(batch_df['plan_000'] == 1) & batch_df['t_in'].between(0, grace)].groupby('id')['A'].sum()

        anyYgp = batch_df[batch_df["t_in"].between(0, grace-1)].groupby("id")["Y"].sum()
        anyAgp = batch_df[batch_df["t_in"].between(0, grace-1)].groupby("id")["A"].sum()
        anyAYgp = pd.concat([anyAgp, anyYgp], axis=1)
        ids_y_within_gp = anyAYgp[(anyAYgp["A"] == 0) & (anyAYgp["Y"] == 1)].index.to_numpy()
        n_y_within_gp = ids_y_within_gp.size

        if plan == 1:
            adhered_ids = anyA[anyA >= 1].index.to_numpy()
        elif plan == 0: 
            adhered_ids = any_000[any_000 == 0].index.to_numpy()
        n_adhered = adhered_ids.size
        keeping_ids = np.concatenate((adhered_ids, ids_y_within_gp), axis=0)
        keeping_ids = np.unique(keeping_ids)
        n_keeping = keeping_ids.size

        # Tracking numbers 
        print(f"round {round_idx}: simulated {remaining}, adhered found {n_adhered}, n_y_within_gp found {n_y_within_gp}, kept = {n_adhered + n_y_within_gp} or {n_keeping}")

        if n_keeping == 0:
            # No one adhered this round 
            print("outer loop no adherers found this round — stopping early")

        # extract rows for adhered ids and remap their ids
        sel_rows = batch_df[batch_df["id"].isin(keeping_ids)].copy()
        # create mapping from per-batch id -> global id
        unique_old_ids = np.unique(sel_rows["id"].to_numpy())
        id_map = {int(old): next_global_id + idx for idx, old in enumerate(unique_old_ids)}
        sel_rows["id"] = sel_rows["id"].map(id_map)
        next_global_id += unique_old_ids.size

        # append selected adhered rows to kept_frames
        kept_frames.append(sel_rows)

        # update remaining count to simulate next round on those who did not adhere in this batch
        nonadhered_count = remaining - n_keeping
        print(f"nonadhered {nonadhered_count}")
        remaining = nonadhered_count
        print(f"remaining {remaining}")
    while (remaining > 0) and (round_idx < max_rounds):
        round_idx += 1
        seed = rng_base * 5 + round_idx + 1
        # Simulate batch of observed population
        batch_df = dgp_grace(k=k, grace=grace, nobs=remaining, timevarying=timevarying, rngseed=seed, plan=None)
        # Identify ids in this batch that adhered (any A at t_in==grace)
        anyA_001 = batch_df[batch_df["t_in"]==grace].groupby("id")["A"].sum()
        any_000 = batch_df[(batch_df['plan_000'] == 1) & batch_df['t_in'].between(0, grace)].groupby('id')['A'].sum()
        
        anyYgp = batch_df[batch_df["t_in"].between(0, grace-1)].groupby("id")["Y"].sum()
        anyAgp = batch_df[batch_df["t_in"].between(0, grace-1)].groupby("id")["A"].sum()
        anyAYgp = pd.concat([anyAgp, anyYgp], axis=1)
        ids_y_within_gp = anyAYgp[(anyAYgp["A"] == 0) & (anyAYgp["Y"] == 1)].index.to_numpy()
        n_y_within_gp = ids_y_within_gp.size

        if plan == 1:
            adhered_ids = anyA_001[anyA_001 >= 1].index.to_numpy()
            n_adhered = adhered_ids.size
            keeping_ids = adhered_ids
            keeping_ids = np.unique(keeping_ids)
            n_keeping = keeping_ids.size
        elif plan == 0: 
            adhered_ids = any_000[any_000 == 0].index.to_numpy()
            n_adhered = adhered_ids.size
            keeping_ids = np.concatenate((adhered_ids, ids_y_within_gp), axis=0)
            keeping_ids = np.unique(keeping_ids)
            n_keeping = keeping_ids.size

        # if n_adhered == 0:
        #     # No one adhered this round, do not take this out do not want to cut off early
        #     print("no adherers found this round — stopping early")
        #     break
        # extract rows for adhered ids and remap their ids
        sel_rows = batch_df[batch_df["id"].isin(keeping_ids)].copy()
        # create mapping from per-batch id -> global id
        unique_old_ids = np.unique(sel_rows["id"].to_numpy())
        id_map = {int(old): next_global_id + idx for idx, old in enumerate(unique_old_ids)}
        sel_rows["id"] = sel_rows["id"].map(id_map)
        next_global_id += unique_old_ids.size

        # append selected adhered rows to kept_frames
        kept_frames.append(sel_rows)

        # update remaining count to simulate next round on those who did not adhere in this batch
        nonadhered_count = remaining - n_keeping
        print(f"nonadhered {nonadhered_count}")
        remaining = nonadhered_count
        print(f"remaining {remaining}")
        # Tracking numbers 
        print(f"round {round_idx}: simulated {remaining}, adhered found {n_adhered} or {n_keeping}")

    # concat kept frames and final formatting 
    # Return a empty dataframe even if there was noone kept and meeting the criteria
    if len(kept_frames) == 0:
        return pd.DataFrame(columns=["id", "t_in", "t_out", "W", "A", "Y"])
    # Else output all of the data 
    out = pd.concat(kept_frames, ignore_index=True)
    out = out.sort_values(by=["id", "t_in"], ignore_index=True)
    # ensure t_out exists; if not, create
    if "t_out" not in out.columns:
        out["t_out"] = out["t_in"] + 1
    # keep only core columns
    cols = [c for c in ["id", "t_in", "t_out", "W", "A", "Y","id_counts", 'plan_000', 'plan_100', 'plan_010', 'plan_001', 'plan_NA'] if c in out.columns]
    return out[cols].copy()

# Alternative plan A truth simulation 


# Testing whether this works
def dgp_grace_planA4(k, grace, nobs_target, timevarying, rngseed):
    """
    Iteratively draw batches from dgp_grace and keep only ids who adhere to Plan A
    (have any A==1 in t_in in 0..grace). Builds pseudopop where everyone adheres to plan A
    """
    rng_base = int(rngseed)
    remaining = nobs_target
    next_global_id = 1
    kept_frames = []
    round_idx = 0
    # Repeat until no remaining ids, but stop if max rounds reached 

    print("first round:", round_idx)
    round_idx += 1
    seed = rng_base * 5 + round_idx + 1
    # Simulate batch of observed population
    batch_df = dgp_grace(k=k, grace=grace, nobs=remaining, timevarying=timevarying, rngseed=seed, plan=None)
    # Identify ids in this batch that adhered (any A in 0..grace), mm
    anyA = batch_df[batch_df["t_in"].between(0, grace)].groupby("id")["A"].sum()
    anyYgp = batch_df[batch_df["t_in"].between(0, grace-1)].groupby("id")["Y"].sum()
    anyAgp = batch_df[batch_df["t_in"].between(0, grace-1)].groupby("id")["A"].sum()
    anyAYgp = pd.concat([anyAgp, anyYgp], axis=1)
    ids_y_within_gp = anyAYgp[(anyAYgp["A"] == 0) & (anyAYgp["Y"] == 1)].index.to_numpy()
    n_y_within_gp = ids_y_within_gp.size
    adhered_ids = anyA[anyA >= 1].index.to_numpy()
    n_adhered = adhered_ids.size
    keeping_ids = np.concatenate((adhered_ids, ids_y_within_gp), axis=0)
    keeping_ids = np.unique(keeping_ids)
    n_keeping = keeping_ids.size

    # Tracking numbers 
    print(f"round {round_idx}: simulated {remaining}, adhered found {n_adhered}, n_y_within_gp found {n_y_within_gp}, kept = {n_adhered + n_y_within_gp} or {n_keeping}")

    if n_keeping == 0:
        # No one adhered this round 
        print("outer loop no adherers found this round — stopping early")

    # extract rows for adhered ids and remap their ids
    sel_rows = batch_df[batch_df["id"].isin(keeping_ids)].copy()
    # create mapping from per-batch id -> global id
    unique_old_ids = np.unique(sel_rows["id"].to_numpy())
    id_map = {int(old): next_global_id + idx for idx, old in enumerate(unique_old_ids)}
    sel_rows["id"] = sel_rows["id"].map(id_map)
    next_global_id += unique_old_ids.size

    # append selected adhered rows to kept_frames
    kept_frames.append(sel_rows)

    # update remaining count to simulate next round on those who did not adhere in this batch
    nonadhered_count = remaining - n_keeping
    print(f"nonadhered {nonadhered_count}")
    remaining = nonadhered_count
    print(f"remaining {remaining}")


    print("now calculating the remaining")
    rng = np.random.default_rng(
        seed=rngseed
    )

    datasets = []
    times = np.arange(start=1, stop=k)
    w_value = rng.binomial(n=1, p=0.3, size=remaining)
    a_value = np.zeros(shape=remaining, dtype=int)
    # Outcome variable at t_in==0
    # y_value = rng.binomial(n=1, p=inverse_logit(-2 + 0.7 * a_value - 0.6 * w_value))
    y_value = np.zeros(shape=remaining, dtype=int)
    d = pd.DataFrame()
    d["W"] = w_value  # baseline W at t_in==0
    d["A"] = a_value  # A at t_in==0 order matters st a_value is the same length of d.shape
    d["p_AmidW"] = inverse_logit(-2 + 0.5 * d["W"])  # P(A|W)
    d["Y"] = y_value  # Y at t_out==1 (t_out is created later)
    d["p_YmidAW"] = inverse_logit(-2 + 0.7 * d["A"] - 0.6 * d["W"])  # P(Y|A,W)
    d["id"] = d.index + 1 + n_keeping
    d["t_in"] = 0
    ds = d.copy()
    datasets.append(ds)
    # Record whether the outcome already happened
    trt_already = a_value
    outcome_already = y_value

    # * Each subsequent t_in until k-1
    # Test just one timepoint for the logic of the compound statement in creating the variables
    # Pretend this is an indent to Incrementally test for each t in times
    for t in times:
        # Only baseline for now
        if timevarying == 0: 
            pass  # already established w_value at t_in==0, so it w will carry over
        elif timevarying == 1:
            w_value = rng.binomial(
                n=1, p=inverse_logit(0.3 - 0.7 * a_value + 0.2 * w_value), size=remaining
            )
        if t < grace:  # strictly not Plan A
            a_value = np.zeros(shape=remaining)
            y_value = np.zeros(shape=remaining)
            # y_value = rng.binomial(
            # n=1,
            # p=inverse_logit(-5 + 0.6 * a_value + 0.7 * w_value + 0.1 * t),
            # size=remaining        )
        elif t == grace:
            a_value = np.ones(shape=remaining)
            y_value = rng.binomial(
            n=1,
            p=inverse_logit(-5 + 0.6 * a_value + 0.7 * w_value + 0.1 * t),
            size=remaining        )
        elif t > grace:
            a_value = np.zeros(shape=remaining)
            y_value = rng.binomial(
            n=1,
            p=inverse_logit(-5 + 0.6 * a_value + 0.7 * w_value + 0.1 * t),
            size=remaining        )

        # if Y already happened at t==0 or any previous timepoint for an id, they would have been removed from the ds in the previous t_in cycle
        # Fill in d from above
        ds = d.copy()  # rewrite d from the prior timepoint
        ds["t_in"] = t
        ds["W"] = w_value
        ds["A"] = a_value
        ds["p_AmidW"] = inverse_logit(
            -2 - 0.8 * ds["A"] - 0.5 * ds["W"] + 0.1 * ds["t_in"]
        )  # P(A|W)
        ds["Y"] = y_value
        ds["p_YmidAW"] = inverse_logit(
            -5 + 0.6 * ds["W"] + 0.7 * ds["A"] + 0.1 * ds["t_in"]
        )  # P(Y|A,W)
        ds = ds.loc[
            outcome_already == 0
        ].copy()  # now in this cycle if you already have the outcome, you are now not in the next t_in ds loop
        # Add timepoint t to list
        datasets.append(ds)

        # Update for the next t_in t value loop
        trt_already = np.where(
            trt_already == 1, 1, a_value
        )  # update on whether trt happened for this cycle of t_in
        outcome_already = np.where(
            outcome_already == 1, 1, y_value
        )  # update update on outcome_already for this cycle of t_out (list of length==nobs, it will update itself every t_in iteration for the Y value of t_out )

    # Finally: return d
    # Concat all t_in's
    d = pd.concat(datasets, ignore_index=True)
    d = d.sort_values(by=["id", "t_in"], ignore_index=True)  # sort for viewing
    # t_out until k
    d["t_out"] = d["t_in"] + 1
    d = d[["id", "t_in", "t_out", "W", "A", "p_AmidW", "Y", "p_YmidAW"]].copy()
    out = pd.concat(kept_frames, ignore_index=True)
    out = out.sort_values(by=["id", "t_in"], ignore_index=True)
    # ensure t_out exists; if not, create
    out["t_out"] = out["t_in"] + 1
    # keep only core columns
    cols = [c for c in ["id", "t_in", "t_out", "W", "A", "Y"] if c in out.columns]
    d = pd.concat([out[cols].copy(), d], ignore_index=True)
    return d

