import os
import os.path as path
import sys

sys.path.insert(0, path.dirname(os.getcwd()))
import datetime as dt

import numpy as np
from IPython.display import Markdown, display
from modelling_pkg.config import *

CONFIG_PATH = "final_configurations/0_onboarding-2025-01"
# CONFIG_PATH = 'configurations'
EPS = 5


def map_final_outcome(o):
    if o == "Accepted":
        return 1
    elif o == "Rejected":
        return 0
    elif o in ["In Progress", np.nan] or np.isnan(o):
        return np.nan
    else:
        raise ValueError(f"Unknown final outcome: {o}")


def printmd(string):
    display(Markdown(string))


def split_trn_tst(data, country, bcn_cols, spans=None, verbose=True):
    if spans is None:
        spans = SPANS[country]
    train_start, train_end, test_start, test_end = (
        dt.datetime.strptime(spans[i], "%Y-%m-%d") for i in range(4)
    )

    # if bcn_cols is not None:
    bin_cols, cat_cols, num_cols = bcn_cols
    # cols = ['organization_id', 'accepted', 'discount_shp', 'revenue_bin'] + bin_cols + cat_cols + num_cols
    # else:
    #     cols = data.columns
    train = data[
        (data.created_date_off >= train_start) & (data.created_date_off < train_end)
    ]  # [cols]
    test = data[
        (data.created_date_off >= test_start) & (data.created_date_off < test_end)
    ]  # [cols]

    if verbose:
        print(spans)
        print(f"Train data shape: {train.shape}")
        print(f"Test data shape: {test.shape}")

    return train, test, spans


def calc_tick_step(m, div=6):
    bases = sum(([10 * 10**i, 25 * 10**i, 50 * 10**i] for i in range(0, 10)), [])[::-1]
    return next(filter(lambda b: b <= m // div, bases))


# %%time
# ### Separate per country
# ORG_ID_R = {v: k for k, v in ORGANIZATION_ID.items()}
# counts = data.groupby('organization_id').offer_id.count().sort_values(ascending=False).reset_index()
# counts.columns = ['organization_id', 'n_offers']
# counts.insert(1, 'country', counts.organization_id.replace(ORG_ID_R))

# counts.to_csv(DATADIR_RTP + '/countries/0_counts.csv', index=False)

# for o, df in data.groupby('organization_id'):
#     if o not in ORG_ID_R:
#         c = o.replace(' ', '_')
#     else:
#         c = ORG_ID_R[o]
#     print(c)
#     df.to_csv(DATADIR_RTP + f'/countries/{c}.csv')
