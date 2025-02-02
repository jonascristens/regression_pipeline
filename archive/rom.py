import os
import os.path as path
import sys

sys.path.insert(0, path.dirname(os.getcwd()))
import dataclasses
import json
import pickle as pkl
import random
import warnings
from abc import ABC
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import pandas as pd
from augmenting import augment_data
from modelling_pkg.config import *
from optuna.samplers import TPESampler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                   OneHotEncoder)
from utils import printmd, split_trn_tst
from xgboost import XGBClassifier

optuna.logging.set_verbosity(optuna.logging.DEBUG)

from optuna.exceptions import ExperimentalWarning
from tqdm.notebook import tqdm

warnings.filterwarnings("ignore", category=ExperimentalWarning)

random.seed(0)

# _, ORGANIZATION_ID = get_organizations('../'+DATADIR)

TARGET = "accepted"

successful_trials = [
    {
        "learning_rate": 0.01087080185549562,
        "max_depth": 4,
        "n_estimators": 729,
        "gamma": 0.01123506477003016,
    },
    {
        "learning_rate": 0.02230934569621654,
        "max_depth": 6,
        "n_estimators": 721,
        "gamma": 0.05498659253091755,
    },
    # {'learning_rate': 0.04718211852154147, 'max_depth': 4, 'n_estimators': 490, 'gamma': 0.01407174331936138},
    # {'learning_rate': 0.05906369914893646, 'max_depth': 3, 'n_estimators': 670, 'gamma': 0.31449066951230936},
    # {'learning_rate': 0.01834364173054051, 'max_depth': 5, 'n_estimators': 288, 'gamma': 0.02199848967394945},
    # {'learning_rate': 0.02203157055707442, 'max_depth': 6, 'n_estimators': 913, 'gamma': 0.03910909162693285},
]


def pandarizer(x):
    return pd.DataFrame(
        x, columns=["discount"] + [str(i) for i in range(x.shape[1] - 1)]
    )


@dataclass
class ROMConfig:
    name: str = None
    estimator: str = "xgbc"
    augm_variant: Literal["orig", "basic", "1sided", "shift", "shift-uni"] = "basic"
    gamma: int = 6
    cc_adjustment: int = (
        0  # color-coding adjusment - increase optimal discount by this amount
    )
    sample: int = None
    augm_basic_params: tuple = (8, 26)
    augm_basic_weight: float = 1
    augm_basic_weight_asym: bool = True
    xgbc__initial_params: bool = True
    n_trials: int = None
    clip_min: int = 0
    timeout_minutes: int = 40
    optuna__seed: int = 0
    random_state: int = 0

    @classmethod
    def from_json_file(cls, path):
        if path[-4:] != "json":
            path = f"{path}/cfg.json"
        return cls(**json.load(open(path, "r")))

    # @classmethod
    # def from_json(cls, obj):
    #     return cls(**obj)


@dataclass
class ROMTrainerConfig:
    experiment: str
    name: str
    estimator: str = "xgbc"
    augm_variant: Literal["orig", "basic", "1sided", "shift", "shift-uni"] = "basic"
    gammas: tuple = (
        4,
        6,
        8,
    )  # determines width of recommended interval (opt_rev - gamma%)
    sample: int = None
    n_trials: int = 5
    timeout_minutes: int = None
    augm_basic_params: tuple = (10, 20)
    augm_basic_weight: float = 1
    augm_basic_weight_asym: bool = False
    augm__rshift_sigma: int = 2
    xgbc__initial_params: bool = True
    optuna__seed: int = 0
    random_state: int = 0
    clip_min: int = 0
    spans: tuple = None

    @classmethod
    def from_json_file(cls, path):
        if path[-4:] != "json":
            path = f"{path}/cfg.json"
        d = json.load(open(path, "r"))
        d.pop("cfgs_dir", None)
        return cls(**d)


class BaseROM(ABC):
    """
    Abstract class with several ROM methods common for ROM and ROMTrainer.
    """

    _n_ast = 30

    def __init__(
        self,
        bin_cols=bin_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        verbose=False,
    ) -> None:
        self.bin_cols = bin_cols
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self._temp_prep_pipe = self._get_prep_pipe(bin_cols, cat_cols, num_cols)
        self.verbose = verbose

    def _print(self, *s, md=False):
        f = printmd if md else print
        if self.verbose:
            f(*s)

    @staticmethod
    def _use_timeout(cfg):
        if cfg.n_trials is not None:
            return False
        elif cfg.timeout_minutes is not None:
            return True
        else:
            raise ValueError(
                "Exactly one of `config.n_trials`, `config.timeout_minutes` must be specified."
            )

    # TODO - low priority - stopping criteria hyperparameter tuning
    @staticmethod
    def _get_stopping_criteria(cfg):
        d = {}
        if cfg.timeout_minutes is not None:
            d["timeout"] = cfg.timeout_minutes * 60
        if cfg.n_trials is not None:
            d["n_trials"] = cfg.n_trials

        if len(d) == 0:
            raise ValueError(
                "At least one of `config.n_trials`, `config.timeout_minutes` must be specified."
            )
        return d

    # DONE
    def _clean_dataset(self, data, country=None, clip_min=0, clip_max=99):
        cnd = (
            (data["discount_shp"] <= clip_max)
            & (data["discount_shp"] >= clip_min)
            & data.accepted.notna()
        )
        # & data.revenue_bin.notna()  # todo remove because removed in production

        # data = data[(data['discount_shp'] <= clip_max)
        #             & (data['discount_shp'] >= clip_min)
        # # & data.revenue_bin.notna()  # todo remove because removed in production
        #             # & data.revenue_bin.notna()
        #             & data.accepted.notna()].drop_duplicates()

        if country is not None:
            cnd &= data.organization_id == ORGANIZATION_ID[country]

        data = data[cnd].drop_duplicates()

        bcn_cols = []
        for cols0 in [self.bin_cols, self.cat_cols, self.num_cols]:
            missing = list(filter(lambda col: col not in data.columns, cols0))
            if missing:
                warnings.warn(f"Missing columns: {missing}")
            bcn_cols.append(list(filter(lambda col: col in data.columns, cols0)))

        return data, bcn_cols

    # DONE
    @staticmethod
    def _split_xy(*data):
        ret = []
        for df in data:
            x = df.drop(columns=TARGET)
            y = df[TARGET]
            ret.extend([x, y])
        return ret

    # DONE
    def _get_prep_pipe(self, bin_cols, cat_cols, num_cols):
        categorical_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore", drop=None, sparse_output=False
                    ),
                ),
            ]
        )

        numerical_pipe = Pipeline(
            [("imputer", SimpleImputer(strategy="median")), ("scaler", MinMaxScaler())]
        )

        binary_pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent"))  # todo try mean
            ]
        )

        # IMPORTANT: 'discount_shp' column must be THE FIRST column in the training dataframe due to the monotonicity constraint for XGBC!
        # keep in mind in case of modifying the ColumnTransformer
        col_pipes = [
            ("dsc", "passthrough", ["discount_shp"]),
            ("bin", binary_pipe, bin_cols),
            ("cat", categorical_pipe, cat_cols),
            ("num", numerical_pipe, num_cols),
        ]
        return ColumnTransformer(col_pipes)

    # DONE
    def _get_objective(self, x, y, bcn_cols, weights, random_state=0):
        #         TODO use get_params function to allow different grids
        # def _get_objective(data, prep_pipe, estimator, get_params, variant):
        # self._print(self.prep_pipe.fit_transform(self.trnx).shape, md=False)
        def objective(trial):
            scoring = "neg_log_loss"  # if npo_variant else 'neg_mean_squared_error'

            if self._cfg.estimator == "xgbc":
                params = {
                    "learning_rate": trial.suggest_float(
                        "learning_rate", 0.01, 0.1, log=True
                    ),
                    "max_depth": trial.suggest_int("max_depth", 3, 6),
                    "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                    "gamma": trial.suggest_float("gamma", 0.01, 1.0, log=True),
                }

                pipeline = Pipeline(
                    [
                        ("prep", self._get_prep_pipe(*bcn_cols)),
                        ("pandarizer", FunctionTransformer(pandarizer)),
                        (
                            "clf",
                            XGBClassifier(
                                **params,
                                objective="binary:logistic",
                                monotone_constraints={"discount": 1},
                                random_state=random_state,
                            ),
                        ),
                    ]
                )
                # pipeline.set_output(transform='pandas')
            else:
                raise NotImplementedError()
            if weights:
                ret = -cross_val_score(
                    pipeline,
                    x,
                    y,
                    cv=5,
                    scoring=scoring,
                    params={"clf__sample_weight": weights},
                ).mean()
            else:
                ret = -cross_val_score(pipeline, x, y, cv=5, scoring=scoring).mean()
            return ret

        return objective

    # DONE
    def _train_model(self, x, y, n_features, cfg, bcn_cols, weights):
        if not self.verbose:
            optuna.logging.set_verbosity(optuna.logging.WARNING)
        objective = self._get_objective(x, y, bcn_cols, weights, cfg.random_state)
        sampler = TPESampler(seed=cfg.optuna__seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        if cfg.xgbc__initial_params:
            for trial in successful_trials:
                study.enqueue_trial(trial)

        study.optimize(
            objective,
            show_progress_bar=self.verbose,
            **self._get_stopping_criteria(cfg),
        )
        best_params = study.best_params

        model = Pipeline(
            [
                ("prep", self._get_prep_pipe(*bcn_cols)),
                (
                    "clf",
                    XGBClassifier(
                        **best_params,
                        objective="binary:logistic",
                        monotone_constraints=tuple(
                            [1] + [0] * (n_features - 1)
                        ),  # the first column (discount_shp) has a positive monotonicity constraint
                    ),
                ),
            ]
        )
        if weights:
            model.fit(x, y, clf__sample_weight=weights)
        else:
            model.fit(x, y)
        return model, study

    # TODO - medium priority - simulate expected revenue
    @staticmethod
    def _predict_exp_revenue(offer, model):
        offer_df = pd.DataFrame([offer] * 100)
        offer_df["discount_shp"] = range(100)
        offer_df["rom_pa"] = model.predict_proba(offer_df)[:, 1]
        offer_df["rom_exp_rev"] = (
            1 - offer_df.discount_shp / 100
        ) * offer_df.rom_pa  # * np.exp(offer_df.log_published_revenue)
        maxr = offer_df.iloc[offer_df.rom_exp_rev.argmax()]
        return offer_df, maxr


class ROM(BaseROM):
    """
    Revenue Optimization Model
    """

    def __init__(
        self,
        config: ROMConfig,
        bin_cols=bin_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        verbose=False,
    ) -> None:
        super().__init__(bin_cols, cat_cols, num_cols, verbose)
        self._cfg = config

    def train(self, data):
        cfg = self._cfg
        if cfg is None:
            raise AttributeError("config is missing!")

        if cfg.sample:
            data = data.sample(cfg.sample, random_state=cfg.random_state)
        data, bcn_cols = self._clean_dataset(data, clip_min=cfg.clip_min)
        s = data.shape
        # TODO - high priority - investigate augmentation process
        data, weights = augment_data(data, cfg)
        self._print("before/after aug", s, data.shape, md=False)
        x, y = self.trn_data = self._split_xy(data)

        n_features = self._temp_prep_pipe.fit(x).transform(x.sample(1)).shape[1]
        self._model, self._study = self._train_model(
            x, y, n_features, cfg, bcn_cols, weights
        )

    # TODO - medium priority - implement recommended prices
    def recommend(self, data, progress_bar=False):
        recommendations = []
        iter_ = tqdm(range(len(data))) if progress_bar else range(len(data))
        for i in iter_:
            offer = data.iloc[i]
            offer_df, maxr = self._predict_exp_revenue(offer, self._model)

            revenue_at_least = (1 - self._cfg.gamma / 100) * maxr.rom_exp_rev
            close_rev = offer_df[offer_df.rom_exp_rev > revenue_at_least].discount_shp
            dmin, dmax, dopt = close_rev.min(), close_rev.max(), maxr.discount_shp
            dadj = dopt + (
                0 if self._cfg.cc_adjustment is None else self._cfg.cc_adjustment
            )
            dadj = min(97, dadj)
            recommendations.append(
                pd.Series(
                    dict(
                        discount_low=dmin,
                        discount_high=dmax,
                        discount_opt=dadj,
                        discount_opt_wo_adj=dopt,
                    )
                )
            )

        return pd.DataFrame(recommendations, index=data.index)

    def predict_acc_prob_and_exp_rev(self, offer):
        return self._predict_exp_revenue(offer, self._model)

    def get_config_as_dict(self):
        return dataclasses.asdict(self._cfg)

    def get_pipeline(self):
        return self._model

    def get_dumpable(self):
        return (self.get_config_as_dict(), self._model)

    @classmethod
    def from_dumpable(
        cls,
        dumpable,
        bin_cols=bin_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        verbose=False,
    ):
        cfg, model = dumpable
        cfg = ROMConfig(**cfg)
        rom = cls(cfg, bin_cols, cat_cols, num_cols, verbose)
        rom._model = model
        return rom

    @classmethod
    def from_config_and_pipe(
        cls,
        config,
        pipeline,
        bin_cols=bin_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        verbose=False,
    ):
        if isinstance(config, dict):
            config = ROMConfig(**config)
        rom = cls(config, bin_cols, cat_cols, num_cols, verbose)
        rom._model = pipeline
        return rom


class ROMTrainer(BaseROM):
    """
    ROMTrainer is a class that facilitates training of multiple ROM models, hyperparameter optimization, making recommendations and saving results.
    """

    def __init__(
        self,
        bin_cols=bin_cols,
        cat_cols=cat_cols,
        num_cols=num_cols,
        verbose=True,
    ) -> None:
        super().__init__(bin_cols, cat_cols, num_cols, verbose)
        # self.models = defaultdict(dict)
        self.models = {}
        # self.studies = defaultdict(dict)
        self.studies = {}
        # self.data = defaultdict(dict)
        self.data = {}
        self.clf_eval_dfs = {}
        self.rom_eval_dfs = {}
        self.rom_example_dfs = {}
        self.eval_dfs_samples = {}

    def train_models(self, data_dict, countries, configs, save_immediately_to):
        n_countries = len(countries)
        for i, c in enumerate(countries):
            cfgs = configs[c] if c in configs else configs
            n_configs = len(cfgs)
            for j, cfg in enumerate(cfgs):
                # self.verify_mdd_preds_exist(c, cfg)
                self._cfg = cfg
                self._print(
                    f"\n\n**{'^' * self._n_ast} {c} ({i + 1}/{n_countries}) - {cfg.name} ({j + 1}/{n_configs}) {'^' * self._n_ast}**\n\n",
                    md=True,
                )
                dataset = data_dict[c]
                if cfg.sample:
                    dataset = dataset.sample(cfg.sample, random_state=cfg.random_state)
                dataset, bcn_cols = self._clean_dataset(
                    dataset, c, clip_min=cfg.clip_min
                )
                trn, tst, cfg.spans = split_trn_tst(
                    dataset, c, bcn_cols, spans=cfg.spans
                )
                s = trn.shape
                trn, weights = augment_data(trn, cfg)
                self._print("before/after aug", s, trn.shape, md=False)
                # (self.trnx, self.trny, self.tstx, self.tsty)
                # TODO implement the following line
                tst_months = cfg.spans[2][:7], cfg.spans[3][:7]
                trnx, trny, tstx, tsty = self.data[c, cfg.name, tst_months] = (
                    self._split_xy(trn, tst)
                )

                n_features = (
                    self._temp_prep_pipe.fit(trnx).transform(trnx.sample(1)).shape[1]
                )
                (
                    self.models[c, cfg.name, tst_months],
                    self.studies[c, cfg.name, tst_months],
                ) = self._train_model(trnx, trny, n_features, cfg, bcn_cols, weights)

                if save_immediately_to is not None:
                    self.save_results(c, cfg, save_immediately_to, spans=cfg.spans)

    def verify_mdd_preds_exist(self, country, cfg):
        tst_months = tst_start, tst_end = cfg.spans[2][:7], cfg.spans[3][:7]
        mdd_fn = f"../{DATADIR}/mdd_preds/{country}_{tst_start}_{tst_end}.csv"
        if not path.isfile(mdd_fn):
            raise Exception(
                f'MDD predictions for {country} not found at "../{DATADIR}/mdd_preds/{country}_{tst_start}_{tst_end}.csv"!'
            )

    # DONE
    def save_results(self, country, cfg, results_dir, spans):
        if spans is None:
            spans = SPANS[country]
        tst_months = tst_start, tst_end = cfg.spans[2][:7], cfg.spans[3][:7]
        res_path = (
            f"{results_dir}/{country}~{tst_start}~{tst_end}/{cfg.experiment}/{cfg.name}"
        )
        self._print(f"Saving results to '{res_path}'", md=False)
        Path(res_path).mkdir(parents=True, exist_ok=True)

        clf_results = self.get_clf_eval_df(country, cfg.name, tst_months)
        rom_results, eval_dfs = self.get_rom_eval_df(
            country, cfg.name, tst_months, gammas=cfg.gammas
        )

        clf_results.to_csv(f"{res_path}/clf.csv", index=True)
        rom_results.to_csv(f"{res_path}/rom.csv", index=True)
        pkl.dump(eval_dfs, open(f"{res_path}/eval.pkl", "wb"))
        json.dump(dataclasses.asdict(cfg), open(f"{res_path}/cfg.json", "w"))

    # TODO - high priority - inference pipeline
    def get_clf_eval_df(self, country, cfg_name, tst_months):
        if (country, cfg_name, tst_months) not in self.clf_eval_dfs:
            self._print(country, md=False)
            trnx, trny, tstx, tsty = self.data[country, cfg_name, tst_months]
            model = self.models[country, cfg_name, tst_months]

            tstx = tstx.copy()
            tstx["pred"] = model.predict_proba(tstx)[:, 1]
            tstx["pred_b"] = model.predict(tstx)
            tstx["accepted"] = tsty.astype(int)
            self.clf_eval_dfs[country, cfg_name, tst_months] = tstx

        return self.clf_eval_dfs[country, cfg_name, tst_months].copy()

    # TODO - to be investigated what the goal if of this function
    def get_rom_eval_df(self, country, cfg_name, tst_months, gammas=[2, 4, 6, 8, 10]):
        if (country, cfg_name, tst_months) not in self.rom_eval_dfs:
            tstx = self.get_clf_eval_df(country, cfg_name, tst_months).rename(
                columns={"discount_shp": "real_discount"}
            )
            model = self.models[country, cfg_name, tst_months]

            if country not in self.eval_dfs_samples:
                # select samples for which eval_dfs will be generated
                # they will be the same for each variant for a particular country
                random.seed(0)
                self.eval_dfs_samples[country] = set(
                    random.sample(range(len(tstx)), min(200, len(tstx)))
                )

            example_dfs = []
            recommendations = []
            # todo parallelize
            # todo try to speed up
            for i in tqdm(range(len(tstx))):
                offer = tstx.iloc[i]
                offer_df, maxr = self._predict_exp_revenue(offer, model)
                maxr = maxr.rename({"discount_shp": "rom_discount"})[
                    ["rom_discount", "rom_pa", "rom_exp_rev"]
                ]

                discount_intervals = {}
                for gamma in gammas:
                    atleast = (1 - gamma / 100) * maxr.rom_exp_rev
                    close_rev = offer_df[offer_df.rom_exp_rev > atleast].discount_shp
                    dmin, dmax, dopt = (
                        close_rev.min(),
                        close_rev.max(),
                        maxr.rom_discount,
                    )
                    dmid = (dmax + dmin) / 2
                    (
                        maxr[f"rom_discount_{gamma}_low"],
                        maxr[f"rom_discount_{gamma}_mid"],
                        maxr[f"rom_discount_{gamma}_high"],
                    ) = dmin, dmid, dmax
                    discount_intervals[gamma] = dmin, dmax, dopt
                recommendations.append(maxr)
                if i in self.eval_dfs_samples[country]:
                    example_dfs.append((discount_intervals, offer_df))

            tst_start, tst_end = tst_months
            # mdd_fn = f'../{DATADIR}/mdd_preds/{country}_{tst_start}_{tst_end}.csv'
            # if path.isfile(mdd_fn):
            #     mdd = pd.read_csv(mdd_fn, index_col=0)
            # else:
            #     raise Exception(f'MDD predictions for {country} not found at "../{DATADIR}/mdd_preds/{country}_{tst_start}_{tst_end}.csv"!')

            rec_df = pd.DataFrame(recommendations)
            res = pd.concat(
                [tstx[["real_discount", "log_published_revenue", "accepted"]], rec_df],
                axis=1,
            )
            res.insert(1, "pub_rev", np.exp(res.log_published_revenue))
            res = res.drop(columns="log_published_revenue")
            res.insert(
                3,
                "real_rev",
                (1 - res.real_discount / 100) * res.pub_rev * res.accepted,
            )
            res["rom_diff"] = res["rom_discount"] - res["real_discount"]

            display(res)
            # display(mdd)
            # res = res.join(mdd).assign(mdd_diff=mdd.mdd_discount-res.real_discount)
            self.rom_eval_dfs[country, cfg_name, tst_months] = res
            self.rom_example_dfs[country, cfg_name, tst_months] = example_dfs

        return self.rom_eval_dfs[
            country, cfg_name, tst_months
        ].copy(), self.rom_example_dfs[country, cfg_name, tst_months].copy()
