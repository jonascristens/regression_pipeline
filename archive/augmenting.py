from random import uniform

import numpy as np
import pandas as pd


def augm_basic(cfg):
    a, b = cfg.augm_basic_params
    w = cfg.augm_basic_weight

    def augm(array, r, weights=[]):
        if r.accepted == 1:
            r2 = r.copy()
            r2.discount_shp -= b
            r2.accepted = 0
            array.append(r2)

            r2 = r.copy()
            r2.discount_shp += a
            array.append(r2)
            weights.extend([1 if cfg.augm_basic_weight_asym else w, 1])
        if r.accepted == 0:
            r2 = r.copy()
            r2.discount_shp -= a
            array.append(r2)

            r2 = r.copy()
            r2.discount_shp += b
            r2.accepted = 1
            array.append(r2)
            weights.extend([1, w])

    return augm


def augm_basic_rshift(cfg):
    a, b = cfg.augm_basic_params

    def get_random_shift():
        return cfg.augm__rshift_sigma * np.random.randn()

    def augm(array, r):
        if r.accepted == 1:
            r2 = r.copy()
            r2.discount_shp -= b + get_random_shift()
            r2.accepted = 0
            array.append(r2)

            r2 = r.copy()
            r2.discount_shp += a + get_random_shift()
            array.append(r2)
        if r.accepted == 0:
            r2 = r.copy()
            r2.discount_shp -= a + get_random_shift()
            array.append(r2)

            r2 = r.copy()
            r2.discount_shp += b + get_random_shift()
            r2.accepted = 1
            array.append(r2)

    return augm


def augm_basic_rshift_uni(cfg):
    # uniform
    a, b = cfg.augm_basic_params

    def get_random_shift(radius):
        return uniform(-radius, radius)

    def augm(array, r):
        if r.accepted == 1:
            r2 = r.copy()
            r2.discount_shp -= b + get_random_shift(3)
            r2.accepted = 0
            array.append(r2)

            r2 = r.copy()
            r2.discount_shp += a + get_random_shift(1.5)
            array.append(r2)
        if r.accepted == 0:
            r2 = r.copy()
            r2.discount_shp -= a + get_random_shift(1.5)
            array.append(r2)

            r2 = r.copy()
            r2.discount_shp += b + get_random_shift(3)
            r2.accepted = 1
            array.append(r2)

    return augm


def augm_orig(array, r):
    if r.accepted == 1:
        r2 = r.copy()
        r2.discount_shp -= 20
        r2.accepted = 0
        array.append(r2)

        r2 = r.copy()
        r2.discount_shp += 10
        array.append(r2)
    if r.accepted == 0:
        r2 = r.copy()
        r2.discount_shp -= 10
        array.append(r2)
        r2 = r.copy()
        r2.discount_shp += 20
        r2.accepted = 1
        array.append(r2)


def augm_1s(array, r):
    ds = 5, 15, 25
    m = 1 if r.accepted == 1 else -1
    for d in ds:
        r2 = r.copy()
        r2.discount_shp += m * d
        if 0 <= r2.discount_shp < 100:
            array.append(r2)
        else:
            break


def get_augm_func(cfg):
    variant = cfg.augm_variant
    if variant == "basic":
        return augm_basic(cfg)
    elif variant == "1sided":
        return augm_1s
    elif variant == "orig":
        return augm_orig
    elif variant == "shift":
        return augm_basic_rshift(cfg)
    elif variant == "shift-uni":
        return augm_basic_rshift_uni(cfg)
    elif variant == "no-augmentation":
        return None
    else:
        raise NotImplementedError()


def is_discount_valid(row):
    return 0 <= row.discount_shp < 100


def augment_data(data, cfg):
    if cfg.augm_basic_weight != 1:
        return augment_data_weighted(data, cfg)
    augm_func = get_augm_func(cfg)
    aug_rows = []
    for _, r in data.iterrows():
        aug_rows.append(r)
        augm_func(aug_rows, r)
    aug_rows = list(filter(is_discount_valid, aug_rows))
    return pd.DataFrame(aug_rows), None


def augment_data_weighted(data, cfg):
    augm_func = get_augm_func(cfg)
    aug_rows = []
    weights = []
    for _, r in data.iterrows():
        aug_rows.append(r)
        weights.append(1)
        augm_func(aug_rows, r, weights)
    df = pd.DataFrame(aug_rows).assign(weights=weights)
    df = df[df.apply(is_discount_valid, axis=1)]
    weights = df.weights.tolist()
    return df.drop(columns="weights"), weights
