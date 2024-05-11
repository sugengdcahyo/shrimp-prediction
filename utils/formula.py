import pandas as pd
import numpy as np


def SR(n_start, n_finish):
    return round((n_start - n_finish) * 100, 2)


def ADG(w_start, w_finish, n_cycle):
    return round(
        (w_start - w_finish) / n_cycle, 2
    )