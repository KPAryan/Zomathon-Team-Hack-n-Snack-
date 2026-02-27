import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error


def generate_data(num_orders=15000,
                  avg_active_orders=6,
                  peak_ratio=0.35,
                  seed=42):

    np.random.seed(seed)

    data = pd.DataFrame({
        "active_orders": np.random.poisson(avg_active_orders, num_orders),
        "complexity": np.random.randint(1, 6, num_orders),
        "peak_hour": np.random.choice([0, 1],
                                      num_orders,
                                      p=[1 - peak_ratio, peak_ratio])
    })

    BASE = 8

    data["true_prep_time"] = (
        BASE
        + 0.9 * data["active_orders"]
        + 1.5 * data["complexity"]
        + 4 * data["peak_hour"]
        + np.random.normal(0, 2, num_orders)
    )

    # Merchant bias
    data["merchant_FOR"] = (
        data["true_prep_time"]
        + np.random.normal(4, 5, num_orders)
    )

    # Rider arrival
    data["rider_arrival"] = (
        data["true_prep_time"]
        - np.random.normal(3, 4, num_orders)
    )

    return data


def apply_baseline(data):
    data["KPT_current"] = data["merchant_FOR"]
    return data


def apply_kli_model(data, kli_weight=0.7):

    data["KLI"] = (
        0.5 * data["active_orders"]
        + 0.3 * data["complexity"]
        + 0.2 * data["peak_hour"]
    )

    data["KPT_proposed"] = (
        data["merchant_FOR"]
        - kli_weight * data["KLI"]
    )

    return data


def calculate_metrics(data):

    mae_current = mean_absolute_error(
        data["true_prep_time"],
        data["KPT_current"]
    )

    mae_proposed = mean_absolute_error(
        data["true_prep_time"],
        data["KPT_proposed"]
    )

    data["wait_current"] = (
        data["KPT_current"] - data["rider_arrival"]
    )

    data["wait_proposed"] = (
        data["KPT_proposed"] - data["rider_arrival"]
    )

    avg_wait_current = data["wait_current"].mean()
    avg_wait_proposed = data["wait_proposed"].mean()

    p90_current = np.percentile(
        abs(data["true_prep_time"] - data["KPT_current"]), 90
    )

    p90_proposed = np.percentile(
        abs(data["true_prep_time"] - data["KPT_proposed"]), 90
    )

    return {
        "mae_current": mae_current,
        "mae_proposed": mae_proposed,
        "avg_wait_current": avg_wait_current,
        "avg_wait_proposed": avg_wait_proposed,
        "p90_current": p90_current,
        "p90_proposed": p90_proposed
    }