#!/usr/local/bin python
# -*- coding: utf-8 -*-
# Parallel coordinates plot for hparams.

import argparse
from pathlib import Path
import pandas as pd
from itertools import islice
import plotly.express as px


def read_args():
    parser = argparse.ArgumentParser(
        description="Parallel coordinates plot for hparams."
    )
    parser.add_argument("exp_dir", type=str, help="Path to experiment directory.")
    return parser.parse_args()


def _next_n_lines(fp, N):
    return [x.strip() for x in islice(fp, N)]


def read_log(log_file):
    data = []
    with open(log_file, "r") as f:
        while True:
            next_3_lines = _next_n_lines(f, 3)
            if not next_3_lines or len(next_3_lines) < 3:
                break
            _, onset, _, offset, _, min_duration_on, _, min_duration_off = (
                next_3_lines[0].strip().split()
            )
            missed_speech = next_3_lines[1].strip().split()[6]
            false_alarm = next_3_lines[2].strip().split()[6]
            data.append(
                {
                    "onset": float(onset),
                    "offset": float(offset),
                    "min_duration_on": float(min_duration_on),
                    "min_duration_off": float(min_duration_off),
                    "error": float(missed_speech) + float(false_alarm),
                }
            )
    df = pd.DataFrame(data)
    return df


def plot_parallel_coords(df):
    fig = px.parallel_coordinates(
        df,
        color="error",
        color_continuous_scale=px.colors.sequential.Plasma,
        labels={
            "onset": "Onset threshold",
            "offset": "Offset threshold",
            "min_duration_on": "Min duration on",
            "min_duration_off": "Min duration off",
        },
    )
    return fig


if __name__ == "__main__":
    args = read_args()
    exp_dir = Path(args.exp_dir)
    df = read_log(exp_dir / "hp_search.log")
    fig = plot_parallel_coords(df)
    fig.write_html(exp_dir / "parallel_coordinates.html")
