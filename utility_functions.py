#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt

import sympy 
from scipy.optimize import curve_fit

import streamlit as st


def five_params_func(x, a, b, c, d, g):
    """Five parameters logistic function"""
    return d + (a - d)/(1 + (x/c)**b)**g


def params_search(func, x_values, y_values, method="trf", maxfev=1000000):
    """Search for the parameters' values """
    try:
        params, _ = curve_fit(func, x_values, y_values, method=method, maxfev=maxfev)
    except RuntimeError:
        print("cannot be solved")
    else:
        return params

def predict_func(func, params, lower, upper):
    """Predict the values for selected range"""
    range_value = upper - lower + 1
    x_values = np.linspace(lower, upper, range_value)
    return func(x_values, *params)

def derivative_func(params, lower, upper, derivative=1):
    """Generate the first derivative values for selected range"""
    range_value = upper - lower + 1 
    x_values = np.linspace(lower, upper, range_value)

    # symbolic variable
    x = sympy.Symbol('x')
    a, b, c, d, g = params
    func = d + (a - d)/(1 + (x/c)**b)**g
    first_deriv = func.diff()

    if derivative == 2:
        second_deriv = first_deriv.diff()
        return np.array([second_deriv.subs({x: x_value}) for x_value in x_values])

    return np.array([first_deriv.subs({x: x_value}) for x_value in x_values])
    
def plot_func(params, x_values, y_values, lower, upper):
    """Plot curve"""
    # original points
    x_value_index = [int(np.where(x_values == x)[0]) for x in x_values if lower <= x <= upper]
    x_values = x_values[x_value_index]
    y_values = y_values[x_value_index]

    # predicted value curve
    range_value = upper - lower + 1
    x = np.linspace(lower, upper, range_value)
    predicted_values = predict_func(five_params_func, params, lower, upper)

    predicted_values[predicted_values < 0] = 0

    # first derivative 
    first_deriv_predicted_values = derivative_func(params, lower, upper)
    max_x_index = np.argmax(first_deriv_predicted_values)

    max_day = x[max_x_index]
    max_growth_rate = round(float(first_deriv_predicted_values[max_x_index]), 3)

    # second derivative
    second_derive_predicted_values = derivative_func(params, lower, upper, 2)
    second_max_x_index = np.argmax(second_derive_predicted_values)
    second_min_x_index = np.argmin(second_derive_predicted_values)
    second_max_day = x[second_max_x_index]
    second_min_day = x[second_min_x_index]
    
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 7))
    # plot orginal data
    axes[0].scatter(x_values, y_values, color="red", label="original data")
    # plot growth curve 
    axes[0].plot(x, predicted_values, label="growth curve")
    axes[0].plot([max_day, max_day], [0, predicted_values[max_x_index]], linestyle="--", label="max growth rate day", color="orange")
    axes[0].plot([lower, upper], [0, 0], color="green")
    axes[0].legend()
    axes[0].set_title("growth curve", fontsize=20, fontweight="bold")

    # plot first derivative
    axes[1].plot(x, first_deriv_predicted_values, label="first deriv")
    axes[1].plot([max_day, max_day], [0, first_deriv_predicted_values[max_x_index]],  linestyle="--", color="orange")
    axes[1].plot([lower, upper], [0, 0], color="green")
    axes[1].set_title("first derivative", fontsize=20, fontweight="bold")

    # plot second derivative
    axes[2].plot(x, second_derive_predicted_values, label="first deriv")
    axes[2].plot([second_min_day, second_min_day],[0, second_derive_predicted_values[second_min_x_index]], linestyle="--", color="orange")
    axes[2].plot([second_max_day, second_max_day],[0, second_derive_predicted_values[second_max_x_index]], linestyle="--", color="orange")
    axes[2].plot([lower, upper], [0, 0], color="green")
    axes[2].set_title("second derivative", fontsize=20, fontweight="bold")

    return fig, max_day, max_growth_rate, second_max_day, second_min_day

if __name__ == "__main__":
    # test data 
    dataset_path = "../dataset/growth_curve_dataset.xlsx"
    dataset = pd.read_excel(dataset_path)
    
    x_value_index =  dataset['x'].notna()

    x_values = dataset['x'][x_value_index].values
    y_values = dataset.iloc[:, 1][x_value_index].values


    assert len(x_values) == len(y_values), "not equal length"

    params = params_search(five_params_func,  x_values, y_values, "trf", 1000000)

    predicted_values = predict_func(five_params_func, params, 10, 30)
    print(predicted_values.shape)

    plot_func(params, x_values, y_values, 1, 140)
    plt.show()
    

    



