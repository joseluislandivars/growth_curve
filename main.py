#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Streamlit main function
"""
import numpy as np
import pandas as pd
import streamlit as st

from utility_functions import five_params_func
from utility_functions import params_search
from utility_functions import predict_func
from utility_functions import plot_func

import warnings
warnings.filterwarnings("ignore")


@st.experimental_memo
def save_predicted_values(dataset, lower, upper):
    x_value_index = dataset['x'].notna()
    x_values = dataset['x'][x_value_index].values
    y_values = dataset.iloc[:, 1:][x_value_index].values.T

    data = []
    for col in y_values:
        params = params_search(five_params_func, x_values, col)
        predicted_values = predict_func(five_params_func, params, lower, upper)
        data.append(predicted_values)

    data = np.array(data).T
    data[data < 0] = 0
    range_value = upper - lower + 1
    x = np.linspace(lower, upper, range_value).reshape((-1, 1))
    data = np.concatenate([x, data], axis=1)
    df = pd.DataFrame(data=data, columns=dataset.columns)
    return df.to_csv(index=False).encode("utf-8")


def main():
    """Strea Frame"""
    st.set_page_config(layout="wide")

    # sidebar
    with st.sidebar:
        # title and formula
        st.header("Growth Curve")
        st.latex(r"y = d + \frac{a -d }{(1 + (\frac{x}{c})^b)^g}")

        # upload data
        uploaded_file = st.file_uploader("choose a file")

        # range of the day
        range_values = st.slider(
                "Select a range: ", 1, 140, (0, 140)
                )

    # load data
    if uploaded_file:
        if uploaded_file.name.split(".")[-1] == "xlsx":
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        num_columns = len(df.columns) - 1
        selected_curve = st.slider("Curve: ", 1, num_columns, 1)

        x_value_index = df["x"].notna()
        x_values = df["x"][x_value_index].values
        y_values = df.iloc[:, selected_curve][x_value_index].values

        # solve parameters
        params = params_search(five_params_func, x_values, y_values)

        fig, max_day, max_growth_rate, sec_max_day, sec_min_day = plot_func(params, x_values, y_values, range_values[0], range_values[1])

        # show results
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Selected curve:", df.columns[selected_curve])
        with col2:
            st.metric("Max growth rate day:", int(max_day))
        with col3:
            st.metric("Max growth rate:", max_growth_rate)
        with col4:
            st.metric("Second derivative max day:", int(sec_max_day))
            st.metric("Second derivative min day:", int(sec_min_day))

        # plot
        st.pyplot(fig)

        # save values
        with st.sidebar:
            csv = save_predicted_values(df, range_values[0], range_values[1])
            st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name="predicted_values.csv",
                    mime="text/csv",
                    )
    else:
        st.title("Magic!")


if __name__ == "__main__":
    main()
