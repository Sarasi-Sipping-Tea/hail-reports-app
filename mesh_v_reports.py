import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Tuple, List, Dict
from scipy.stats import lognorm
from glob import glob
import os
import re

# --- Configuration Constants ---

# Get the directory of the currently executing script (streamlit_app.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# the structure is:
# /some/path/
# ├── streamlit_app.py
# └── hail-reports-app/
#     └── comparison_data/
#         └── daily_combined_data_200305xx.csv

DATA_DIR = os.path.join(BASE_DIR, "hail-reports-app", "comparison_data")

SAMPLE_FILE = "daily_combined_data_20030501.csv" # used as a local fallback check

# ... rest of the constants

# Define the directory where the daily CSV files are stored
#DATA_DIR = "hail-reports-app/comparison_data"
#SAMPLE_FILE = "daily_combined_data_20030501.csv" # Provided sample file name

# Columns for Hail Size Selection (For Log-Normal Fit)
MESH_SIZE_COLS = {
    'Max MESH 75': 'MESH75_max',
    'Max MESH 95': 'MESH95_max',
}
REPORT_SIZE_COLS = {
    'Max Reported Hail': 'reported_max_hail',
    'Average Reported Hail': 'reported_ave_hail',
}

# Columns for Event Frequency/Threshold Comparison
MESH_EVENT_COLS = {
    'MESH 75 ($\ge 1$ in)': 'MaxMESH75_1in',
    'MESH 75 ($\ge 2$ in)': 'MaxMESH75_2in',
    'MESH 95 ($\ge 1$ in)': 'MaxMESH95_1in',
    'MESH 95 ($\ge 2$ in)': 'MaxMESH95_2in',
}
# Report Event count is derived from non-NaN reported_max_hail

# Columns for Severity Matchup (Boolean Flags)
SEVERITY_MATCHUP_COLS = {
    'sev75 (MESH75/Report $\ge 1$in match)': 'sev75',
    'sigsev75 (MESH75/Report $\ge 2$in match)': 'sigsev75',
    'sev95 (MESH95/Report $\ge 1$in match)': 'sev95',
    'sigsev95 (MESH95/Report $\ge 2$in match)': 'sigsev95',
}

# Environmental Variables (for potential future use or display)
ENVIRONMENTAL_COLS = ['SHI_max', 'z0C', 'z20C', 'llrh', 'laps', 'pwat', 'lap24', 'shear']


@st.cache_data
def load_and_preprocess_data_combined(data_dir: str, sample_file: str) -> pd.DataFrame:
    """
    Loads, combines, and preprocesses all daily data files.
    """
    all_data = []
    file_list = []
    
    # 1. Search for files in the specified data directory
    if os.path.isdir(data_dir):
        # Look for all files matching the pattern
        # The os.path.join correctly handles slashes for different operating systems
        file_list = glob(os.path.join(data_dir, "daily_combined_data_*.csv"))
    
    # 2. Fallback: If no files were found in the full directory, check the root 
    # for the sample file (in case the user only uploaded the sample).
    if not file_list and os.path.exists(sample_file):
        file_list.append(sample_file) # Use the single file as the dataset

    if not file_list:
        st.error(f"Error: No data files found in '{data_dir}' or the root directory. Please check your data path.")
        return pd.DataFrame()

    # 3. Concatenate all dataframes
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # 4. Preprocessing and cleaning
    # Convert required columns to numeric, coercing errors to NaN
    for col in list(MESH_SIZE_COLS.values()) + list(REPORT_SIZE_COLS.values()):
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')

    # Convert boolean columns (they were read as bool, but ensure for safety)
    for col in list(SEVERITY_MATCHUP_COLS.values()):
        # Convert True/False to integer for plotting if needed, but bool is fine for filtering
        combined_df[col] = combined_df[col].astype(bool) 
        
    # Create the derived report event column
    combined_df['Report_Event'] = combined_df['reported_max_hail'].notna()

    return combined_df

def fit_lognormal(series: pd.Series) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    """Fits log-normal distribution to a series and returns data for plotting."""
    data = series.dropna().to_numpy()
    # Filter for positive sizes for log-normal fit
    data = data[data > 0]

    if len(data) < 2:
        return None, None, None, None, None

    try:
        # Fit the Log-Normal Distribution
        shape, loc, scale = lognorm.fit(data, floc=0) # Fix location (loc) at 0
        
        hist, bins = np.histogram(data, bins=30)
        bin_width = bins[1] - bins[0]
        
        # Create data for the fitted curve (Probability Density Function)
        x_min, x_max = data.min(), data.max()
        x_range = np.linspace(x_min, x_max, 100)
        
        pdf_values = lognorm.pdf(x_range, s=shape, loc=loc, scale=scale)
        # Scale PDF to match histogram counts (Area * N * bin_width = Total Area/Count)
        scaled_pdf_values = pdf_values * len(data) * bin_width
        
        return x_range, scaled_pdf_values, shape, loc, scale
    except Exception as e:
        st.warning(f"Log-Normal Fitting failed: {e}")
        return None, None, None, None, None

def create_size_distribution_chart(df: pd.DataFrame, mesh_col: str, report_col: str, mesh_label: str, report_label: str) -> alt.Chart:
    """
    Creates a side-by-side histogram with log-normal fit for MESH and Report sizes.
    """
    if df.empty:
        st.warning("No data for size distribution analysis.")
        return None

    # --- MESH Data Prep and Fit ---
    mesh_x, mesh_y, mesh_s, mesh_loc, mesh_scale = fit_lognormal(df[mesh_col])
    
    mesh_curve_df = pd.DataFrame()
    mesh_params = "Fitting failed for MESH."
    if mesh_x is not None:
        mesh_curve_df = pd.DataFrame({
            'Size': mesh_x,
            'Count': mesh_y,
            'Source': mesh_label
        })
        mesh_params = f"MESH Fit: Shape (s)={mesh_s:.4f}, Scale (e^μ)={mesh_scale:.4f}"

    # --- Report Data Prep and Fit ---
    report_x, report_y, report_s, report_loc, report_scale = fit_lognormal(df[report_col])

    report_curve_df = pd.DataFrame()
    report_params = "Fitting failed for Report."
    if report_x is not None:
        report_curve_df = pd.DataFrame({
            'Size': report_x,
            'Count': report_y,
            'Source': report_label
        })
        report_params = f"Report Fit: Shape (s)={report_s:.4f}, Scale (e^μ)={report_scale:.4f}"
        
    # --- Combine and Plot ---
    combined_curve_df = pd.concat([mesh_curve_df, report_curve_df], ignore_index=True)
    
    # Create a source DataFrame for the two histograms
    hist_data = pd.DataFrame({
        'Size': df[mesh_col].dropna().tolist() + df[report_col].dropna().tolist(),
        'Source': [mesh_label] * df[mesh_col].dropna().shape[0] + [report_label] * df[report_col].dropna().shape[0]
    })
    hist_data = hist_data[hist_data['Size'] > 0] # Filter for positive sizes

    if hist_data.empty:
         st.warning("Not enough positive hail size data to visualize.")
         return None

    # Base chart for layering
    base = alt.Chart(hist_data).encode(
        x=alt.X('Size:Q', bin=alt.Bin(maxbins=30), title='Hail Size (Inches)'),
        y=alt.Y('count()', title='Number of Events', scale=alt.Scale(zero=True)),
        color=alt.Color('Source:N', scale=alt.Scale(domain=[mesh_label, report_label], range=['#1f77b4', '#ff7f0e'])),
        tooltip=['Source', alt.Tooltip('Size:Q', bin=True, title='Hail Size Bin'), 'count()']
    ).properties(
        title='Max Hail Size Distribution (Log-Normal Fit)'
    )
    
    # Histogram layer
    hist_chart = base.mark_bar(opacity=0.7).encode(
        # Position bars side-by-side using the Source column
        column=alt.Column('Source:N', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
        x=alt.X('Size:Q', bin=alt.Bin(maxbins=30), axis=None)
    )

    # Fitted Curve layer
    curve_chart = alt.Chart(combined_curve_df).mark_line(strokeDash=[5, 5]).encode(
        x=alt.X('Size:Q'),
        y=alt.Y('Count:Q', title='Number of Events'),
        color=alt.Color('Source:N'),
        column=alt.Column('Source:N', header=alt.Header(titleOrient="bottom", labelOrient="bottom")),
        tooltip=[alt.Tooltip('Source:N'), alt.Tooltip('Size:Q'), alt.Tooltip('Count:Q', title='Lognorm Count', format='.1f')]
    )
    
    # Combine charts and remove the duplicated x-axis label on the columns
    combined_chart = (hist_chart + curve_chart).resolve_scale(
        x='independent', y='independent' # Independent scales for clarity
    ).interactive()
    
    return combined_chart, mesh_params, report_params

def create_event_frequency_chart(df: pd.DataFrame, mesh_col: str, report_col: str, mesh_label: str) -> alt.Chart:
    """
    Creates a bar chart comparing the total count of MESH events (value > 0) 
    vs. reported events (not NaN) for the entire dataset.
    """
    if df.empty:
        return None

    # Calculate MESH events (where the indicator column is > 0)
    mesh_count = (df[mesh_col] > 0).sum()
    
    # Calculate Report events (where reported_max_hail is not NaN)
    report_count = df['Report_Event'].sum()
    
    # Create the plotting DataFrame
    source = pd.DataFrame({
        'Source': [mesh_label, 'Reported Hail Event'],
        'Total Events': [mesh_count, report_count]
    })
    
    chart = alt.Chart(source).mark_bar().encode(
        x=alt.X('Source:N', title='Source'),
        y=alt.Y('Total Events:Q', title='Total Grid Points with Event', scale=alt.Scale(zero=True)),
        color=alt.Color('Source:N', legend=None),
        tooltip=['Source', 'Total Events']
    ).properties(
        title=f'Total Event Frequency Comparison ({mesh_label})'
    ).interactive()
    
    return chart

def create_severity_matchup_chart(df: pd.DataFrame, sev_col: str) -> alt.Chart:
    """
    Creates a bar chart showing the count of True/False for a selected severity matchup column.
    """
    if df.empty:
        return None

    # Count the True/False values
    counts = df[sev_col].value_counts().reset_index()
    counts.columns = ['Match', 'Count']
    counts['Match'] = counts['Match'].astype(str) # For categorical axis

    # Add column for percentage for tooltips
    total_count = counts['Count'].sum()
    counts['Percentage'] = counts['Count'] / total_count
    
    chart = alt.Chart(counts).mark_bar().encode(
        x=alt.X('Match:N', title='Match (True/False)'),
        y=alt.Y('Count:Q', title='Number of Grid Points', scale=alt.Scale(zero=True)),
        color=alt.Color('Match:N', scale=alt.Scale(domain=['True', 'False'], range=['#2ca02c', '#d62728'])),
        tooltip=['Match', 'Count', alt.Tooltip('Percentage', format='.1%')]
    ).properties(
        title=f'Severity Matchup: {sev_col}'
    ).interactive()
    
    return chart

# --- Streamlit App Layout ---
def main():
    """The main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="MESH vs. Hail Report Comparison")
    st.title('MESH Radar vs. Hail Report Analysis (2003-05)')
    st.markdown("""
        Compare MESH radar estimated hail size and frequency against ground-reported hail data 
        for the month of **May 2003**. The analysis aggregates all events across all grid points 
        and days in the dataset.
    """)

    # --- Data Loading ---
    data_load_state = st.empty()
    data_load_state.text(f"Loading and combining daily data from: `{DATA_DIR}`...")
    
    df = load_and_preprocess_data_combined(DATA_DIR, SAMPLE_FILE)

    if df.empty:
        data_load_state.error(f"Data is empty or could not be loaded/cleaned. Please ensure your daily CSV files are in the `{DATA_DIR}` folder.")
        st.stop()
    
    total_dates = df['Date'].dt.date.nunique()
    total_grid_points = len(df)
    data_load_state.success(f"Data loaded successfully! Total dates: **{total_dates}**. Total grid points: **{total_grid_points:,}**.")

    # --- Sidebar Selections ---
    st.sidebar.header("Visualization Settings")
    
    # 1. Hail Size Distribution Selection
    st.sidebar.subheader("Max Hail Size Comparison (Log-Normal Fit)")
    selected_mesh_size_label = st.sidebar.selectbox(
        'Select MESH Size Variable',
        options=list(MESH_SIZE_COLS.keys()),
        index=0 # Default MESH 75
    )
    selected_report_size_label = st.sidebar.selectbox(
        'Select Reported Hail Size Variable',
        options=list(REPORT_SIZE_COLS.keys()),
        index=0 # Default Max Reported Hail
    )
    mesh_size_col = MESH_SIZE_COLS[selected_mesh_size_label]
    report_size_col = REPORT_SIZE_COLS[selected_report_size_label]

    st.sidebar.markdown("---")
    
    # 2. Event Frequency Selection
    st.sidebar.subheader("Event Frequency Comparison")
    selected_mesh_event_label = st.sidebar.selectbox(
        'Select MESH Event Threshold',
        options=list(MESH_EVENT_COLS.keys()),
        index=0 # Default MESH 75 (> 1in)
    )
    mesh_event_col = MESH_EVENT_COLS[selected_mesh_event_label]
    
    st.sidebar.markdown("---")

    # 3. Severity Matchup Selection
    st.sidebar.subheader("Severity Matchup Analysis")
    selected_sev_label = st.sidebar.selectbox(
        'Select Severity Matchup Flag',
        options=list(SEVERITY_MATCHUP_COLS.keys()),
        index=0 # Default sev75
    )
    sev_col = SEVERITY_MATCHUP_COLS[selected_sev_label]
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Environmental Data")
    st.sidebar.info(f"The data contains {len(ENVIRONMENTAL_COLS)} environmental variables for future analysis.")

    # --- Section 1: Max Hail Size Distribution ---
    st.header('1. Max Hail Size Distribution (MESH vs. Report)')
    st.markdown(f"""
        Comparing the distribution of **{selected_mesh_size_label}** and **{selected_report_size_label}**. 
        A Log-Normal distribution is fitted to both datasets.
    """)
    
    size_chart, mesh_params, report_params = create_size_distribution_chart(
        df, mesh_size_col, report_size_col, selected_mesh_size_label, selected_report_size_label
    )
    
    if size_chart:
        st.altair_chart(size_chart, use_container_width=True)
        st.code(mesh_params + "\n" + report_params)
    else:
        st.warning("Could not generate Size Distribution chart due to insufficient data or fitting failure.")
        
    st.markdown("---")

    # --- Section 2: Event Frequency Comparison ---
    st.header('2. Event Frequency Comparison')
    st.markdown(f"""
        Comparing the total number of grid points that registered a **{selected_mesh_event_label}** event 
        against the total number of grid points with *any* **Reported Hail Event**.
    """)

    freq_chart = create_event_frequency_chart(
        df, mesh_event_col, 'Report_Event', selected_mesh_event_label
    )
    
    if freq_chart:
        st.altair_chart(freq_chart, use_container_width=True)
    else:
        st.warning("Could not generate Event Frequency chart.")

    st.markdown("---")

    # --- Section 3: Severity Matchup Analysis ---
    st.header('3. Severity Matchup Analysis')
    st.markdown(f"""
        Analyzing the **{selected_sev_label}** flag, which indicates a match between MESH 
        and reported hail based on their size thresholds ($\ge 1$ inch or $\ge 2$ inches).
    """)

    sev_chart = create_severity_matchup_chart(df, sev_col)
    
    if sev_chart:
        st.altair_chart(sev_chart, use_container_width=True)
    else:
        st.warning("Could not generate Severity Matchup chart.")


if __name__ == '__main__':
    main()
