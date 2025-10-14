import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Tuple

# --- Configuration Constants ---

# Define the full list of 30 columns (0 to 29) based on the file structure.
COLUMNS_FULL = [f'Col_{i}' for i in range(30)]

# Assign meaningful names to the columns required for the analysis based on their index.
COLUMNS_FULL[6] = 'Timezone' # For cleaning (Index 6)
COLUMNS_FULL[10] = 'HailSize' # Hail Size (Index 10)
COLUMNS_FULL[11] = 'Injuries' # Injuries (Index 11)
COLUMNS_FULL[12] = 'Fatalities' # Fatalities (Index 12)
COLUMNS_FULL[15] = 'Latitude' # Starting Latitude (Index 15)
COLUMNS_FULL[16] = 'Longitude' # Starting Longitude (Index 16)
COLUMNS_FULL[29] = 'Date' # UTC_time (Index 29, used for Time Series)

TIME_SERIES_COLUMNS = ['HailSize', 'Injuries', 'Fatalities']
# Geographic constants
KM_PER_DEG_LAT = 110.574  # km per degree of latitude (approx)
KM_PER_DEG_LON_EQUATOR = 111.320  # km per degree of longitude at the equator (approx)
BOX_SIZE_KM = 80.0

@st.cache_data
def load_and_preprocess_data(uploaded_file):
    """Loads, names columns, cleans, and preprocesses the data."""
    if uploaded_file is None:
        return pd.DataFrame()

    # Load data without a header, and assign custom column names
    df = pd.read_csv(uploaded_file, header=None, names=COLUMNS_FULL)

    # The original file uses the first row as the header, so the true data starts on the second row (index 1)
    df = df.iloc[1:].copy()

    # Select only the columns needed for the app
    columns_to_keep = [name for name in COLUMNS_FULL if name.startswith('Col_') is False]
    df = df[columns_to_keep]

    # --- Data Cleaning ---

    # 1. Handle Timezone exclusion: Filter out 'unknown' and '9=GMT' events
    # Note: For this file, the 'Timezone' column (index 6) contains string representations
    # of the Timezone code (e.g., '3'). We check for the specific exclusion strings.
    df = df[~df['Timezone'].isin(['unknown', '9=GMT'])]

    # 2. Convert to numeric and date types
    for col in TIME_SERIES_COLUMNS + ['Latitude', 'Longitude']:
        # Coerce to numeric, turning non-convertible values into NaN
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Drop any rows where key columns are NaN after conversion
    df.dropna(subset=TIME_SERIES_COLUMNS + ['Latitude', 'Longitude'], inplace=True)

    # Convert Date column (UTC_time) to datetime objects
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    # Add Year and Year-Month columns for filtering (using the UTC Date)
    df['Year'] = df['Date'].dt.year
    df['Year-Month'] = df['Date'].dt.strftime('%Y-%m')

    return df.sort_values('Date')

def get_geographic_bounds(lat: float, lon: float) -> Tuple[float, float, float, float]:
    """
    Calculates the latitude/longitude bounds for an 80km x 80km box centered at (lat, lon).
    """
    # Calculate degree offset for the box
    lat_offset = (BOX_SIZE_KM / 2.0) / KM_PER_DEG_LAT

    # Longitude degrees are shorter away from the equator. Use the cosine of the latitude.
    lon_deg_at_lat = KM_PER_DEG_LON_EQUATOR * np.cos(np.deg2rad(lat))
    # Safety check for near-polar latitudes
    if lon_deg_at_lat == 0:
        lon_deg_at_lat = 1e-6
    lon_offset = (BOX_SIZE_KM / 2.0) / lon_deg_at_lat

    # Calculate the box bounds
    lat_min = lat - lat_offset
    lat_max = lat + lat_offset
    lon_min = lon - lon_offset
    lon_max = lon + lon_offset

    return lat_min, lat_max, lon_min, lon_max

def filter_geographic_data(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    """Filters the DataFrame for events within the 80km x 80km box."""
    if df.empty:
        return df

    lat_min, lat_max, lon_min, lon_max = get_geographic_bounds(lat, lon)

    # Filter by the bounding box
    filtered_df = df[
        (df['Latitude'] >= lat_min) & (df['Latitude'] <= lat_max) &
        (df['Longitude'] >= lon_min) & (df['Longitude'] <= lon_max)
    ].copy()

    return filtered_df

def create_timeseries_chart(df: pd.DataFrame, y_col: str, title: str) -> alt.Chart:
    """Creates a time series chart for a given column."""
    if df.empty:
        return None

    # Aggregate by date, summing up the values (e.g., total injuries per day)
    # HailSize shows the mean size per day.
    agg_method = 'mean' if y_col == 'HailSize' else 'sum'
    source = df.groupby(pd.Grouper(key='Date', freq='D'))[y_col].agg(agg_method).reset_index().rename(columns={y_col: 'Value'})
    # Remove rows where no events occurred on a day (Value is NaN)
    source.dropna(subset=['Value'], inplace=True)

    chart = alt.Chart(source).mark_line(point=True).encode(
        x=alt.X('Date:T', title='UTC Date'),
        y=alt.Y('Value:Q', title=y_col),
        tooltip=[alt.Tooltip('Date:T', title='Date (UTC)'), alt.Tooltip('Value:Q', title=y_col, format='.2f')]
    ).properties(
        title=title
    ).interactive()

    return chart

def create_histogram(df: pd.DataFrame, title: str) -> alt.Chart:
    """Creates a histogram for Hail Size."""
    if df.empty:
        return None

    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('HailSize:Q', bin=alt.Bin(maxbins=50), title='Hail Size (Inches)'),
        y=alt.Y('count()', title='Number of Events'),
        tooltip=[alt.Tooltip('HailSize:Q', bin=True), 'count()']
    ).properties(
        title=title
    ).interactive()

    return chart


# --- Streamlit App Layout ---
def main():
    """The main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="Hail Report Analysis (2000-2004)")
    st.title('Hail Report Analysis (2000-2004)')
    st.markdown("""
        Analyze time series data for hail reports using **UTC time** and perform geographic-specific analysis 
        around a chosen latitude and longitude.
    """)

    # --- Data Loading ---
    data_load_state = st.empty()
    data_load_state.text("Loading and preprocessing data...")
    # NOTE: Since the file is already uploaded, we use its accessible name
    try:
        df = load_and_preprocess_data("2000-2004_hail_utc.csv")
    except Exception as e:
        data_load_state.error(f"Error loading data: {e}. Please ensure the file is a valid CSV.")
        st.stop()

    if df.empty:
        data_load_state.error("Data is empty or could not be loaded/cleaned. Check your file format.")
        st.stop()

    data_load_state.success(f"Data loaded successfully! Total events: {len(df):,}")

    # --- Sidebar for Geographic Filtering ---
    st.sidebar.header("Geographic Cell Selection")
    
    # Set default lat/lon to a central point in the data's geographic range
    default_lat = round(df['Latitude'].mean(), 2)
    default_lon = round(df['Longitude'].mean(), 2)

    with st.sidebar:
        st.markdown(f"**Box Size:** {BOX_SIZE_KM} km x {BOX_SIZE_KM} km")
        selected_lat = st.number_input('Center Latitude ($^\circ$)', min_value=-90.0, max_value=90.0, value=default_lat, format="%.2f")
        selected_lon = st.number_input('Center Longitude ($^\circ$)', min_value=-180.0, max_value=180.0, value=default_lon, format="%.2f")

        # Get unique years and year-months for filtering
        all_years = sorted(df['Year'].unique())
        all_year_months = sorted(df['Year-Month'].unique())

        filter_options = ['All Time', 'Year', 'Year-Month']
        selected_filter = st.radio("Time-Based Histogram Filter", options=filter_options, index=0)

        year_filter = None
        year_month_filter = None

        if selected_filter == 'Year':
            # Default to the most recent year in the data
            default_index = len(all_years) - 1 if all_years else 0
            year_filter = st.selectbox('Select Year', options=all_years, index=default_index)
        elif selected_filter == 'Year-Month':
            # Default to the most recent month in the data
            default_index = len(all_year_months) - 1 if all_year_months else 0
            year_month_filter = st.selectbox('Select Year-Month', options=all_year_months, index=default_index)


    # --- Section 1: Overall Time Series Analysis ---
    st.header('1. Overall Time Series Analysis (All Data)')
    st.markdown("**(Mean Hail Size, Total Injuries, and Total Fatalities per Day, using UTC Time)**")
    
    cols = st.columns(len(TIME_SERIES_COLUMNS))
    for i, col in enumerate(TIME_SERIES_COLUMNS):
        with cols[i]:
            chart = create_timeseries_chart(df, col, f'Overall Time Series: {col}')
            if chart:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info(f"No data available for {col}.")

    st.markdown("---")

    # --- Section 2: Geographic Cell Analysis ---
    st.header(f'2. Analysis for Geographic Cell Centered at ${selected_lat}^\circ, {selected_lon}^\circ$')

    # Filter data based on the selected geographic point
    cell_df = filter_geographic_data(df, selected_lat, selected_lon)

    if cell_df.empty:
        st.warning("No hail events found within the 80km x 80km box around the selected point. Try adjusting the latitude/longitude.")
        st.stop()

    st.info(f"Found **{len(cell_df):,}** events within the selected geographic cell.")

    # 2.i. Time Series Analysis for Hail Size in Geographic Cell
    st.subheader('i. Hail Size Time Series in Cell')
    cell_ts_chart = create_timeseries_chart(cell_df, 'HailSize', 'Hail Size Time Series (Mean Daily in Cell)')
    if cell_ts_chart:
        st.altair_chart(cell_ts_chart, use_container_width=True)

    # 2.ii. Histogram of ALL Hail Sizes in Geographic Cell
    st.subheader('ii. Histogram of All Hail Sizes in Cell')
    cell_hist_chart = create_histogram(cell_df, 'Distribution of All Hail Sizes in Cell')
    if cell_hist_chart:
        st.altair_chart(cell_hist_chart, use_container_width=True)

    # 2.iii. Histogram of Hail Sizes in Geographic Cell Restricted by Year/Month
    st.subheader('iii. Time-Restricted Hail Size Histogram in Cell')

    restricted_df = cell_df.copy()
    filter_label = 'All Time'
    if year_filter is not None:
        restricted_df = restricted_df[restricted_df['Year'] == year_filter]
        filter_label = f'Year: {year_filter}'
    elif year_month_filter is not None:
        restricted_df = restricted_df[restricted_df['Year-Month'] == year_month_filter]
        filter_label = f'Month: {year_month_filter}'

    if restricted_df.empty:
        st.warning(f"No hail events found in the cell for the selected period: **{filter_label}**.")
    else:
        st.info(f"Showing **{len(restricted_df):,}** events for the period: **{filter_label}**")
        restricted_hist_chart = create_histogram(restricted_df, f'Hail Size Distribution in Cell ({filter_label})')
        if restricted_hist_chart:
            st.altair_chart(restricted_hist_chart, use_container_width=True)


if __name__ == '__main__':
    # Use the local file name since the file is already uploaded
    # The main function logic handles loading using this name
    main()
