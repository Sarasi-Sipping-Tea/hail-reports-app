import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from typing import Tuple
from scipy.stats import lognorm

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
    # Note: Hail events typically have a time component which we ignore for daily aggregation
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.dropna(subset=['Date'], inplace=True)

    # Convert 'Date' to just the date part for easier daily grouping/imputation
    df['Date'] = df['Date'].dt.floor('D')

    # Add Year, Year-Month, and Day columns for filtering (using the UTC Date)
    df['Year'] = df['Date'].dt.year
    df['Year-Month'] = df['Date'].dt.strftime('%Y-%m')
    df['Year-Month-Day'] = df['Date'].dt.strftime('%Y-%m-%d')


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
    """
    Creates a time series chart for a given column, ensuring all dates in the range are present.
    """
    if df.empty:
        return None

    # 1. Aggregate existing data by date
    # HailSize uses the mean size per day. Other columns use the sum (total per day).
    agg_method = 'mean' if y_col == 'HailSize' else 'sum'
    
    # We aggregate first to get daily values
    daily_agg = df.groupby('Date')[y_col].agg(agg_method).reset_index().rename(columns={y_col: 'Value'})

    # 2. Generate a full date range and impute missing dates with 0
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    # Create a full date range series
    full_range = pd.date_range(start=min_date, end=max_date, freq='D')
    full_dates_df = pd.DataFrame({'Date': full_range})

    # Merge the aggregated data with the full date range.
    # Non-matching dates (where no event occurred) will have NaN for 'Value'.
    source = pd.merge(full_dates_df, daily_agg, on='Date', how='left')

    # For 'Injuries' and 'Fatalities' (sum), fill missing values (no events) with 0.
    # For 'HailSize' (mean), we assume the user intends the average size of a day with no
    # reported events to be 0 for plotting purposes.
    source['Value'] = source['Value'].fillna(0)


    chart = alt.Chart(source).mark_line(point=True).encode(
        x=alt.X('Date:T', title='UTC Date'),
        y=alt.Y('Value:Q', title=y_col, scale=alt.Scale(zero=True)), # Ensure scale starts at 0
        tooltip=[alt.Tooltip('Date:T', title='Date (UTC)'), alt.Tooltip('Value:Q', title=y_col, format='.2f')]
    ).properties(
        title=title
    ).interactive()

    return chart

def create_histogram(df: pd.DataFrame, title: str) -> Tuple[alt.Chart, str]:
    """
    Creates a histogram for Hail Size, fits a log-normal distribution,
    and visualizes the fitted curve and parameters.
    Returns the combined chart and the parameter string.
    """
    if df.empty:
        return None, "No data to visualize."

    # 1. Prepare data for fitting (must be > 0 for log-normal)
    # Hail sizes are typically reported in inches and are positive.
    hail_sizes = df['HailSize'].to_numpy()
    
    # Filter out zeros just in case, though hail size typically starts at 0.75"
    hail_sizes = hail_sizes[hail_sizes > 0] 

    if len(hail_sizes) < 2:
        return None, "Not enough positive hail size data to fit a distribution."
    
    # 2. Fit the Log-Normal Distribution
    # lognorm.fit returns (shape, loc, scale)
    try:
        shape, loc, scale = lognorm.fit(hail_sizes, floc=0) # Fix location (loc) at 0
        
        # Determine the maximum height of the histogram bins for scaling the fitted curve
        hist, bins = np.histogram(hail_sizes, bins=50)
        
        # Calculate the bin width for normalization
        bin_width = bins[1] - bins[0]
        
        # 3. Create data for the fitted curve (Probability Density Function)
        x_min, x_max = hail_sizes.min(), hail_sizes.max()
        x_range = np.linspace(x_min, x_max, 200)

        # PDF value * Total counts * Bin Width = Scaled curve height
        # This scales the PDF to match the histogram counts (Area under PDF * total N = N)
        pdf_values = lognorm.pdf(x_range, s=shape, loc=loc, scale=scale)
        scaled_pdf_values = pdf_values * len(hail_sizes) * bin_width
        
        curve_df = pd.DataFrame({
            'HailSize': x_range,
            'Density': scaled_pdf_values
        })
        
        param_str = (
            f"Fitted Log-Normal Parameters (lognorm.fit):\n"
            f"Shape (s): {shape:.4f}\n"
            f"Scale (e^μ): {scale:.4f}\n"
            f"Location (loc): {loc:.4f} (Fixed at 0)"
        )
        
    except Exception as e:
        st.warning(f"Could not fit Log-Normal distribution: {e}")
        param_str = "Fitting failed."
        curve_chart = None
        # Proceed with just the histogram if fitting fails

    # 4. Create the Altair Histogram (Bar Chart)
    hist_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('HailSize:Q', bin=alt.Bin(maxbins=50), title='Hail Size (Inches)'),
        # CORRECTED: Replaced axis=alt.Axis(min=0) with scale=alt.Scale(zero=True)
        y=alt.Y('count()', title='Number of Events', scale=alt.Scale(zero=True)),
        tooltip=[alt.Tooltip('HailSize:Q', bin=True, title='Hail Size Bin'), 'count()']
    ).properties(
        title=title
    )
    
    # 5. Create the Altair Fitted Curve (Line Chart)
    if 'curve_df' in locals():
        curve_chart = alt.Chart(curve_df).mark_line(color='red', strokeDash=[5, 5]).encode(
            x=alt.X('HailSize:Q'),
            # Use the same scale for the overlaid curve for consistency
            y=alt.Y('Density:Q', title='Number of Events', scale=alt.Scale(zero=True)),
            tooltip=[alt.Tooltip('HailSize:Q'), alt.Tooltip('Density:Q', title='Lognorm Count', format='.1f')]
        )
        
        # Combine the histogram and the curve
        combined_chart = hist_chart + curve_chart
    else:
        # Return only the histogram if the curve failed to generate
        combined_chart = hist_chart


    return combined_chart.interactive(), param_str


# --- Streamlit App Layout ---
def main():
    """The main function to run the Streamlit app."""
    st.set_page_config(layout="wide", page_title="Hail Report Analysis (2000-2004)")
    st.title('Hail Report Analysis (2000-2004)')
    st.markdown("""
        Analyze time series data for hail reports using **UTC time** and perform geographic-specific analysis 
        around a chosen latitude and longitude. Missing dates in the time series are imputed with 0 events.
    """)

    # --- Data Loading ---
    data_load_state = st.empty()
    data_load_state.text("Attempting to load and preprocess data...")
    
    # The name of the data file is usually '2000-2004_hail_utc.csv'
    DATA_FILE_NAME = "2000-2004_hail_utc.csv"
    
    try:
        df = load_and_preprocess_data(DATA_FILE_NAME)
    except Exception as e:
        # If loading the expected file fails, check if the error is due to file not found
        if "No such file or directory" in str(e):
             data_load_state.error(f"Error: Data file '{DATA_FILE_NAME}' not found. Please ensure the required CSV data file is uploaded with the correct name.")
        else:
            # Handle other loading/parsing errors
            data_load_state.error(f"Error loading data from '{DATA_FILE_NAME}': {e}. Check the file format.")
        st.stop()


    if df.empty:
        data_load_state.error("Data is empty or could not be loaded/cleaned. Check your file format.")
        st.stop()

    data_load_state.success(f"Data loaded successfully! Total events: {len(df):,}")

    # --- Sidebar for Geographic Filtering ---
    st.sidebar.header("Geographic Cell Selection")
    
    # Set default lat/lon to a central point in the data's geographic range
    default_lat_overall = round(df['Latitude'].mean(), 2)
    default_lon_overall = round(df['Longitude'].mean(), 2)
    
    # --- Geographic Map Selector ---
    with st.sidebar:
        st.subheader("Map Selection (Default: Center of US Data)")
        st.markdown(f"**Box Size:** {BOX_SIZE_KM} km x {BOX_SIZE_KM} km")
        
        
        # Provide user with an option to manually enter or revert to default coordinates
        coord_choice = st.radio(
            "Set Geographic Center By:",
            ('Manual Entry', 'Mean of All Data'),
            index=1 # Default to mean of all data
        )

        # Initialize coordinates based on choice
        if coord_choice == 'Manual Entry':
            if 'selected_lat' not in st.session_state:
                st.session_state['selected_lat'] = default_lat_overall
            if 'selected_lon' not in st.session_state:
                st.session_state['selected_lon'] = default_lon_overall
        else: # Mean of All Data
            st.session_state['selected_lat'] = default_lat_overall
            st.session_state['selected_lon'] = default_lon_overall

        selected_lat = st.number_input('Center Latitude ($^\circ$)', 
                                       min_value=-90.0, max_value=90.0, 
                                       value=st.session_state['selected_lat'], 
                                       key='lat_input', format="%.2f")
        selected_lon = st.number_input('Center Longitude ($^\circ$)', 
                                       min_value=-180.0, max_value=180.0, 
                                       value=st.session_state['selected_lon'], 
                                       key='lon_input', format="%.2f")

        # Update session state with manual inputs
        st.session_state['selected_lat'] = selected_lat
        st.session_state['selected_lon'] = selected_lon

        # --- Create Map with Selected Point Highlighted ---
        
        # 1. Prepare data for map (limit points for performance)
        map_df = df[['Latitude', 'Longitude']].rename(columns={'Latitude': 'lat', 'Longitude': 'lon'})
        
        # 2. Create the center point DataFrame
        center_point_df = pd.DataFrame({
            'lat': [selected_lat],
            'lon': [selected_lon],
            'color': ['#FF0000']
        })

        # 3. Altair Chart for all hail events (faint points)
        base = alt.Chart(map_df).encode(
            latitude='lat:Q',
            longitude='lon:Q',
            tooltip=['lat', 'lon']
        ).properties(
            title='Hail Events and Selected Center'
        )

        # Layer 1: All hail events (small, light markers)
        hail_events = base.mark_circle(size=10, opacity=0.3, color='#006400').encode(
            tooltip=['lat', 'lon']
        )

        # Layer 2: The selected center point (large, red marker)
        center_marker = alt.Chart(center_point_df).mark_circle(size=200, color='red', opacity=0.8, stroke='black', strokeWidth=1).encode(
            latitude='lat:Q',
            longitude='lon:Q',
            tooltip=[
                alt.Tooltip('lat', title='Center Lat', format='.2f'), 
                alt.Tooltip('lon', title='Center Lon', format='.2f')
            ]
        )
        
        # Combine layers
        map_chart = (hail_events + center_marker).interactive().properties(
            height=300
        )
        
        st.altair_chart(map_chart, use_container_width=True)
        st.markdown(f"**Selected Center:** ${selected_lat}^\circ, {selected_lon}^\circ$")

        st.markdown("---") # Separator for Time Filter


    # --- Time Filter Initialization (Moved outside the sidebar to reuse variables later)
    # Get unique years, year-months, and days for filtering
    all_years = sorted(df['Year'].unique())
    all_year_months = sorted(df['Year-Month'].unique())
    all_year_month_days = sorted(df['Year-Month-Day'].unique()) # Full list of days

    with st.sidebar:
        filter_options = ['All Time', 'Year', 'Year-Month', 'Day'] # Added 'Day' option
        selected_filter = st.radio("Time-Based Histogram Filter", options=filter_options, index=0)

        year_filter = None
        year_month_filter = None
        day_filter = None # New filter variable

        if selected_filter == 'Year':
            default_index = len(all_years) - 1 if all_years else 0
            year_filter = st.selectbox('Select Year', options=all_years, index=default_index)
        elif selected_filter == 'Year-Month':
            default_index = len(all_year_months) - 1 if all_year_months else 0
            year_month_filter = st.selectbox('Select Year-Month', options=all_year_months, index=default_index)
        elif selected_filter == 'Day':
            default_index = len(all_year_month_days) - 1 if all_year_month_days else 0
            day_filter = st.selectbox('Select Day', options=all_year_month_days, index=default_index)

    # --- Apply Time Filter to FULL dataset ---
    full_restricted_df = df.copy()
    filter_label = 'All Time'
    
    if selected_filter == 'Year' and year_filter is not None:
        full_restricted_df = full_restricted_df[full_restricted_df['Year'] == year_filter]
        filter_label = f'Year: {year_filter}'
    elif selected_filter == 'Year-Month' and year_month_filter is not None:
        full_restricted_df = full_restricted_df[full_restricted_df['Year-Month'] == year_month_filter]
        filter_label = f'Month: {year_month_filter}'
    elif selected_filter == 'Day' and day_filter is not None:
        full_restricted_df = full_restricted_df[full_restricted_df['Year-Month-Day'] == day_filter]
        filter_label = f'Day: {day_filter}'

    # --- Section 1: Overall Time Series Analysis ---
    st.header('1. Overall Time Series Analysis (All Data)')
    st.markdown("**(Mean Hail Size, Total Injuries, and Total Fatalities per Day, using UTC Time). Dates with no reported events are shown as zero.**")
    
    cols = st.columns(len(TIME_SERIES_COLUMNS))
    for i, col in enumerate(TIME_SERIES_COLUMNS):
        with cols[i]:
            chart = create_timeseries_chart(df, col, f'Overall Time Series: {col}')
            if chart:
                st.altair_chart(chart, use_container_width=True)
            else:
                st.info(f"No data available for {col}.")

    st.markdown("---")

    # --- Section 3: Full Dataset Analysis (Time-Restricted) ---
    st.header(f'3. Global Hail Size Distribution (Filtered by Time)')
    
    if full_restricted_df.empty:
        st.warning(f"No hail events found globally for the selected period: **{filter_label}**.")
    else:
        st.info(f"Showing **{len(full_restricted_df):,}** total global events for the period: **{filter_label}**")
        global_hist_chart, global_params = create_histogram(
            full_restricted_df, 
            f'Global Hail Size Distribution ({filter_label})'
        )
        if global_hist_chart:
            st.altair_chart(global_hist_chart, use_container_width=True)
            st.code(global_params)

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

    # 2.ii. Histogram of ALL Hail Sizes in Geographic Cell (All Time)
    st.subheader('ii. Histogram of All Hail Sizes in Cell (All Time)')
    cell_hist_chart, cell_params = create_histogram(cell_df, 'Distribution of All Hail Sizes in Cell')
    if cell_hist_chart:
        st.altair_chart(cell_hist_chart, use_container_width=True)
        st.code(cell_params)

    # 2.iii. Histogram of Hail Sizes in Geographic Cell Restricted by Time Filter
    st.subheader('iii. Time-Restricted Hail Size Histogram in Cell')
    
    # Filter the cell data using the already determined time filter variables
    restricted_df = cell_df.copy()
    
    if selected_filter == 'Year' and year_filter is not None:
        restricted_df = restricted_df[restricted_df['Year'] == year_filter]
    elif selected_filter == 'Year-Month' and year_month_filter is not None:
        restricted_df = restricted_df[restricted_df['Year-Month'] == year_month_filter]
    elif selected_filter == 'Day' and day_filter is not None:
        restricted_df = restricted_df[restricted_df['Year-Month-Day'] == day_filter]

    # Output Results
    if restricted_df.empty:
        st.warning(f"No hail events found in the cell for the selected period: **{filter_label}**.")
    else:
        st.info(f"Showing **{len(restricted_df):,}** events for the period: **{filter_label}**")
        restricted_hist_chart, restricted_params = create_histogram(restricted_df, f'Hail Size Distribution in Cell ({filter_label})')
        if restricted_hist_chart:
            st.altair_chart(restricted_hist_chart, use_container_width=True)
            st.code(restricted_params)


if __name__ == '__main__':
    main()
