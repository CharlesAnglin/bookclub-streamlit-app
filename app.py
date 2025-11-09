import streamlit as st
import requests
import pandas as pd
import io
import seaborn as sns

# Make the dataframe take up the full width with wider layout
st.set_page_config(layout="wide")

# Add CSS to center content
st.markdown("""
<style>
.main .block-container {
    max-width: 1200px;
    margin: 0 auto;
    padding-left: 2rem;
    padding-right: 2rem;
}
</style>
""", unsafe_allow_html=True)

def get_google_sheets_data(sheet_num=0):
    """Fetch data from Google Sheets and return as bytes."""
    url = st.secrets["google_drive"]["link"]
    sheet_ids = st.secrets["google_drive"]["sheet_ids"]
    sheet_id = sheet_ids[sheet_num]
    
    # Convert Google Sheets URL to CSV export format
    csv_url = url.replace('/edit?usp=sharing', '/export?format=csv')
    csv_url += f"&gid={sheet_id}"
    
    response = requests.get(csv_url)
    data = response.content
    return data

def get_local_data():
    """Read data from local data.csv file and return as bytes."""
    with open('data.csv', 'rb') as file:
        data = file.read()
    return data

# Retrieve query parameter (?token=...)
query_params = st.experimental_get_query_params()
token = query_params.get("token", [None])[0]

if token == st.secrets["token"]:
        
    data = get_google_sheets_data(2)
    # data = get_local_data()
    
    # Parse CSV data into a pandas DataFrame
    try:
        df = pd.read_csv(io.StringIO(data.decode('utf-8')))
        
        # # Remove the last column
        df = df.iloc[:, :-1]
        
        # Format numeric columns to max 2 decimal places, strip trailing zeros
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2)
        
        st.subheader("Data from Google Sheets:")
        # Apply different color styling to different columns
        styled_df = df.style
        
        # Apply display formatting to numeric columns (keeps underlying data as numbers)
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            styled_df = styled_df.format({col: lambda x: f"{x:g}" if pd.notna(x) else "" for col in numeric_cols})
        
        cm = sns.light_palette("green", as_cmap=True)
        # Can't get this to look pretty but the below allows more control:
        # cm = sns.diverging_palette(121, 10, l=77, s=30, as_cmap=True)
        styled_df = styled_df.background_gradient(cmap=cm, subset=df.columns[3:], high=0.4)

        st.dataframe(styled_df, 
                    use_container_width=True,
                    hide_index=True,
                    height=460)  # Set explicit height in pixels so that 12 rows can fit in
        
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        st.text("Raw data:")
        st.text(data.decode('utf-8'))
    
else:
    st.error("🚫 Access denied.")