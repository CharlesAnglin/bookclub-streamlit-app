import streamlit as st
import requests
import pandas as pd
import io

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

# Compare to secret
if token == st.secrets["token"]:

    data = get_google_sheets_data(2)
    # data = get_local_data()
    
    # Parse CSV data into a pandas DataFrame
    try:
        df = pd.read_csv(io.StringIO(data.decode('utf-8')))
        
        # Format numeric columns to max 2 decimal places, strip trailing zeros
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2)
                # Convert to string to remove trailing zeros, then back to numeric
                df[col] = df[col].apply(lambda x: f"{x:g}" if pd.notna(x) else x)
        
        st.subheader("Data from Google Sheets:")
        # Apply different color styling to different columns
        styled_df = df.style
        
        # Apply styling based on column positions
        if len(df.columns) > 9:  # Ensure we have enough columns
            # Columns 0-2: No style (first three columns)
            # Columns 3-8: Red to Green style
            styled_df = styled_df.background_gradient(cmap='RdYlGn', subset=df.columns[3:9], low=0.2, high=0.2)
            # Penultimate column (second to last): Red to Green style
            styled_df = styled_df.background_gradient(cmap='RdYlGn', subset=[df.columns[-2]], low=0.2, high=0.2)
            # Last column: Red to Green style
            styled_df = styled_df.background_gradient(cmap='RdYlGn', subset=[df.columns[-1]], low=0.2, high=0.2)

        else:
            # If fewer columns, apply default styling to all
            styled_df = styled_df.background_gradient(cmap='RdYlGn', low=0.2, high=0.2)
        
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        st.subheader("Basic Chart:")
        # Create a simple line chart
        st.line_chart(df.set_index(df.columns[0]))
        
        st.subheader("Bar Chart:")
        # Create a bar chart
        st.bar_chart(df.set_index(df.columns[0]))
        
    except Exception as e:
        st.error(f"Error parsing data: {e}")
        st.text("Raw data:")
        st.text(data.decode('utf-8'))
    
else:
    st.error("🚫 Access denied.")