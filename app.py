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

    data = get_google_sheets_data(0)
    # data = get_local_data()
    
    # Parse CSV data into a pandas DataFrame
    try:
        df = pd.read_csv(io.StringIO(data.decode('utf-8')))
        
        st.subheader("Data from Google Sheets:")
        # Apply color styling: red for low values, green for high values
        styled_df = df.style.background_gradient(cmap='RdYlGn', axis=None)
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