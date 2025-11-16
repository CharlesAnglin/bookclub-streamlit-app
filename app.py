import streamlit as st
import requests
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt

#TODO: interation 2 reranking data

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

@st.cache_data(ttl=300, show_spinner=False)
def get_google_sheets_data(sheet_id=0):
    """Fetch data from Google Sheets and return as DataFrame."""
    try:
        url = st.secrets["google_drive"]["link"]

        # Convert Google Sheets URL to CSV export format
        csv_url = url.replace('/edit?usp=sharing', '/export?format=csv')
        csv_url += f"&gid={sheet_id}"
        
        return pd.read_csv(csv_url)
    
    except pd.errors.EmptyDataError:
        return pd.DataFrame()  # Return empty DataFrame

def style_and_print_dataframe_as_table(dataframe):
    try:
        df = dataframe
        
        # Format numeric columns to max 2 decimal places, strip trailing zeros
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col].round(2)
        
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
                    hide_index=True,
                    height=460)  # Set explicit height in pixels so that 12 rows can fit in
        
    except Exception as e:
        st.error(f"Error processing dataframe: {e}")
        st.text("DataFrame info:")
        st.write(dataframe)

def print_dataframe_as_slope_graph(dataframe):
    # Set figsize
    plt.figure(figsize=(4, 6))

    # Get the actual data range for proper scaling
    max_rank = max(df['Original ranking'].max(), df['Reranked ranking'].max())
    min_rank = min(df['Original ranking'].min(), df['Reranked ranking'].min())
    
    # Set proper y-axis limits
    plt.ylim(max_rank + 1, min_rank - 1)  # Inverted because rank 1 should be at top

    # Vertical lines for before and after
    plt.axvline(x=1, color='black', linestyle='--', linewidth=1)
    plt.axvline(x=2, color='black', linestyle='--', linewidth=1)

    # Add labels at appropriate y-positions
    plt.text(0.5, min_rank - 0.5, 'ORIGINAL', fontsize=12, color='black', fontweight='bold')
    plt.text(2.1, min_rank - 0.5, 'RERANKED', fontsize=12, color='black', fontweight='bold')

    # Plot lines for each book
    left_label_positions = {}  # Track left label positions to avoid overlaps
    right_label_positions = {}  # Track right label positions to avoid overlaps
    
    for i, row in df.iterrows():
        # Determine line color based on ranking change
        original_rank = row['Original ranking']
        reranked_rank = row['Reranked ranking']
        
        if original_rank > reranked_rank:  # Rank improved (lower number = better rank)
            line_color = 'green'
        elif original_rank < reranked_rank:  # Rank got worse (higher number = worse rank)
            line_color = 'red'
        else:  # Rank stayed the same
            line_color = 'darkgray'
        
        # Plot the line connecting original and reranked positions
        plt.plot([1, 2], [row['Original ranking'], row['Reranked ranking']], 
                marker='o', alpha=0.7, color=line_color)
        
        # Calculate LEFT label position with offset for overlapping ranks
        left_base_position = row['Original ranking']
        
        # Check if this position is already used on the left
        if left_base_position in left_label_positions:
            # Add a small offset to avoid overlap
            offset = len(left_label_positions[left_base_position]) * 0.35
            left_label_y = left_base_position + offset
            left_label_positions[left_base_position].append(left_label_y)
        else:
            # First label at this position
            left_label_y = left_base_position
            left_label_positions[left_base_position] = [left_label_y]
        
        # Add book title and original ranking on the left side
        plt.text(0.8, left_label_y, 
                f"{row['Book']} ({row['Original ranking']})", 
                fontsize=9, 
                ha='right', 
                va='center',
                alpha=0.8)

        # Calculate RIGHT label position with offset for overlapping ranks
        right_base_position = row['Reranked ranking']
        
        # Check if this position is already used on the right
        if right_base_position in right_label_positions:
            # Add a small offset to avoid overlap
            offset = len(right_label_positions[right_base_position]) * 0.35
            right_label_y = right_base_position + offset
            right_label_positions[right_base_position].append(right_label_y)
        else:
            # First label at this position
            right_label_y = right_base_position
            right_label_positions[right_base_position] = [right_label_y]
        
        # Add book title and new ranking on the right side
        plt.text(2.2, right_label_y, 
                f"{row['Book']} ({row['Reranked ranking']})", 
                fontsize=9, 
                ha='left', 
                va='center',
                alpha=0.8)

    plt.xticks([1, 2], ['Original', 'Reranked'])
    plt.title('Book Ranking Changes')
    plt.yticks([]) # Remove y-axis
    plt.box(False) # Remove the bounding box around plot
    plt.grid(True, alpha=0.3)
    
    st.pyplot(plt)

def interations_dropdown(include_all_years=True):
    # Create iteration options based on the secret
    num_iterations = len(st.secrets["iteration"])
    individual_iteration_options = []
    for i in range(num_iterations):
        year = st.secrets["iteration"][str(i)]["year"]
        individual_iteration_options.append(f"Iteration {i+1} ({year})")

    if include_all_years:
        iteration_options = ["All years"] + individual_iteration_options
    else:
        iteration_options = individual_iteration_options

    selected_iteration = st.selectbox(
        "Bookclub iteration:",
        options=iteration_options,
        index=len(iteration_options) - 1  # default to the most recent iteration
    )

    return selected_iteration, iteration_options 

# Retrieve query parameter (?token=...)
query_params = st.query_params
token = query_params.get("token", None)

# Only show content if token matches secret
if token == st.secrets["token"]:

    # Create columns for logo and text side by side
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.image("bookclub_logo_white.png", width=250)

    with col2:
        st.title("Book Club Dashboard")
        st.write("Welcome to the Book Club Dashboard! Here you can explore scores, reranking, and personal stats from our book club iterations.")
    
    st.markdown("---")

    # Add three mutually exclusive buttons
    view_option = st.radio(
        "Select view:",
        # TODO: Add "Personal stats" functionality
        options=["Scores", "Reranking"],
        index=0,  # "Scores" selected by default
        horizontal=True
    )

    # Show iteration dropdown only when "Scores" is selected
    if view_option == "Scores":

        selected_iteration, iteration_options = interations_dropdown()

        if selected_iteration == "All years":
            # Fetch data from all iterations and combine them using pandas
            all_dataframes = []
            for i in range(len(st.secrets["iteration"])):
                sheet_id = st.secrets["iteration"][f"{i}"]["scoring_sheet_id"]
                df_temp = get_google_sheets_data(sheet_id)
                all_dataframes.append(df_temp)
            
            # Combine all dataframes into one
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            data = combined_df
        else:
            selected_index = iteration_options.index(selected_iteration)
            sheet_id = st.secrets["iteration"][f"{selected_index - 1}"]["scoring_sheet_id"]
            data = get_google_sheets_data(sheet_id)

        st.markdown("---")

        style_and_print_dataframe_as_table(data)

    if view_option == "Reranking":

        selected_iteration, iteration_options = interations_dropdown(False)

        selected_index = iteration_options.index(selected_iteration)
        sheet_id = st.secrets["iteration"][f"{selected_index}"]["reranking_sheet_id"]
        data = get_google_sheets_data(sheet_id)

        st.markdown("---")

        # Check if dataframe is empty before creating slope graph
        if data.empty:
            st.info("No reranking data available for the selected iteration.")
        else:
            # Keep only specific columns (adjust column names as needed)
            columns_to_keep = ['Book', 'Original ranking', 'Reranked ranking']  # Add your desired columns here
            df = data[columns_to_keep]
            print_dataframe_as_slope_graph(df)

else:
    st.error("🚫 Access denied.")