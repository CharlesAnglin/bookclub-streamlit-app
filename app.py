import streamlit as st
import requests
import pandas as pd
import io
import seaborn as sns
import matplotlib.pyplot as plt

#TODO: interation 2 reranking data
#TODO: standard deviation per book (which ones were most divisive)
#TODO: score vs publication year
#TODO: genres analysis

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
    plt.title('Book Reranking')
    plt.yticks([]) # Remove y-axis
    plt.box(False) # Remove the bounding box around plot
    plt.grid(True, alpha=0.3)
    
    st.pyplot(plt)

def create_facet_grid_with_stats(dataframe):
    """Create a seaborn ridge plot from the dataframe after removing first 4 columns."""    
    # Remove first 4 columns
    df_analysis = dataframe.iloc[:, 4:].copy()
    
    # Melt the dataframe to create long format for FacetGrid
    df_melted = pd.melt(df_analysis.reset_index(), 
                       id_vars=['index'], 
                       value_vars=df_analysis.columns,
                       var_name='Member', 
                       value_name='Score')
    
    # Remove rows with NaN scores
    df_melted = df_melted.dropna(subset=['Score'])
    
    if df_melted.empty:
        st.info("📊 No valid scores found for analysis.")
        return
    
    # Set the style for ridge plot
    sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    
    # Create a color palette
    n_categories = len(df_melted['Member'].unique())
    pal = sns.cubehelix_palette(n_categories, rot=-.25, light=.7)
    
    # Initialize the FacetGrid object with exaggerated ridge size
    g = sns.FacetGrid(df_melted, row="Member", hue="Member", 
                      aspect=8, height=1, palette=pal)
    
    # Draw the densities in a few steps
    g.map(sns.kdeplot, "Score",
          bw_adjust=.5, clip_on=False,
          fill=True, alpha=1, linewidth=1.5, clip=(0, 10))
    g.map(sns.kdeplot, "Score", clip_on=False, color="w", lw=2, bw_adjust=.5, clip=(0, 10))
    
    # Add a horizontal reference line
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
    
    # Define function to label the plot
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    
    g.map(label, "Score")
    
    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[], ylabel="", xlim=(-1.5, 10.0))
    g.despine(bottom=True, left=True)
    
    # Add a title and axis label
    g.figure.suptitle('Score Distribution Ridge Plot', y=1.02, fontsize=16, fontweight='bold')
    g.set_axis_labels("Score", "")
    
    plt.tight_layout()
    st.pyplot(plt)
    
    # Reset seaborn theme to default after plotting
    sns.reset_defaults()

def create_styled_summary_table(dataframe):
    """Create a styled summary statistics table for member scores."""
    # Remove first 4 columns
    df_analysis = dataframe.iloc[:, 4:].copy()
    
    # Melt the dataframe to create long format
    df_melted = pd.melt(df_analysis.reset_index(), 
                       id_vars=['index'], 
                       value_vars=df_analysis.columns,
                       var_name='Member', 
                       value_name='Score')
    
    # Remove rows with NaN scores
    df_melted = df_melted.dropna(subset=['Score'])
    
    if df_melted.empty:
        st.info("📊 No valid scores found for summary table.")
        return
    
    # Preserve the original column order from the dataframe (after removing first 4 columns)
    original_member_order = dataframe.columns[4:].tolist()
    
    # Create summary stats with preserved order
    summary_stats = df_melted.groupby('Member')['Score'].agg(['mean', 'std', 'min', 'max']).round(2)
    
    # Rename columns to more descriptive names
    summary_stats.columns = ['Mean score', 'Score deviation', 'Lowest awarded score', 'Highest awarded score']
    
    # Reindex to match the original column order
    summary_stats = summary_stats.reindex(original_member_order)
    
    # Apply styling similar to the Raw Scores dataframe
    cm = sns.light_palette("green", as_cmap=True)
    styled_summary = summary_stats.style.background_gradient(cmap=cm, high=0.4)
    
    # Format to 2 decimal places like the Raw Scores table
    styled_summary = styled_summary.format("{:.2f}")
    
    st.dataframe(styled_summary)

def create_page_count_vs_score_scatter(dataframe):
    """Create a scatter plot of page count vs book score with book annotations."""
    # Check if required columns exist
    required_cols = ['Page Count', 'Average Score', 'Book']
    missing_cols = [col for col in required_cols if col not in dataframe.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.info(f"Available columns: {', '.join(dataframe.columns)}")
        return
    
    # Remove rows with missing data in required columns
    plot_data = dataframe[required_cols].dropna()
    
    if plot_data.empty:
        st.info("📊 No valid data found for scatter plot.")
        return
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Plot points
    plt.scatter(plot_data['Page Count'], plot_data['Average Score'], 
               alpha=0.7, s=80, c='steelblue', edgecolors='darkblue', linewidth=1)
    
    # Add book titles as annotations
    for i, row in plot_data.iterrows():
        book_title = row['Book']
        # Truncate long titles for readability
        if len(book_title) > 25:
            book_title = book_title[:22] + "..."
        
        plt.annotate(book_title, 
                    (row['Page Count'], row['Average Score']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', 
                             alpha=0.7, edgecolor='gray'),
                    ha='left')
    
    # Customize the plot
    plt.xlabel('Page Count', fontsize=12, fontweight='bold')
    plt.ylabel('Average Score', fontsize=12, fontweight='bold')
    plt.title('Book Score vs Page Count', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Add some padding to prevent labels from being cut off
    plt.tight_layout()
    plt.subplots_adjust(right=0.95, top=0.9)
    
    st.pyplot(plt)

def create_publication_year_vs_score_scatter(dataframe):
    """Create a scatter plot of publication year vs book score with book annotations."""
    # Check if required columns exist
    required_cols = ['Publication Year', 'Average Score', 'Book']
    missing_cols = [col for col in required_cols if col not in dataframe.columns]
    
    if missing_cols:
        st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        st.info(f"Available columns: {', '.join(dataframe.columns)}")
        return
    
    # Remove rows with missing data in required columns
    plot_data = dataframe[required_cols].dropna()
    
    if plot_data.empty:
        st.info("📊 No valid data found for scatter plot.")
        return
    
    # Create the scatter plot
    plt.figure(figsize=(12, 8))
    
    # Plot points
    plt.scatter(plot_data['Publication Year'], plot_data['Average Score'], 
               alpha=0.7, s=80, c='darkorange', edgecolors='darkred', linewidth=1)
    
    # Add book titles as annotations
    for i, row in plot_data.iterrows():
        book_title = row['Book']
        # Truncate long titles for readability
        if len(book_title) > 25:
            book_title = book_title[:22] + "..."
        
        plt.annotate(book_title, 
                    (row['Publication Year'], row['Average Score']),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=9, alpha=0.8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcyan', 
                             alpha=0.7, edgecolor='gray'),
                    ha='left')
    
    # Customize the plot
    plt.xlabel('Publication Year', fontsize=12, fontweight='bold')
    plt.ylabel('Average Score', fontsize=12, fontweight='bold')
    plt.title('Book Score vs Publication Year', fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    
    # Add some padding to prevent labels from being cut off
    plt.tight_layout()
    plt.subplots_adjust(right=0.95, top=0.9)
    
    st.pyplot(plt)

def interations_dropdown(include_all_years=True, default_to_most_recent=True):
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

    if default_to_most_recent:
        selected_iteration = len(iteration_options) - 1 # default to the most recent iteration
    else:
        selected_iteration = 0

    selected_iteration = st.selectbox(
        "Bookclub iteration:",
        options=iteration_options,
        index=selected_iteration
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
        
    st.markdown("---")

    # Add three mutually exclusive buttons
    view_option = st.radio(
        "Select view:",
        options=["Score Table", "Score Distribution", "Reranking", "Page Length", "Publication Year"],
        index=0,  # "Scores" selected by default
        horizontal=True
    )

    # Show iteration dropdown only when "Scores" is selected
    if view_option == "Score Table":

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

        st.subheader("Score Table")
        st.write("Click on the table headers to sort. Tap the fullscreen icon in the top right to expand table. If viewing on a smartphone it may help to turn your phone to landscape mode.")

        only_score_data = data.drop(columns=['Page Count', 'Publication Year'], errors='ignore')

        style_and_print_dataframe_as_table(only_score_data)

        # Also show a summary table
        st.subheader("Scores Stats Per Member")
        create_styled_summary_table(only_score_data)

    if view_option == "Score Distribution":

        selected_iteration, iteration_options = interations_dropdown(default_to_most_recent=False)

        if selected_iteration == "All years":
            # Fetch data from all iterations and combine them using pandas
            all_dataframes = []
            for i in range(len(st.secrets["iteration"])):
                sheet_id = st.secrets["iteration"][f"{i}"]["scoring_sheet_id"]
                df_temp = get_google_sheets_data(sheet_id)
                if not df_temp.empty:  # Only add non-empty dataframes
                    all_dataframes.append(df_temp)
            
            # Combine all dataframes into one
            combined_df = pd.concat(all_dataframes, ignore_index=True)
            data = combined_df
        else:
            selected_index = iteration_options.index(selected_iteration)
            sheet_id = st.secrets["iteration"][f"{selected_index - 1}"]["scoring_sheet_id"]
            data = get_google_sheets_data(sheet_id)

        st.markdown("---")

        st.subheader("Score Distribution")
        st.write("This graph shows the distribution of scores across different bookclub members.")

        only_score_data = data.drop(columns=['Page Count', 'Publication Year'], errors='ignore')

        create_facet_grid_with_stats(only_score_data)

    if view_option == "Reranking":

        selected_iteration, iteration_options = interations_dropdown(False, False)

        selected_index = iteration_options.index(selected_iteration)
        sheet_id = st.secrets["iteration"][f"{selected_index}"]["reranking_sheet_id"]
        data = get_google_sheets_data(sheet_id)

        st.markdown("---")

        st.subheader("Book Reranking")
        st.write("This graph shows how our opinion of books change. The left axis shows our ranking of books based on our scores. The right axis shows how we ranked books after reconsidering them after having read them all.")

        # Check if dataframe is empty before creating slope graph
        if data.empty:
            st.info("No reranking data available for the selected iteration.")
        else:
            # Keep only specific columns (adjust column names as needed)
            columns_to_keep = ['Book', 'Original ranking', 'Reranked ranking']  # Add your desired columns here
            df = data[columns_to_keep]
            print_dataframe_as_slope_graph(df)
    
    if view_option == "Page Length":
        
        selected_iteration, iteration_options = interations_dropdown(default_to_most_recent=False)

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

        st.subheader("Page count vs Book Score")
        st.write("This scatter plot shows the relationship between the page count of books and their average score. Each point represents a book, annotated with its title.")

        create_page_count_vs_score_scatter(data)

    if view_option == "Publication Year":
        
        selected_iteration, iteration_options = interations_dropdown(default_to_most_recent=False)

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

        st.subheader("Publication Year vs Book Score")
        st.write("This scatter plot shows the relationship between the Publication Year of books and their average score. Each point represents a book, annotated with its title.")

        create_publication_year_vs_score_scatter(data)

else:
    st.error("🚫 Access denied.")