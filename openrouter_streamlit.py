import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.express as px
import plotly.figure_factory as ff
from datetime import datetime

# Page config
st.set_page_config(
    page_title="OpenRouter Models Explorer",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'models_data' not in st.session_state:
    st.session_state.models_data = None
if 'visible_columns' not in st.session_state:
    st.session_state.visible_columns = {
        'name': True, 'id': True, 'context_length': True,
        'prompt_price': True, 'completion_price': True,
        'image_price': True, 'modality': True, 'tokenizer': True,
        'provider_context_length': True, 'max_completion_tokens': True,
        'description': False, 'created': False, 'is_moderated': True
    }

def convert_to_microunits(price):
    """Convert price to micro-units (1 unit = 1/1,000,000)"""
    return price * 1_000_000

def format_price_microunits(price):
    """Format price in micro-units for display"""
    return f"${price:,.2f}µ"

def fetch_models():
    """Fetch models from OpenRouter API"""
    url = "https://openrouter.ai/api/v1/models"
    
    # Get API key from Streamlit secrets
    api_key = st.secrets.get("OPENROUTER_API_KEY", None)
    if not api_key:
        st.error("OpenRouter API key not found in secrets. Please add it to .streamlit/secrets.toml")
        return None
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        data = response.json()
        
        # Process models data
        processed_models = []
        for model in data.get('data', []):
            # Skip AutoRouter entry
            if model.get('id', '') == 'openrouter/auto':
                continue
                
            is_moderated = model.get('top_provider', {}).get('is_moderated', False)
            # Convert prices to micro-units
            prompt_price = convert_to_microunits(float(model.get('pricing', {}).get('prompt', 0)))
            completion_price = convert_to_microunits(float(model.get('pricing', {}).get('completion', 0)))
            image_price = convert_to_microunits(float(model.get('pricing', {}).get('image', 0)))
            
            processed_models.append({
                'id': model.get('id', 'N/A'),
                'name': model.get('name', 'N/A'),
                'context_length': model.get('context_length', 0),
                'prompt_price': prompt_price,
                'completion_price': completion_price,
                'image_price': image_price,
                'modality': model.get('architecture', {}).get('modality', 'N/A'),
                'tokenizer': model.get('architecture', {}).get('tokenizer', 'N/A'),
                'provider_context_length': model.get('top_provider', {}).get('context_length', 'N/A'),
                'max_completion_tokens': model.get('top_provider', {}).get('max_completion_tokens', 'N/A'),
                'created': model.get('created', 'N/A'),
                'description': model.get('description', 'N/A'),
                'is_moderated': is_moderated
            })
        return pd.DataFrame(processed_models)
    except Exception as e:
        st.error(f"Error fetching models: {str(e)}")
        return None

def display_statistics(df, column):
    """Display statistics for a numeric column"""
    if df[column].dtype in ['int64', 'float64']:
        stats = df[column].describe()
        st.write(f"Statistics for {column}:")
        col1, col2, col3, col4 = st.columns(4)
        
        # Format values based on column type
        if column in ['prompt_price', 'completion_price', 'image_price']:
            col1.metric("Mean", format_price_microunits(stats['mean']))
            col2.metric("Median", format_price_microunits(stats['50%']))
            col3.metric("Min", format_price_microunits(stats['min']))
            col4.metric("Max", format_price_microunits(stats['max']))
        else:
            col1.metric("Mean", f"{stats['mean']:.2f}")
            col2.metric("Median", f"{stats['50%']:.2f}")
            col3.metric("Min", f"{stats['min']:.2f}")
            col4.metric("Max", f"{stats['max']:.2f}")
        
        # Distribution plot
        fig = px.histogram(
            df,
            x=column,
            title=f'{column} Distribution',
            hover_data=['name'],  # Show model name on hover
            labels={column: f"{column} ({get_unit_label(column)})", 'count': 'Number of Models'},
            color='is_moderated'  # Color by moderation status
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        # For categorical columns
        value_counts = df[column].value_counts().reset_index()
        value_counts.columns = [column, 'count']
        
        # Create a mapping of values to model names
        hover_text = []
        for val in value_counts[column]:
            models = df[df[column] == val]['name'].tolist()
            hover_text.append(f"Models:<br>" + "<br>".join(models))
        
        fig = px.pie(
            value_counts,
            names=column,
            values='count',
            title=f'{column} Distribution',
            custom_data=[hover_text]
        )
        
        # Update hover template to show model names
        fig.update_traces(
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{customdata[0]}"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def get_unit_label(column):
    """Get the unit label for a column"""
    if column in ['prompt_price', 'completion_price', 'image_price']:
        return "micro-units (µ)"
    elif column == 'context_length':
        return "tokens"
    return ""

def filter_numeric_column(df, col, min_val, max_val):
    """Helper function to filter numeric columns"""
    if pd.api.types.is_numeric_dtype(df[col]):
        return df[(df[col] >= min_val) & (df[col] <= max_val)]
    return df

def main():
    st.title("OpenRouter Models Explorer")
    st.markdown("Explore and analyze models from OpenRouter")
    
    # Fetch data button
    if st.button("Refresh Data") or st.session_state.models_data is None:
        with st.spinner("Fetching models data..."):
            st.session_state.models_data = fetch_models()
    
    if st.session_state.models_data is not None:
        df = st.session_state.models_data
        
        # Sidebar
        st.sidebar.header("Settings")
        
        # Moderation filter
        moderation_filter = st.sidebar.radio(
            "Show models",
            ["All Models", "Unmoderated Only", "Moderated Only"],
            index=1
        )
        
        if moderation_filter == "Unmoderated Only":
            df = df[~df['is_moderated']]
        elif moderation_filter == "Moderated Only":
            df = df[df['is_moderated']]
        
        # Column visibility
        st.sidebar.subheader("Column Visibility")
        for col in st.session_state.visible_columns.keys():
            st.session_state.visible_columns[col] = st.sidebar.checkbox(
                f"Show {col}",
                value=st.session_state.visible_columns[col]
            )
        
        # Filters
        st.sidebar.header("Filters")
        
        # Text search
        search_term = st.sidebar.text_input("Search by name or ID")
        if search_term:
            df = df[
                df['name'].str.contains(search_term, case=False) |
                df['id'].str.contains(search_term, case=False)
            ]
        
        # Price filters
        st.sidebar.subheader("Price Filters (in micro-units µ)")
        price_cols = ['prompt_price', 'completion_price', 'image_price']
        for col in price_cols:
            min_price = float(df[col].min())
            max_price = float(df[col].max())
            if min_price != max_price:
                col1, col2 = st.sidebar.columns(2)
                with col1:
                    min_val = st.number_input(f"Min {col} (µ)", 
                                            min_value=min_price,
                                            max_value=max_price,
                                            value=min_price,
                                            format="%.2f")
                with col2:
                    max_val = st.number_input(f"Max {col} (µ)",
                                            min_value=min_price,
                                            max_value=max_price,
                                            value=max_price,
                                            format="%.2f")
                df = filter_numeric_column(df, col, min_val, max_val)
        
        # Context length filter
        st.sidebar.subheader("Context Length Filter")
        if 'context_length' in df.columns:
            min_ctx = int(df['context_length'].min())
            max_ctx = int(df['context_length'].max())
            if min_ctx != max_ctx:
                ctx_range = st.sidebar.slider(
                    "Context Length (tokens)",
                    min_value=min_ctx,
                    max_value=max_ctx,
                    value=(min_ctx, max_ctx),
                    step=1024  # Common context window increment
                )
                df = filter_numeric_column(df, 'context_length', ctx_range[0], ctx_range[1])
        
        # Categorical filters
        st.sidebar.subheader("Categorical Filters")
        categorical_cols = ['modality', 'tokenizer']
        for col in categorical_cols:
            if col in df.columns:
                options = sorted(df[col].unique().tolist())
                if len(options) > 1:
                    selected = st.sidebar.multiselect(
                        f"Select {col}",
                        options=options,
                        default=options
                    )
                    if selected:
                        df = df[df[col].isin(selected)]
        
        # Main content area
        tab1, tab2 = st.tabs(["Data View", "Statistics"])
        
        with tab1:
            # Display summary metrics
            col1, col2, col3 = st.columns(3)
            total_models = len(df)
            moderated_models = df['is_moderated'].sum()
            unmoderated_models = total_models - moderated_models
            
            col1.metric("Total Models", total_models)
            col2.metric("Moderated Models", int(moderated_models))
            col3.metric("Unmoderated Models", int(unmoderated_models))
            
            # Sorting options
            col1, col2 = st.columns([2, 1])
            with col1:
                sort_col = st.selectbox(
                    "Sort by",
                    [col for col in df.columns if st.session_state.visible_columns[col]]
                )
            with col2:
                sort_order = st.radio("Sort order", ['Ascending', 'Descending'], horizontal=True)
            
            # Apply sorting
            df_sorted = df.sort_values(
                by=sort_col,
                ascending=(sort_order == 'Ascending')
            )
            
            # Display data with only visible columns
            visible_cols = [col for col, visible in st.session_state.visible_columns.items() if visible]
            st.dataframe(
                df_sorted[visible_cols],
                hide_index=True,
                column_config={
                    'description': st.column_config.TextColumn(width='medium'),
                    'context_length': st.column_config.NumberColumn(format="%d tokens"),
                    'prompt_price': st.column_config.NumberColumn(format="$%.2fµ"),
                    'completion_price': st.column_config.NumberColumn(format="$%.2fµ"),
                    'image_price': st.column_config.NumberColumn(format="$%.2fµ"),
                    'is_moderated': st.column_config.CheckboxColumn("Moderated")
                }
            )
        
        with tab2:
            # Moderation distribution
            st.subheader("Moderation Status Distribution")
            mod_fig = px.pie(
                df,
                names='is_moderated',
                title='Models by Moderation Status',
                hover_data=['name'],
                labels={'is_moderated': 'Moderation Status'},
                category_orders={'is_moderated': [True, False]},
                color_discrete_map={True: 'green', False: 'red'}
            )
            mod_fig.update_traces(
                textinfo='percent+label',
                hovertemplate="<b>%{label}</b><br>Count: %{value}<br>Models: %{customdata[0]}"
            )
            st.plotly_chart(mod_fig, use_container_width=True)
            
            # Statistical analysis
            st.subheader("Column Statistics")
            selected_column = st.selectbox(
                "Select column for analysis",
                [col for col in df.columns if col not in ['id', 'description', 'created']]
            )
            display_statistics(df, selected_column)
            
            # Correlation matrix for numeric columns
            st.subheader("Correlation Matrix")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            if len(numeric_cols) > 1:
                corr_matrix = df[numeric_cols].corr()
                fig = px.imshow(
                    corr_matrix,
                    labels=dict(color="Correlation"),
                    title="Correlation Matrix of Numeric Features"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Export options
        st.subheader("Export Data")
        col3, col4 = st.columns(2)
        
        with col3:
            if st.button("Export to CSV"):
                csv = df_sorted[visible_cols].to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"openrouter_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col4:
            if st.button("Export to JSON"):
                json_str = df_sorted[visible_cols].to_json(orient='records', indent=2)
                st.download_button(
                    label="Download JSON",
                    data=json_str,
                    file_name=f"openrouter_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

if __name__ == "__main__":
    main()