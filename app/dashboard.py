import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date
import psycopg2
from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from llm_analyzer import OllamaClassroomAnalyzer
from report_generator import ReportGenerator

url_object = URL.create(
    "postgresql",
    username="postgres",
    password="sql@123",
    host="localhost",
    database="security",
    port=5432
)

# Page configuration
st.set_page_config(
    page_title="Classroom Monitoring Dashboard",
    page_icon="🏫",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f4e79;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f4e79;
    }
    .alert-card {
        background-color: #ffebee;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #f44336;
    }
</style>
""", unsafe_allow_html=True)

# Database connection function
@st.cache_resource
def init_connection():
    """Initialize database connection"""
    try:
        engine = create_engine(url_object)
        return engine
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None

# Data loading function
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_data():
    """Load data from PostgreSQL database"""
    engine = init_connection()
    if engine:
        try:
            query = """
            SELECT id, timestamp, category, headcount, reasoning, video_path
            FROM events
            ORDER BY timestamp DESC
            """
            df = pd.read_sql_query(query, engine)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['date'] = df['timestamp'].dt.date
            df['hour'] = df['timestamp'].dt.hour
            return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return None
    else:
        # Fallback sample data for demo
        return create_sample_data()

def generate_report_section(filtered_df):
    """Generate comprehensive report section"""
    st.subheader("📥 Generate Comprehensive Report")
    
    # Add debug info
    st.info(f"📊 Data available: {len(filtered_df)} events")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("🤖 **AI-Powered Analysis**\nThis report uses Ollama LLM for intelligent insights")
        
        # Model selection - only your two models
        model_options = ["qwen2.5vl:3b", "security_manager:latest"]
        selected_model = st.selectbox("Select LLM Model:", model_options)
        
        # Report options
        include_predictions = st.checkbox("Include Future Predictions", value=True)
        detailed_analysis = st.checkbox("Detailed Temporal Analysis", value=True)
    
    with col2:
        st.info("📊 **Available Formats**\n- PDF (Professional)\n- HTML (Interactive)\n- JSON (Data Export)")
        
        if st.button("🚀 Generate Complete Report Package", type="primary"):
            if len(filtered_df) == 0:
                st.error("No data to analyze. Please adjust your filters.")
                return
            
            with st.spinner("🔄 Analyzing data with AI..."):
                try:
                    st.info(f"Using model: {selected_model}")
                    st.info(f"Analyzing {len(filtered_df)} events")
                    
                    # Initialize components
                    analyzer = OllamaClassroomAnalyzer(model_name=selected_model)
                    generator = ReportGenerator()
                    
                    # Generate reports
                    files = generator.generate_complete_report_package(filtered_df)
                    
                    st.success("✅ Reports generated successfully!")
                    
                    # Display download buttons
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        with open(files['pdf'], 'rb') as f:
                            st.download_button(
                                "📄 Download PDF Report",
                                f.read(),
                                file_name=f"report_{datetime.now().strftime('%Y%m%d')}.pdf",
                                mime="application/pdf"
                            )
                    
                    with col2:
                        with open(files['html'], 'r', encoding='utf-8') as f:
                            st.download_button(
                                "🌐 Download HTML Report",
                                f.read(),
                                file_name=f"report_{datetime.now().strftime('%Y%m%d')}.html",
                                mime="text/html"
                            )
                    
                    with col3:
                        with open(files['json'], 'r', encoding='utf-8') as f:
                            st.download_button(
                                "📋 Download JSON Report",
                                f.read(),
                                file_name=f"report_{datetime.now().strftime('%Y%m%d')}.json",
                                mime="application/json"
                            )
                    
                except Exception as e:
                    st.error(f"❌ Report generation failed: {str(e)}")
                    st.info("💡 Make sure Ollama is running: `ollama serve`")
                    
                    # Show detailed error info
                    import traceback
                    with st.expander("🔍 Error Details"):
                        st.code(traceback.format_exc())

def create_sample_data():
    """Create sample data for demonstration"""
    import numpy as np
    
    timestamps = pd.date_range('2025-08-22 08:00', '2025-08-24 18:00', freq='30min')
    categories = np.random.choice(
        ['normal', 'pre_alert', 'alert', 'human_intervention_needed'], 
        size=len(timestamps), 
        p=[0.7, 0.2, 0.08, 0.02]
    )
    
    df = pd.DataFrame({
        'id': range(1, len(timestamps) + 1),
        'timestamp': timestamps,
        'category': categories,
        'headcount': np.random.randint(0, 30, len(timestamps)),
        'reasoning': ['Sample reasoning text'] * len(timestamps),
        'video_path': ['captured_clip.mp4'] * len(timestamps)
    })
    
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    return df

# Main dashboard
def main():
    st.markdown('<h1 class="main-header">🏫 Classroom Monitoring Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None or df.empty:
        st.error("No data available")
        return
    
    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    
    # Category filter
    st.sidebar.subheader("Alert Categories")
    categories = ['normal', 'pre_alert', 'alert', 'human_intervention_needed']
    selected_categories = []
    
    for category in categories:
        if st.sidebar.checkbox(f"📊 {category.replace('_', ' ').title()}", value=True):
            selected_categories.append(category)
    
    # Date range filter
    st.sidebar.subheader("📅 Date Range")
    min_date = df['date'].min()
    max_date = df['date'].max()
    
    start_date = st.sidebar.date_input("Start Date", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End Date", max_date, min_value=min_date, max_value=max_date)
    
    # Time filter
    st.sidebar.subheader("🕐 Time Range")
    time_range = st.sidebar.slider("Hour Range", 0, 23, (8, 18))
    
    # Apply filters
    filtered_df = df[
        (df['category'].isin(selected_categories)) &
        (df['date'] >= start_date) &
        (df['date'] <= end_date) &
        (df['hour'] >= time_range[0]) &
        (df['hour'] <= time_range[1])
    ]
    
    if filtered_df.empty:
        st.warning("No data matches the selected filters")
        return
    
    # Main dashboard content
    col1, col2, col3, col4 = st.columns(4)
    
    # Key metrics
    with col1:
        total_events = len(filtered_df)
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Events", total_events)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        alert_events = len(filtered_df[filtered_df['category'].isin(['alert', 'human_intervention_needed'])])
        st.markdown('<div class="alert-card">', unsafe_allow_html=True)
        st.metric("Critical Alerts", alert_events)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        avg_headcount = filtered_df['headcount'].mean()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Avg Headcount", f"{avg_headcount:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col4:
        unique_dates = filtered_df['date'].nunique()
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Days Monitored", unique_dates)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Charts section
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Category Distribution")
        category_counts = filtered_df['category'].value_counts()
        
        # Define colors for each category
        colors = {
            'normal': '#2E8B57',
            'pre_alert': '#FFD700', 
            'alert': '#FF6347',
            'human_intervention_needed': '#DC143C'
        }
        
        fig_pie = px.pie(
            values=category_counts.values,
            names=category_counts.index,
            title="Alert Category Distribution",
            color=category_counts.index,
            color_discrete_map=colors
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        st.subheader("📈 Daily Trends")
        daily_counts = filtered_df.groupby(['date', 'category']).size().reset_index(name='count')
        
        fig_line = px.line(
            daily_counts,
            x='date',
            y='count',
            color='category',
            title="Daily Alert Trends",
            color_discrete_map=colors
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    # Hourly heatmap
    st.subheader("🕐 Hourly Activity Heatmap")
    hourly_category = filtered_df.groupby(['hour', 'category']).size().reset_index(name='count')
    hourly_pivot = hourly_category.pivot(index='hour', columns='category', values='count').fillna(0)
    
    fig_heatmap = px.imshow(
        hourly_pivot.T,
        title="Activity Patterns by Hour and Category",
        labels=dict(x="Hour", y="Category", color="Count"),
        aspect="auto"
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    # Recent events table
    st.subheader("📋 Recent Events")
    
    # Additional filters for table
    show_only_alerts = st.checkbox("Show only alerts")
    if show_only_alerts:
        table_df = filtered_df[filtered_df['category'].isin(['alert', 'human_intervention_needed'])]
    else:
        table_df = filtered_df
    
    # Display table
    display_df = table_df[['timestamp', 'category', 'headcount', 'reasoning']].head(20)
    display_df['timestamp'] = display_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    st.dataframe(
        display_df,
        use_container_width=True,
        column_config={
            "timestamp": "Timestamp",
            "category": st.column_config.SelectboxColumn(
                "Category",
                options=['normal', 'pre_alert', 'alert', 'human_intervention_needed']
            ),
            "headcount": "Head Count",
            "reasoning": "Reasoning"
        }
    )
    
    # Add the report generation section
    generate_report_section(filtered_df)
    
    # Real-time refresh
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

if __name__ == "__main__":
    main()
