import math
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Water Quality Analyser", layout="wide")

# ---- Custom CSS for Pro Dashboard ----
st.markdown("""
<style>
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
        text-align: center;
        margin-bottom: 15px;
    }
    .metric-title {
        color: #888888;
        font-size: 14px;
        font-weight: 500;
        margin-bottom: 5px;
    }
    .metric-value {
        color: #333333;
        font-size: 28px;
        font-weight: 700;
    }
    .metric-delta {
        font-size: 14px;
        font-weight: 500;
    }
    .assessment-container {
        background-color: #f0f8ff;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #0077b6;
        margin-bottom: 20px;
    }
    /* Simple badge style */
    .status-badge {
        background-color: #d4edda;
        color: #155724;
        padding: 4px 8px;
        border-radius: 12px;
        font-size: 12px;
        font-weight: 600;
        display: inline-block;
        margin-left: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ---- Helper Functions ----

def split_location(loc_str):
    """Splits 'River_Point' string into ('River', 'Point')."""
    if "_" in loc_str:
        parts = loc_str.split("_", 1)
        return parts[0], parts[1].replace('_', ' ')
    return "Unknown River", loc_str

def calculate_river_assessment(df):
    """Calculates metrics and generates assessment warnings per river."""
    warnings = []
    
    # helper to find columns
    def find_col(name, d):
        for col in d.columns:
            if col.lower() == name.lower(): return col
        return None

    turb_col = find_col('Turbidity_NTU', df)
    do_col = find_col('Dissolved_Oxygen_mgL', df)
    temp_col = find_col('Temp_C', df)
    
    # Iterate through each river present in the current filtered data
    unique_rivers = df['River_Name'].unique() if 'River_Name' in df.columns else []
    
    for river in unique_rivers:
        # Filter for this specific river
        river_df = df[df['River_Name'] == river]
        
        # Identify Upstream/Downstream columns for this river
        if 'Point' in river_df.columns:
            upstream_df = river_df[river_df['Point'].str.contains('Upstream', case=False, na=False)]
            downstream_df = river_df[river_df['Point'].str.contains('Downstream', case=False, na=False)]
            
            if not upstream_df.empty and not downstream_df.empty:
                # Turbidity Check
                if turb_col:
                    up_turb = upstream_df[turb_col].mean()
                    down_turb = downstream_df[turb_col].mean()
                    if down_turb > up_turb * 1.10:
                        warnings.append(f"⚠ **[{river}] Sedimentation/Runoff**: Downstream turbidity is >10% higher than upstream.")

                # DO Check
                if do_col:
                    up_do = upstream_df[do_col].mean()
                    down_do = downstream_df[do_col].mean()
                    if up_do - down_do > 1.0:
                        warnings.append(f"⚠ **[{river}] Organic Pollution**: Dissolved oxygen drops >1.0 mg/L downstream.")

                # Temp Check
                if temp_col:
                    up_temp = upstream_df[temp_col].mean()
                    down_temp = downstream_df[temp_col].mean()
                    if (down_temp - up_temp > 1.0):
                        warnings.append(f"⚠ **[{river}] Thermal Pollution**: Temperature is significantly higher downstream.")
    
    return warnings

def render_time_travel_analysis(df):
    """Renders the Time-Travel analysis section with side-by-side flexible plots."""
    col_map = {c.lower(): c for c in df.columns}
    dt_col = col_map.get('datetime')
    
    if not dt_col:
        st.warning("Time series data not available.")
        return

    # Prepare Data
    df_time = df.copy()
    df_time[dt_col] = pd.to_datetime(df_time[dt_col])
    # Sort for correct plotting
    df_time = df_time.sort_values(by=dt_col)
    
    st.markdown("### 📊 Comparative Analysis Panels")
    
    # Helper for Plot Panel
    def render_plot_panel(col_obj, key_suffix, default_mode="Compare Locations"):
        with col_obj:
            st.markdown(f"**Panel {key_suffix}**")
            modes = ["Compare Locations", "Compare Parameters"]
            default_index = modes.index(default_mode) if default_mode in modes else 0
            
            mode = st.radio("Plot Mode", modes, index=default_index, key=f"mode_{key_suffix}", horizontal=True)
            
            numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
            all_locs = sorted(df['Location'].unique().tolist())
            
            if mode == "Compare Locations":
                # Select 1 Param, N Locations
                param = st.selectbox("Parameter", options=numeric_cols, key=f"param_{key_suffix}")
                locs = st.multiselect("Locations", options=all_locs, default=all_locs[:min(5, len(all_locs))], key=f"locs_{key_suffix}")
                
                if locs and param:
                    plot_df = df_time[df_time['Location'].isin(locs)]
                    fig = px.line(plot_df, x=dt_col, y=param, color='Location', title=f"{param} by Location")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_loc_{key_suffix}")
            
            else: # Compare Parameters
                # Select 1 Location, N Params
                loc = st.selectbox("Location", options=all_locs, key=f"loc_{key_suffix}")
                params = st.multiselect("Parameters", options=numeric_cols, 
                                      default=numeric_cols[:2] if len(numeric_cols) > 1 else numeric_cols, 
                                      key=f"params_{key_suffix}")
                
                if loc and params:
                    plot_df = df_time[df_time['Location'] == loc]
                    # Melt for plotting multiple params
                    melted = plot_df.melt(id_vars=[dt_col], value_vars=params, var_name='Parameter', value_name='Value')
                    fig = px.line(melted, x=dt_col, y='Value', color='Parameter', title=f"Parameters at {loc}")
                    st.plotly_chart(fig, use_container_width=True, key=f"chart_param_{key_suffix}")

    # Side-by-Side Layout
    c1, c2 = st.columns(2)
    render_plot_panel(c1, "A", default_mode="Compare Locations")
    render_plot_panel(c2, "B", default_mode="Compare Parameters")


def diagnose_pollution_section(df):
    """Renders the Pollution Forensics section."""
    col_map = {c.lower(): c for c in df.columns}
    req_cols = ['flow_cfs', 'turbidity_ntu', 'dissolved_oxygen_mgl', 'ph', 'temp_c']
    available = [c for c in req_cols if c in col_map]
    
    if not available:
        st.warning("Insufficient data for forensics.")
        return

    st.markdown("### 🕵️ AI Forensics")
    
    # Simple thresholds
    thresholds = {
        'turbidity_ntu': (50, '>'),
        'dissolved_oxygen_mgl': (5.0, '<'),
        'ph': ((6.5, 8.5), 'outside'),
        'temp_c': (25.0, '>')
    }
    
    def check_row(row):
        reasons = []
        for k, v in col_map.items():
            if k in thresholds and k in row.index:
                thresh_val, op = thresholds[k]
                val = row[v]
                if pd.isna(val): continue
                
                if op == '>' and val > thresh_val:
                    reasons.append(f"High {k.split('_')[0]}")
                elif op == '<' and val < thresh_val:
                    reasons.append(f"Low {k.split('_')[0]}")
                elif op == 'outside' and (val < thresh_val[0] or val > thresh_val[1]):
                    reasons.append(f"Abnormal {k.split('_')[0]}")
        return ", ".join(reasons) if reasons else "Normal"

    forensics_df = df.copy()
    forensics_df['Diagnosis'] = forensics_df.apply(check_row, axis=1)
    issues = forensics_df[forensics_df['Diagnosis'] != "Normal"]
    
    if not issues.empty:
        st.warning(f"Detected {len(issues)} anomalous events.")
        fig = px.bar(issues['Diagnosis'].value_counts().reset_index(), x='Diagnosis', y='count', title="Event Types")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.success("No anomalies detected based on standard thresholds.")


# ---- Main App Logic ----

st.title("Water Quality Analyser")

# Load Data
# Initialize Session State for Data
if 'main_df' not in st.session_state:
    # Load initial data
    try:
        init_df = pd.read_csv("water_data.csv")
        # standardizing cols
        init_df.columns = [c.strip() for c in init_df.columns]
        
        # 1. Data Processing (Run Once)
        if "Location" in init_df.columns:
            init_df[['River_Name', 'Point']] = init_df['Location'].apply(lambda x: pd.Series(split_location(x)))
        else:
            init_df['River_Name'] = "Unknown River"
            init_df['Point'] = "Unknown"
            
        st.session_state['main_df'] = init_df
        
    except FileNotFoundError:
        st.error("water_data.csv not found. Please upload data.")
        st.session_state['main_df'] = pd.DataFrame(columns=['Location', 'River_Name', 'Point', 'Date'])

# Use the session state DF
df = st.session_state['main_df']
col_map = {c.lower(): c for c in df.columns}
dt_col = col_map.get('datetime')
if dt_col:
    df[dt_col] = pd.to_datetime(df[dt_col])

# ---- Regional Overview (Fleet Leaderboard) ----
with st.expander("🗺️ Regional Overview & Comparative Hydrology", expanded=True):
    col_reg1, col_reg2 = st.columns(2)
    
    # Leaderboard
    with col_reg1:
        st.subheader("🏆 Site Health Leaderboard")
        # Group by specific Location for granular view
        site_stats = df.groupby(['Location', 'River_Name'])[['Turbidity_NTU', 'Dissolved_Oxygen_mgL']].mean().reset_index()
        site_stats = site_stats.sort_values(by='Turbidity_NTU', ascending=False)
        
        fig_leader = px.bar(
            site_stats, 
            x='Location', 
            y='Turbidity_NTU', 
            title="Avg Turbidity by Site (Lower is Better)",
            hover_data=['River_Name', 'Dissolved_Oxygen_mgL'],
            color='Turbidity_NTU',
            color_continuous_scale=['green', 'yellow', 'red']
        )
        st.plotly_chart(fig_leader, use_container_width=True)

    # Comparative Hydrology
    # Comparative Hydrology
    with col_reg2:
        st.subheader("💧 Comparative Hydrology")
        if 'Flow_cfs' in df.columns and dt_col:
            # Aggregate to daily average
            df_hydro = df.copy()
            df_hydro['DateOnly'] = df_hydro[dt_col].dt.date
            daily_flow = df_hydro.groupby(['DateOnly', 'River_Name'])['Flow_cfs'].mean().reset_index()
            
            fig_hydro = px.line(
                daily_flow, 
                x='DateOnly', 
                y='Flow_cfs', 
                color='River_Name',
                title="Daily Average River Flow (cfs)",
                labels={'Flow_cfs': 'Avg Flow (cfs)', 'River_Name': 'River', 'DateOnly': 'Date'},
                markers=False
            )
            st.plotly_chart(fig_hydro, use_container_width=True)
        else:
            st.info("Flow and Time data required for Hydrograph.")

# ---- Sidebar: River Selector & Filters ----
st.sidebar.markdown("## Filters <span class='status-badge'>🟢 Live</span>", unsafe_allow_html=True)

# River Selector
# River Selector
unique_rivers = sorted(df['River_Name'].unique().tolist())
selected_rivers = st.sidebar.multiselect("Select Waterway", options=unique_rivers, default=[unique_rivers[0]] if unique_rivers else [])

if not selected_rivers:
    st.warning("Please select at least one waterway to view the dashboard.")
    st.stop()

# Filter Dataset to Selected Rivers
river_df = df[df['River_Name'].isin(selected_rivers)]

# Secondary Location Filter (Points within River) -> REMOVED to decouple context
# filtered_df starts as the river_df (River Context)
filtered_df = river_df

# Timeframe logic
if dt_col: 
    
    min_date, max_date = df[dt_col].min().date(), df[dt_col].max().date()
    start_date = st.sidebar.date_input("Start", min_date, min_value=min_date, max_value=max_date)
    end_date = st.sidebar.date_input("End", max_date, min_value=min_date, max_value=max_date)
    
    # Filter only the river-specific DF for the main tabs
    mask = (filtered_df[dt_col].dt.date >= start_date) & (filtered_df[dt_col].dt.date <= end_date)
    filtered_df = filtered_df[mask]

# ---- Top Section: Assessment ----
# ---- Top Section: Assessment ----
st.markdown("### 📝 Water Quality Assessment")
assessment_warnings = calculate_river_assessment(filtered_df)

st.markdown('<div class="assessment-container">', unsafe_allow_html=True)
if assessment_warnings:
    for w in assessment_warnings:
        st.markdown(f"- {w}")
else:
    st.markdown("- ✅ **Healthy System:** No significant issues detected. Water quality parameters appear within expected ranges.")
st.markdown('</div>', unsafe_allow_html=True)

# ---- Tabbed Layout ----
tab_overview, tab_cross_site, tab_forensics, tab_raw = st.tabs(["📊 Overview", "🌐 Cross-Site Compare", "🔍 Forensics & Trends", "📝 Raw Data"])

# ==== TAB 1: OVERVIEW ====
with tab_overview:

    # -- Hero Section --
    # Calculate averages
    avg_ph = filtered_df[col_map.get('ph')].mean() if col_map.get('ph') else 0
    max_turb = filtered_df[col_map.get('turbidity_ntu')].max() if col_map.get('turbidity_ntu') else 0
    avg_do = filtered_df[col_map.get('dissolved_oxygen_mgl')].mean() if col_map.get('dissolved_oxygen_mgl') else 0

    col1, col2, col3 = st.columns(3)
    
    def metric_card(title, value, suffix=""):
        return f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}{suffix}</div>
        </div>
        """

    with col1:
        st.markdown(metric_card("Average pH", f"{avg_ph:.2f}"), unsafe_allow_html=True)
    with col2:
        st.markdown(metric_card("Max Turbidity", f"{max_turb:.1f}", " NTU"), unsafe_allow_html=True)
    with col3:
        st.markdown(metric_card("Avg Dissolved Oxygen", f"{avg_do:.2f}", " mg/L"), unsafe_allow_html=True)

    # -- Drill Down Chart --
    st.markdown("### 📈 Main Analysis Chart")
    
    # Parameter Selector
    param_options = {
        'Turbidity': {'col': col_map.get('turbidity_ntu'), 'color': '#8c564b', 'limit': 50},
        'pH': {'col': col_map.get('ph'), 'color': '#9467bd', 'min': 6.5, 'max': 8.5},
        'Dissolved Oxygen': {'col': col_map.get('dissolved_oxygen_mgl'), 'color': '#1f77b4', 'min': 5.0},
        'Temperature': {'col': col_map.get('temp_c'), 'color': '#d62728', 'max': 25.0}
    }
    # Filter out missing columns
    available_params = {k: v for k, v in param_options.items() if v['col'] in filtered_df.columns}
    
    if available_params:
        selected_param_name = st.radio("Select Parameter", list(available_params.keys()), horizontal=True)
        selected_info = available_params[selected_param_name]
        col_name = selected_info['col']
        
        # Sort by date for plotting
        chart_df = filtered_df.sort_values(by=dt_col) if dt_col else filtered_df
        
        # Determine coloring strategy
        if len(chart_df['Location'].unique()) > 1:
            color_arg = "Location" # Distinguish by Site (Location)
            color_seq = None  
        else:
            color_arg = None
            color_seq = [selected_info['color']]

        fig = px.line(
            chart_df, 
            x=dt_col if dt_col else chart_df.index, 
            y=col_name,
            color=color_arg,
            title=f"{selected_param_name} over Time",
            color_discrete_sequence=color_seq,
            markers=True
        )
        
        # Add Reference Lines
        if 'min' in selected_info:
            fig.add_hline(y=selected_info['min'], line_dash="dash", line_color="red", annotation_text="Min Safe Limit")
        if 'max' in selected_info:
            fig.add_hline(y=selected_info['max'], line_dash="dash", line_color="red", annotation_text="Max Safe Limit")
        if 'limit' in selected_info:
             fig.add_hline(y=selected_info['limit'], line_dash="dash", line_color="orange", annotation_text="Threshold")

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No numeric parameters available for charting.")

# ==== TAB 2: CROSS-SITE COMPARISON ====
with tab_cross_site:
    st.header("🌐 Cross-Site Analysis")
    st.caption("Compare distinct locations across different rivers.")
    
    # Independent Filters
    c1, c2 = st.columns(2)
    with c1:
        # Get all unique locations from the MAIN dataframe
        all_locs = sorted(df['Location'].unique().tolist())
        compare_locs = st.multiselect("Select Sites to Compare", options=all_locs, default=all_locs[:min(3, len(all_locs))])
    
    with c2:
        # Numeric columns only
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and "datetime" not in c.lower()]
        compare_param = st.selectbox("Select Parameter", options=num_cols, index=num_cols.index('Turbidity_NTU') if 'Turbidity_NTU' in num_cols else 0)

    if compare_locs and compare_param:
        # Filter main DF
        comp_df = df[df['Location'].isin(compare_locs)].copy()
        
        if dt_col:
            comp_df = comp_df.sort_values(by=dt_col)
            fig_comp = px.line(
                comp_df,
                x=dt_col,
                y=compare_param,
                color='Location',
                title=f"Cross-Site Comparison: {compare_param}",
                markers=True,
                hover_data=['River_Name']
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        else:
            st.warning("Time data not available for plotting.")
    else:
        st.info("Select sites and a parameter to visualize.")

# ==== TAB 3: FORENSICS ====
with tab_forensics:
    st.header("⏳ Time-Travel & Forensics")
    # Use GLOBAL df for Forensics to allow exploring outside the river context
    render_time_travel_analysis(df)
    st.markdown("---")
    diagnose_pollution_section(filtered_df)

# ==== TAB 3: RAW DATA ====
with tab_raw:
    st.header("💾 Data Center")
    
    dm_tab1, dm_tab2, dm_tab3 = st.tabs(["View Data", "Add Data", "Export Data"])
    
    # --- SUB-TAB 1: VIEW DATA ---
    with dm_tab1:
        # Column Config
        column_config = {}
        if col_map.get('temp_c'):
            column_config[col_map['temp_c']] = st.column_config.ProgressColumn(
                "Temp (°C)", min_value=0, max_value=35, format="%.1f"
            )
        if col_map.get('turbidity_ntu'):
            column_config[col_map['turbidity_ntu']] = st.column_config.ProgressColumn(
                "Turbidity (NTU)", min_value=0, max_value=100, format="%.1f"
            )
        if dt_col:
            column_config[dt_col] = st.column_config.DatetimeColumn("Date Time", format="MM-DD HH:mm")

        st.dataframe(filtered_df, column_config=column_config, use_container_width=True)

    # --- SUB-TAB 2: ADD DATA ---
    with dm_tab2:
        st.subheader("📤 Upload New Data")
        
        with st.expander("ℹ️ Data Format Guidelines & Template", expanded=False):
            st.markdown("""
            **Ensure your CSV matches the system format:**
            - **Date**: YYYY-MM-DD
            - **Location**: Must use format `RiverName_PointName` (e.g. `Cache_Upstream`, `Poudre_Downstream`).
            - **Metrics**: `pH`, `Turbidity_NTU`, `Dissolved_Oxygen_mgL`, `Temp_C`, `Flow_cfs` (optional).
            """)
            
            # Template Creation
            template_data = {
                'Date': ['2023-01-01'],
                'Location': ['ExampleRiver_Upstream'],
                'pH': [7.2],
                'Turbidity_NTU': [5.5],
                'Dissolved_Oxygen_mgL': [8.1],
                'Temp_C': [12.5],
                'Flow_cfs': [100]
            }
            template_df = pd.DataFrame(template_data)
            csv_template = template_df.to_csv(index=False).encode('utf-8')
            
            st.download_button(
                label="📥 Download CSV Template",
                data=csv_template,
                file_name="wq_upload_template.csv",
                mime="text/csv",
            )
        
        uploaded_file = st.file_uploader("Upload CSV (Append to current dataset)", type=['csv'])
        if uploaded_file is not None:
            try:
                new_data = pd.read_csv(uploaded_file)
                # Ensure columns match (basic check)
                if not new_data.empty:
                    # Run location split on new data
                    if "Location" in new_data.columns and "River_Name" not in new_data.columns:
                        new_data[['River_Name', 'Point']] = new_data['Location'].apply(lambda x: pd.Series(split_location(x)))
                    
                    if st.button("Confirm Upload & Merge"):
                        st.session_state['main_df'] = pd.concat([st.session_state['main_df'], new_data], ignore_index=True)
                        st.success("Data merged successfully!")
                        st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {e}")
        
        st.markdown("---")
        st.subheader("✍️ Manual Entry")
        with st.form("entry_form"):
            c1, c2 = st.columns(2)
            new_date = c1.date_input("Date")
            new_loc_str = c2.text_input("Location (Format: River_Upstream/Downstream)")
            
            c3, c4, c5, c6 = st.columns(4)
            val_ph = c3.number_input("pH", 0.0, 14.0, 7.0)
            val_turb = c4.number_input("Turbidity (NTU)", 0.0, 1000.0, 10.0)
            val_do = c5.number_input("DO (mg/L)", 0.0, 20.0, 8.0)
            val_temp = c6.number_input("Temp (°C)", -10.0, 40.0, 20.0)
            
            if st.form_submit_button("Add Record"):
                if new_loc_str:
                    # Create row
                    riv, pt = split_location(new_loc_str)
                    new_row = {
                        'Date': pd.to_datetime(new_date),
                        'Location': new_loc_str,
                        'River_Name': riv,
                        'Point': pt,
                        'pH': val_ph,
                        'Turbidity_NTU': val_turb,
                        'Dissolved_Oxygen_mgL': val_do,
                        'Temp_C': val_temp
                    }
                    st.session_state['main_df'] = pd.concat([st.session_state['main_df'], pd.DataFrame([new_row])], ignore_index=True)
                    st.success("Record added!")
                    st.rerun()
                else:
                    st.warning("Location is required.")

    # --- SUB-TAB 3: EXPORT DATA ---
    with dm_tab3:
        st.subheader("📥 Export Custom Dataset")
        
        # filters for export
        ex_locs = st.multiselect("Select Locations", options=unique_rivers, default=unique_rivers[:1]) # Using rivers for high level, or all unique locations? User asked for locations.
        # Let's use actual locations for granularity
        all_locs = sorted(df['Location'].unique().tolist())
        ex_sites = st.multiselect("Select Sites", options=all_locs, default=all_locs[:min(5, len(all_locs))])
        
        ex_params = st.multiselect("Select Columns", options=df.columns.tolist(), default=df.columns.tolist())
        
        if st.button("Generate Preview"):
            ex_df = df.copy()
            if ex_sites:
                ex_df = ex_df[ex_df['Location'].isin(ex_sites)]
            
            ex_df = ex_df[ex_params]
            st.dataframe(ex_df.head())
            
            csv = ex_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name="wq_export.csv",
                mime="text/csv",
            )

