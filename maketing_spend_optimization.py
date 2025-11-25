import streamlit as st
import pickle
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import statsmodels.api as sm
import plotly.graph_objects as go


# IMPORTANT: Define transformation functions before loading model
def sqrt_transform(x):
    """Square root transformation"""
    return np.sqrt(np.clip(x, 0, None))

def month_cyclical(X):
    """Cyclical transformation for month"""
    m = np.asarray(X).astype(float)
    return np.column_stack([
        np.sin(2*np.pi*m/12), 
        np.cos(2*np.pi*m/12)
    ])

def month_names(transformer, input_features):
    """Generate feature names for month transformation"""
    base = input_features[0] if input_features else 'Month'
    return np.array([f'{base}_sin', f'{base}_cos'], dtype=object)

# Load the saved model
@st.cache_resource
def load_model():
    with open('moveins_model.pkl', 'rb') as f:
        return pickle.load(f)

# Load location data
@st.cache_data
def load_location_data():
    df = pd.read_csv('LocationData.csv')  # no index_col
    # Normalize headers: strip spaces, lower, collapse non-alnum to underscore
    norm = (
        df.columns
          .str.replace(r'^\ufeff', '', regex=True)  # drop BOM if present
          .str.strip()
          .str.lower()
          .str.replace(r'[^a-z0-9]+', '_', regex=True)
    )
    df.columns = norm

    # Map normalized names back to the exact ones the app expects
    rename_map = {
        'property': 'Property',
        'unit_count': 'Unit_Count',
        'area': 'Area',
        'rate_psf': 'rate_psf',
    }
    # Build dynamic map for common variants
    variants = {
        'Property':  ['property', 'prop', 'location', 'facility', 'propertyname', 'property_id'],
        'Unit_Count':['unit_count','units','unit_total','total_units','unittotal'],
        'Area':      ['area','avg_sqft','average_unit_size','avg_unit_size','avg_sf'],
        'rate_psf':  ['rate_psf','rate_per_sqft','price_psf','ratepsf'],
    }
    final_map = {}
    for target, cands in variants.items():
        for c in cands:
            if c in df.columns:
                final_map[c] = target
                break

    df = df.rename(columns=final_map)

    # Validate required columns exist
    required = ['Property', 'Unit_Count', 'Area', 'rate_psf']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(
            "LocationData.csv is missing required column(s): "
            + ", ".join(missing)
            + "\n\nFound columns: "
            + ", ".join(df.columns.astype(str))
        )
        st.stop()

    # Type safety
    df['Unit_Count'] = pd.to_numeric(df['Unit_Count'], errors='coerce').fillna(0).astype(int)
    df['Area'] = pd.to_numeric(df['Area'], errors='coerce')
    df['rate_psf'] = pd.to_numeric(df['rate_psf'], errors='coerce')

    return df


def calculate_optimal_spend(property_row, model_artifacts, month, length_of_stay, spend_range=(0, 10000)):
    """
    Calculate optimal marketing spend for a single property based on max net revenue
    """
    preprocessor = model_artifacts['preprocessor']
    nb_model = model_artifacts['nb_model']
    eps = model_artifacts['eps']
    
    # Create spend grid
    spend_grid = np.linspace(spend_range[0], spend_range[1], 100)
    
    # Prepare input data
    plot_df = pd.DataFrame({
        "rate_psf": property_row['rate_psf'],
        "Spend": spend_grid,
        "Month": month
    })
    
    # Transform and predict
    total_units = float(property_row['Unit_Count'])
    offset_plot = np.log(np.clip(total_units, 1.0, None)) * np.ones(len(spend_grid))
    plot_mat = preprocessor.transform(plot_df)
    plot_mat_sm = sm.add_constant(plot_mat, has_constant="add")
    y_counts = nb_model.predict(plot_mat_sm, offset=offset_plot)
    
    # Calculate revenues
    rev_per_movein = property_row['rate_psf'] * property_row['Area']
    gross_rev = y_counts * rev_per_movein * length_of_stay
    net_rev = gross_rev - spend_grid
    
    # Find optimal spend (max net revenue)
    optimal_idx = np.argmax(net_rev)
    optimal_spend = spend_grid[optimal_idx]
    optimal_net_rev = net_rev[optimal_idx]
    optimal_moveins = y_counts[optimal_idx]
    optimal_gross_rev = gross_rev[optimal_idx]
    
    return {
        'Property': property_row['Property'],
        'Optimal_Spend': optimal_spend,
        'Unit_Count': int(property_row['Unit_Count']),
        #'Area_SqFt': property_row['Area'],
        'Rate_PSF': property_row['rate_psf'],
        #'Optimal_Spend': optimal_spend,
        'Predicted_MoveIns': optimal_moveins,
        'Gross_Revenue': optimal_gross_rev,
        'Net_Revenue': optimal_net_rev,
        #'ROI': (optimal_net_rev / optimal_spend * 100) if optimal_spend > 0 else 0,
        'Cost_Per_MoveIn': (optimal_spend / optimal_moveins) if optimal_moveins > 0 else 0
    }


model_artifacts = load_model()
preprocessor = model_artifacts['preprocessor']
nb_model = model_artifacts['nb_model']
eps = model_artifacts['eps']

# Load location data
location_df = load_location_data()

# App title and description
st.title("Move-Ins & CLV Prediction Model")
st.markdown("""
This app predicts the number of move-ins based on marketing spend, rental rates, seasonality, and property size.
""")

# Add tabs for different views
tab1, tab2 = st.tabs(["📊 Single Property Analysis", "📈 All Properties Optimization"])

with tab1:
    # Original single property analysis
    st.subheader("📍 Select Property")
    location_options = ["Custom"] + location_df['Property'].sort_values().tolist()
    selected_location = st.selectbox(
        "Property Location",
        options=location_options,
        help="Select a property from the list or choose 'Custom' to enter your own values"
    )

    if selected_location == "Custom":
        default_units = 332
        default_area = 137.0
        default_rate = 0.75
    else:
        row = location_df.loc[location_df['Property'].eq(selected_location)].iloc[0]
        default_units = int(row['Unit_Count'])
        default_area = float(row['Area'])
        default_rate = float(row['rate_psf'])

    # --- Inputs (single set) ---
    col1, col2, col3 = st.columns(3)
    with col1:
        spend = st.number_input(
            "Marketing Spend ($)", min_value=0.0, max_value=10000.0,
            value=500.0, step=10.0, help="Total marketing budget"
        )
        rate_psf = st.number_input(
            "Rate per Square Foot ($)", min_value=0.0, max_value=10.0,
            value=default_rate, step=0.01, help="Rental price per square foot"
        )
    with col2:
        month = st.selectbox(
            "Month", options=list(range(1, 13)),
            format_func=lambda x: datetime(2024, x, 1).strftime('%B'),
            index=datetime.now().month - 1, help="Month of the year for prediction"
        )
        total_units = st.number_input(
            "Total Units", min_value=1, max_value=2500,
            value=default_units, step=1, help="Total number of units in the property"
        )
    with col3:
        avg_sqft = st.number_input(
        "Avg Unit Size (sqft)", min_value=1, max_value=2000,
        value=int(round(default_area)), step=1,
        help="Average square footage per unit (used for revenue estimates)"
        )
        length_of_stay = st.number_input(
            "Average Length of Stay (months)", min_value=1, max_value=24,
            value=9, step=1, help="Average tenant retention period"
        )

    # --- ONE BUTTON to run everything ---
    if st.button("Run Prediction & Charts", type="primary"):
        with st.spinner("Running model and generating insights..."):
            # Prediction
            input_data = pd.DataFrame({
                'rate_psf': [rate_psf],
                'Spend': [spend],
                'Month': [month]
            })
            X_input = preprocessor.transform(input_data)
            X_input_sm = sm.add_constant(X_input, has_constant="add")
            offset = np.log(np.clip(total_units, eps, None))
            pred_point = nb_model.get_prediction(X_input_sm, offset=offset)
            prediction = nb_model.predict(X_input_sm, offset=offset)[0]

            # KPIs
            k1, k2, k3, k4 = st.columns(4)
            with k1:
                st.metric("Predicted Move-Ins", f"{prediction:.1f}")
            with k2:
                move_in_rate = (prediction / total_units) * 100 if total_units > 0 else 0.0
                st.metric("Rental Velocity", f"{move_in_rate:.1f}%")
            with k3:
                cpm = spend / prediction if prediction > 0 else 0.0
                st.metric("Cost per Move-In", f"${cpm:,.0f}")
            with k4:
                clv = rate_psf * avg_sqft * length_of_stay
                st.metric("CLV per Move-In", f"${clv:,.0f}")

            # Charts
            spend_grid = np.linspace(0, 10000, 60)
            plot_df = pd.DataFrame({
                "rate_psf": rate_psf,
                "Spend": spend_grid,
                "Month": month
            })
            plot_units = float(total_units)
            offset_plot = np.log(np.clip(plot_units, 1.0, None)) * np.ones(len(spend_grid))
            plot_mat = preprocessor.transform(plot_df)
            plot_mat_sm = sm.add_constant(plot_mat, has_constant="add")
            pred_grid = nb_model.get_prediction(plot_mat_sm, offset=offset_plot)
            y_counts = nb_model.predict(plot_mat_sm, offset=offset_plot)
            sf = pred_grid.summary_frame()

            # Pick mean/CI columns robustly
            def pick(name_opts):
                for name in name_opts:
                    for c in sf.columns:
                        if c.lower() == name:
                            return c
                for name in name_opts:
                    for c in sf.columns:
                        if name in c.lower():
                            return c
                raise KeyError(f"Couldn't find {name_opts}")

            mean_col = pick(("mean", "predicted_mean"))
            lcol = pick(("mean_ci_lower", "ci_lower", "lower"))
            ucol = pick(("mean_ci_upper", "ci_upper", "upper"))

            maybe_mean = sf[mean_col].to_numpy()
            maybe_low  = sf[lcol].to_numpy()
            maybe_high = sf[ucol].to_numpy()

            if np.allclose(y_counts, maybe_mean, rtol=1e-4, atol=1e-8):
                y_mean, y_low, y_high = maybe_mean, np.maximum(0, maybe_low), np.maximum(0, maybe_high)
            else:
                y_mean, y_low, y_high = np.exp(maybe_mean), np.exp(maybe_low), np.exp(maybe_high)

            mask = np.isfinite(y_mean) & np.isfinite(y_low) & np.isfinite(y_high)

            rev_per_movein = rate_psf * avg_sqft
            gross_rev = y_mean * rev_per_movein * length_of_stay
            gross_lo  = y_low  * rev_per_movein * length_of_stay
            gross_hi  = y_high * rev_per_movein * length_of_stay
            net_rev   = gross_rev - spend_grid
            net_lo    = gross_lo  - spend_grid
            net_hi    = gross_hi  - spend_grid

            # Chart 1: Predicted Move-Ins vs Spend
            st.subheader("Predicted Move-Ins vs Marketing Spend")
            fig1 = go.Figure()
            fig1.add_trace(go.Scatter(
                x=spend_grid[mask],
                y=y_mean[mask],
                mode="lines",
                name="Predicted move-ins",
                line=dict(color="#1f77b4", width=3),
                customdata=np.stack([y_low[mask], y_high[mask]], axis=-1),
                hovertemplate=(
                    "Spend: $%{x:,.0f}<br>"
                    "Predicted: %{y:.2f}<br>"
                    "CI Low: %{customdata[0]:.2f}<br>"
                    "CI High: %{customdata[1]:.2f}<extra></extra>"
                )
            ))
            fig1.add_trace(go.Scatter(
                x=np.concatenate([spend_grid[mask], spend_grid[mask][::-1]]),
                y=np.concatenate([y_low[mask], y_high[mask][::-1]]),
                fill="toself",
                fillcolor="rgba(31,119,180,0.2)",
                line=dict(width=0),
                hoverinfo="skip",
                name="95% CI"
            ))
            fig1.add_vline(
                x=spend,
                line=dict(color="blue", width=2, dash="dot"),
                annotation_text=f"Current Spend (${spend:,.0f})",
                annotation_position="top right",
                annotation_font=dict(color="black")
            )
            fig1.update_layout(
                title=dict(
                    text=f"Monthly Move-Ins vs Spend - {datetime(2024, month, 1).strftime('%B')}",
                    font=dict(size=16, color="black")
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
                hovermode="x unified",
                legend=dict(bgcolor="rgba(255,255,255,0.7)", font=dict(color="black"))
            )
            fig1.update_xaxes(
                title_text="Marketing Spend ($)",
                title_font=dict(color="black", size=14),
                tickfont=dict(color="black"),
                gridcolor="rgba(0,0,0,0.1)"
            )
            fig1.update_yaxes(
                title_text="Predicted Move-Ins (count)",
                title_font=dict(color="black", size=14),
                tickfont=dict(color="black"),
                gridcolor="rgba(0,0,0,0.1)"
            )
            st.plotly_chart(fig1, use_container_width=True)

            # Chart 2: Gross Lifetime Revenue vs Spend
            st.subheader("Gross Lifetime Revenue vs Marketing Spend")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=spend_grid[mask],
                y=gross_rev[mask],
                mode="lines",
                name="Lifetime revenue",
                line=dict(color="#2ca02c", width=3),
                customdata=np.stack([gross_lo[mask], gross_hi[mask]], axis=-1),
                hovertemplate=(
                    "Spend: $%{x:,.0f}<br>"
                    "Gross Revenue: $%{y:,.0f}<br>"
                    "CI Low: $%{customdata[0]:,.0f}<br>"
                    "CI High: $%{customdata[1]:,.0f}<extra></extra>"
                )
            ))
            fig2.add_trace(go.Scatter(
                x=np.concatenate([spend_grid[mask], spend_grid[mask][::-1]]),
                y=np.concatenate([gross_lo[mask], gross_hi[mask][::-1]]),
                fill="toself",
                fillcolor="rgba(44,160,44,0.2)",
                line=dict(width=0),
                hoverinfo="skip",
                name="95% CI (gross)"
            ))
            fig2.add_vline(
                x=spend,
                line=dict(color="green", width=2, dash="dot"),
                annotation_text=f"Current Spend (${spend:,.0f})",
                annotation_position="top right",
                annotation_font=dict(color="black")
            )
            fig2.update_layout(
                title=dict(text="Predicted Gross Revenue vs Spend", font=dict(size=16, color="black")),
                plot_bgcolor="white",
                paper_bgcolor="white",
                hovermode="x unified",
                legend=dict(bgcolor="rgba(255,255,255,0.7)", font=dict(color="black"))
            )
            fig2.update_xaxes(
                title_text="Marketing Spend ($)",
                title_font=dict(color="black", size=14),
                tickfont=dict(color="black"),
                gridcolor="rgba(0,0,0,0.1)"
            )
            fig2.update_yaxes(
                title_text="Lifetime Revenue Moved In ($)",
                title_font=dict(color="black", size=14),
                tickfont=dict(color="black"),
                gridcolor="rgba(0,0,0,0.1)"
            )
            st.plotly_chart(fig2, use_container_width=True)

            # Chart 3: Net Lifetime Revenue vs Spend
            st.subheader("Net Lifetime Revenue vs Marketing Spend")
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=spend_grid[mask],
                y=net_rev[mask],
                mode="lines",
                name="Net revenue (gross − spend)",
                line=dict(color="#d62728", width=3),
                customdata=np.stack([net_lo[mask], net_hi[mask]], axis=-1),
                hovertemplate=(
                    "Spend: $%{x:,.0f}<br>"
                    "Net Revenue: $%{y:,.0f}<br>"
                    "CI Low: $%{customdata[0]:,.0f}<br>"
                    "CI High: $%{customdata[1]:,.0f}<extra></extra>"
                )
            ))
            fig3.add_trace(go.Scatter(
                x=np.concatenate([spend_grid[mask], spend_grid[mask][::-1]]),
                y=np.concatenate([net_lo[mask], net_hi[mask][::-1]]),
                fill="toself",
                fillcolor="rgba(214,39,40,0.2)",
                line=dict(width=0),
                hoverinfo="skip",
                name="95% CI (net)"
            ))
            fig3.add_vline(
                x=spend,
                line=dict(color="Red", width=2, dash="dot"),
                annotation_text=f"Current Spend (${spend:,.0f})",
                annotation_position="top right",
                annotation_font=dict(color="black")
            )
            fig3.update_layout(
                title=dict(text="Predicted Net Revenue vs Spend", font=dict(size=16, color="black")),
                plot_bgcolor="white",
                paper_bgcolor="white",
                hovermode="x unified",
                legend=dict(bgcolor="rgba(255,255,255,0.7)", font=dict(color="black"))
            )
            fig3.update_xaxes(
                title_text="Marketing Spend ($)",
                title_font=dict(color="black", size=14),
                tickfont=dict(color="black"),
                gridcolor="rgba(0,0,0,0.1)"
            )
            fig3.update_yaxes(
                title_text="Net Lifetime Revenue Moved In ($)",
                title_font=dict(color="black", size=14),
                tickfont=dict(color="black"),
                gridcolor="rgba(0,0,0,0.1)"
            )
            st.plotly_chart(fig3, use_container_width=True)

with tab2:
    st.subheader("Optimal Marketing Spend for All Properties")
    st.markdown("""
    Calculate the optimal marketing spend for each property based on maximizing net lifetime revenue.
    """)
    
    col_a, col_b = st.columns(2)
    with col_a:
        opt_month = st.selectbox(
            "Select Month for Analysis",
            options=list(range(1, 13)),
            format_func=lambda x: datetime(2024, x, 1).strftime('%B'),
            index=datetime.now().month - 1,
            key="opt_month"
        )
    with col_b:
        opt_length_of_stay = st.number_input(
            "Average Length of Stay (months)",
            min_value=1, max_value=24, value=9, step=1,
            key="opt_los"
        )
    
    if st.button("Calculate Optimal Spend for All Properties", type="primary"):
        with st.spinner(f"Calculating optimal spend for {len(location_df)} properties..."):
            results = []
            
            # Progress bar
            progress_bar = st.progress(0)
            
            for idx, (_, property_row) in enumerate(location_df.iterrows()):
                result = calculate_optimal_spend(
                    property_row, 
                    model_artifacts, 
                    opt_month, 
                    opt_length_of_stay
                )
                results.append(result)
                progress_bar.progress((idx + 1) / len(location_df))
            
            # Create dataframe
            results_df = pd.DataFrame(results)
            
            # Sort by net revenue descending
            results_df = results_df.sort_values('Net_Revenue', ascending=False)
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Optimal Spend", f"${results_df['Optimal_Spend'].sum():,.0f}")
            with col2:
                st.metric("Total Predicted Move-Ins", f"{results_df['Predicted_MoveIns'].sum():.0f}")
            with col3:
                st.metric("Total Net Revenue", f"${results_df['Net_Revenue'].sum():,.0f}")
                        
            # Format the dataframe for display
            display_df = results_df.copy()
            display_df['Optimal_Spend'] = display_df['Optimal_Spend'].apply(lambda x: f"${x:,.0f}")
            display_df['Predicted_MoveIns'] = display_df['Predicted_MoveIns'].apply(lambda x: f"{x:.1f}")
            display_df['Gross_Revenue'] = display_df['Gross_Revenue'].apply(lambda x: f"${x:,.0f}")
            display_df['Net_Revenue'] = display_df['Net_Revenue'].apply(lambda x: f"${x:,.0f}")
            #display_df['ROI'] = display_df['ROI'].apply(lambda x: f"{x:.1f}%")
            display_df['Cost_Per_MoveIn'] = display_df['Cost_Per_MoveIn'].apply(lambda x: f"${x:,.0f}")
            #display_df['Rate_PSF'] = display_df['Rate_PSF'].apply(lambda x: f"${x:.2f}")
            #display_df['Area_SqFt'] = display_df['Area_SqFt'].apply(lambda x: f"{x:.0f}")
            
            st.dataframe(display_df, use_container_width=True, height=400)
            
            # # Add visualization
            # st.subheader("📊 Visualization")
            
            # # Top 20 properties by net revenue
            # top_20 = results_df.head(20)
            
            # fig = go.Figure()
            # fig.add_trace(go.Bar(
            #     x=top_20['Property'],
            #     y=top_20['Net_Revenue'],
            #     marker_color='#2ca02c',
            #     name='Net Revenue',
            #     hovertemplate="<b>%{x}</b><br>Net Revenue: $%{y:,.0f}<extra></extra>"
            # ))
            
            # fig.update_layout(
            #     title=dict(text="Top 20 Properties by Net Revenue", font=dict(size=16, color="black")),
            #     xaxis_title="Property",
            #     yaxis_title="Net Revenue ($)",
            #     plot_bgcolor="white",
            #     paper_bgcolor="white",
            #     hovermode="x",
            #     xaxis_tickangle=-45
            # )
            
            # fig.update_xaxes(
            #     title_font=dict(color="black", size=14),
            #     tickfont=dict(color="black"),
            #     gridcolor="rgba(0,0,0,0.1)"
            # )
            # fig.update_yaxes(
            #     title_font=dict(color="black", size=14),
            #     tickfont=dict(color="black"),
            #     gridcolor="rgba(0,0,0,0.1)"
            # )
            
            # st.plotly_chart(fig, use_container_width=True)


# Sidebar with additional information
with st.sidebar:
    st.header("📊 Model Information")
    st.markdown("""
    This model uses:
    - **Negative Binomial GLM**
    - Square root transformation on Spend
    - Cyclical encoding for Month
    - Offset for Total Units
    
    The model accounts for:
    - Seasonal variations
    - Property size effects
    - Marketing investment returns
    - Pricing impacts
    """)
    
    st.markdown("---")
    st.markdown("### 💡 Tips")
    st.markdown("""
    - Higher spend typically increases move-ins
    - Lower rates may attract more residents
    - Summer months often see higher activity
    - Larger properties need proportionally more marketing
    """)
    
    st.markdown("---")
    st.markdown("### 📈 Using Model Insights")
    st.markdown("""
    The charts show:
    1. **Move-ins curve**: Diminishing returns at high spend
    2. **Gross revenue**: Total value from new tenants
    3. **Net revenue**: Profit after marketing costs
    
    Find the optimal spend where net revenue peaks!
    """)
    
    st.markdown("---")
    st.markdown(f"### 📍 Properties Available")
    st.markdown(f"**{len(location_df)}** locations loaded")

    # cd "C:\Users\NicholasHarris\Storage Rentals of America\Operations - Revenue Management\Promo Analysis\Move In Regression"
    # python -m streamlit run app_4.py