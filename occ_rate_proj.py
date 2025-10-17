
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(layout="wide")

# Initialize session state for storing whether to show predictions
if 'show_predictions' not in st.session_state:
    st.session_state.show_predictions = False

def get_user_inputs():
    first_occ_rate_per_sqft = st.sidebar.number_input("In Place SS Rent/PSF", min_value=0.0, max_value=10.0, value=1.0, step=0.01, key='first_occ')
    # sum_unit = st.sidebar.number_input("Total Units", min_value=0, max_value=1000000, value=100, step=1, key='sum_unit')
    # sqft = st.sidebar.number_input("Total RSF", min_value=0, max_value=1000000000, value=10000, step=1, key='sqft')
    MHHI = st.sidebar.number_input("3 Mile 2025 HHI", min_value=0, max_value=1000000, value=100000, step=1, key='mhhi')
    # Pop = st.sidebar.number_input("3 Mile 2025 Population", min_value=0, max_value=1000000, value=100000, step=1, key='pop')
    office = st.sidebar.checkbox("Is Office?", value=True, key='office')
    Supply = st.sidebar.number_input("3 Mile 2025 SS RSF/Capita", min_value=0.0, max_value=100.0, value=10.0, step=0.01, key='supply')
    perc_cc = st.sidebar.number_input("Percent CC", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key='perc_cc')
    perc_ncc = 1 - perc_cc
    zip_code = st.sidebar.text_input("Zip Code", value="10001", max_chars=5, key='zip_code')
    zip_code = zip_code.zfill(5)
    seller_reference = 1
    user_inputs = pd.DataFrame({
        'Seller Reference #': [seller_reference],
        'first_occ_rate_per_sqft': [first_occ_rate_per_sqft],
        # 'sum_unit': [sum_unit],
        # 'sqft': [sqft],
        'MHHI': [MHHI],
        # 'Pop': [Pop],
        'Office': [office],
        'Supply': [Supply],
        'perc_cc': [perc_cc],
        # 'perc_ncc': [perc_ncc],
        'Zip Code': [zip_code]
    })
    return user_inputs

def get_zip_codes():
    zip_codes = pd.read_csv('georef-united-states-of-america-zc-point.csv', delimiter=';')
    zip_codes[['Geo Point.1', 'Geo Point.2']] = zip_codes['Geo Point'].str.split(',', expand=True)
    zip_codes['Geo Point.1'] = pd.to_numeric(zip_codes['Geo Point.1'], errors='coerce')
    zip_codes['Geo Point.2'] = pd.to_numeric(zip_codes['Geo Point.2'], errors='coerce')
    zip_codes['Zip Code'] = zip_codes['Zip Code'].astype(str).str.zfill(5)
    zip_codes = zip_codes[['Zip Code', 'Geo Point.1', 'Geo Point.2']]
    zip_codes = zip_codes[
        zip_codes['Geo Point.1'].notnull() &
        zip_codes['Geo Point.2'].notnull() &
        zip_codes['Zip Code'].notnull() &
        (zip_codes['Zip Code'] != '') &
        (zip_codes['Geo Point.1'] != '') &
        (zip_codes['Geo Point.2'] != '')
    ].reset_index(drop=True)
    return zip_codes

def get_housing_data():
    housing_data = pd.read_csv('HousingData.csv')
    housing_data.columns = housing_data.iloc[0]
    housing_data = housing_data.drop(0)
    housing_data.reset_index(drop=True, inplace=True)
    housing_data.columns = [column.replace('Median (dollars)', 'Median dollars') for column in housing_data.columns]
    housing_data['zip'] = housing_data['Geographic Area Name'].str[-5:]
    housing = housing_data[[
        "zip",
        "Estimate!!ROOMS!!Total housing units!!Median rooms",
        "Estimate!!VALUE!!Owner-occupied units!!Median dollars",
        "Percent!!UNITS IN STRUCTURE!!Total housing units!!1-unit, detached",
        "Percent!!HOUSING TENURE!!Occupied housing units!!Renter-occupied",
        "Estimate!!HOUSING TENURE!!Occupied housing units!!Average household size of renter-occupied unit",
        "Estimate!!HOUSING TENURE!!Occupied housing units!!Average household size of owner-occupied unit"
    ]]
    housing = housing.replace({'-': np.nan})
    housing["Estimate!!ROOMS!!Total housing units!!Median rooms"] = housing["Estimate!!ROOMS!!Total housing units!!Median rooms"].replace({'9.0+': '9'})
    housing["Estimate!!VALUE!!Owner-occupied units!!Median dollars"] = housing["Estimate!!VALUE!!Owner-occupied units!!Median dollars"].replace({'2,000,000+': '2000000'})
    housing["Estimate!!VALUE!!Owner-occupied units!!Median dollars"] = housing["Estimate!!VALUE!!Owner-occupied units!!Median dollars"].replace({'10,000-': '10000'})
    housing[housing.columns.difference(['zip'])] = housing[housing.columns.difference(['zip'])].astype(float)
    housing['Household_Size'] = housing["Estimate!!HOUSING TENURE!!Occupied housing units!!Average household size of renter-occupied unit"] * housing["Percent!!HOUSING TENURE!!Occupied housing units!!Renter-occupied"] / 100 + housing["Estimate!!HOUSING TENURE!!Occupied housing units!!Average household size of owner-occupied unit"] * (1 - housing["Percent!!HOUSING TENURE!!Occupied housing units!!Renter-occupied"] / 100)
    housing = housing.drop(columns=["Estimate!!HOUSING TENURE!!Occupied housing units!!Average household size of renter-occupied unit", "Estimate!!HOUSING TENURE!!Occupied housing units!!Average household size of owner-occupied unit"])
    housing = housing.rename(columns={'zip': 'Zip Code'})
    housing_cols = [col for col in housing.columns if col != 'Zip Code']
    return housing, housing_cols

def prepare_inputs(user_data, zip_codes, housing, housing_cols):
    properties = user_data[['Seller Reference #', 'Zip Code']].merge(zip_codes, on='Zip Code', how='left')
    properties = properties.rename(columns={'Geo Point.1': 'Latitude', 'Geo Point.2': 'Longitude'})
    properties = properties[['Seller Reference #', 'Latitude', 'Longitude']]

    properties['key'] = 1
    zip_codes['key'] = 1
    prop_zip = properties.merge(zip_codes, on='key').drop('key', axis=1)

    def haversine(lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        miles = 3958.8 * c
        return miles

    prop_zip['DistanceMiles'] = haversine(
        prop_zip['Latitude'], prop_zip['Longitude'],
        prop_zip['Geo Point.1'], prop_zip['Geo Point.2']
    )

    filtered = prop_zip[prop_zip['DistanceMiles'] <= 5].copy()
    filtered = filtered[['Seller Reference #', 'Zip Code']]

    if 'Zip Code' in user_data.columns:
        storage_zip = pd.concat([
            filtered,
            user_data[['Seller Reference #', 'Zip Code']]
        ], ignore_index=True).drop_duplicates()
    else:
        storage_zip = filtered.copy()

    storage_zip = storage_zip.merge(housing, on='Zip Code', how='left')
    loc_zip_grouped = storage_zip.groupby('Seller Reference #')[housing_cols].mean().reset_index()
    user_data = user_data.merge(loc_zip_grouped, on='Seller Reference #', how='left')

    user_data.columns = [column.replace('!!', '_') for column in user_data.columns]
    user_data.columns = [column.replace('-', '_') for column in user_data.columns]
    user_data.columns = [column.replace('.', '') for column in user_data.columns]
    user_data.columns = [column.replace(' ', '') for column in user_data.columns]
    user_data.columns = [column.replace('/', '') for column in user_data.columns]
    user_data.columns = [column.replace(',', '') for column in user_data.columns]

    return user_data

def predict_occ_rate(user_data, model):
    occ_rate_preds = []

    for y in range(10):
        df_pred = user_data.copy()
        df_pred['years_of_growth'] = y
        if 'occ_rate_per_sqft' in df_pred.columns:
            X_pred = df_pred.drop(columns=['occ_rate_per_sqft'])
        else:
            X_pred = df_pred
        y_pred = model.predict(X_pred)
        df_pred['occ_rate_per_sqft_pred'] = y_pred
        df_pred['years_of_growth'] = y
        df_pred['growth'] = (df_pred['occ_rate_per_sqft_pred']/df_pred['first_occ_rate_per_sqft'])**(1/(y+1)) - 1
        occ_rate_preds.append(df_pred[['SellerReference#', 'years_of_growth', 'occ_rate_per_sqft_pred', 'growth']])

    occ_rate_pred_df = pd.concat(occ_rate_preds, ignore_index=True)
    occ_rate_pred_df = occ_rate_pred_df.merge(user_data, on='SellerReference#', how='left')
    
    # Calculate growth rate
    occ_rate_pred_df = occ_rate_pred_df.sort_values(['SellerReference#', 'years_of_growth'])
    # occ_rate_pred_df['growth'] = occ_rate_pred_df.groupby('SellerReference#')['occ_rate_per_sqft_pred'].pct_change()
    
    return occ_rate_pred_df

# Get user inputs (this will always display the sidebar)
user_data = get_user_inputs()

# Button to trigger prediction
if st.sidebar.button('Predict'):
    st.session_state.show_predictions = True

# Show predictions if button was clicked
if st.session_state.show_predictions:
    with st.spinner('Processing predictions...'):
        zip_codes = get_zip_codes()
        housing, housing_cols = get_housing_data()
        input_data = prepare_inputs(user_data, zip_codes, housing, housing_cols)
        linear_reg_occ_rate_model = joblib.load('linear_regression_occ_rate_model.pkl')
        pred_data = predict_occ_rate(input_data, linear_reg_occ_rate_model)
        
        # Calculate and display CAGR
        avg_growth = pred_data.loc[pred_data['years_of_growth'] == pred_data['years_of_growth'].max(), 'growth'].mean()
        st.metric(label="CAGR", value=f"{avg_growth:.2%}")
        
        # Format 'growth' column as percentage with 2 decimal points for display
        display_df = pred_data[['years_of_growth', 'occ_rate_per_sqft_pred', 'growth']].copy()
        display_df['growth'] = display_df['growth'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
        st.write(display_df)

# cd 'C:\Users\jrandall\Storage Rentals of America\Operations - Revenue Management\231117 - Move-In-Projections'
# python -m streamlit run occ_rate_proj.py