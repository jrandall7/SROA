
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import datetime
from sklearn.metrics.pairwise import euclidean_distances
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

# Load configuration from YAML file (or create default)
try:
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=SafeLoader)
except FileNotFoundError:
    st.error("config.yaml file not found. Please create it with user credentials.")
    st.stop()

# Initialize authenticator
authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

# Render login widget
authenticator.login()

# Check if user is authenticated
if st.session_state["authentication_status"]:
    st.set_page_config(layout="wide")
    col1, col2 = st.columns([20, 1])
    with col1:
        st.title("In-place Rate Projections")
    with col2:
        authenticator.logout(location='main')
    

    # Initialize session state for storing whether to show predictions
    if 'show_predictions' not in st.session_state:
        st.session_state.show_predictions = False

    def get_user_inputs():
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
        with col1:
            first_occ_rate_per_sqft = st.number_input("In Place SS Rent/PSF", min_value=0.0, max_value=10.0, value=1.0, step=0.01, key='first_occ')
        with col2:
            MHHI = st.number_input("5 Mile HHI", min_value=0, max_value=1000000, value=65000, step=1, key='mhhi')
        with col3:
            Pop = st.number_input("5 Mile Population", min_value=0, max_value=1000000, value=50000, step=1, key='pop')
        with col4:
            office = st.checkbox("Is Office?", value=True, key='office')
        with col5:
            Supply = st.number_input("5 Mile SS RSF/Capita", min_value=0.0, max_value=100.0, value=10.0, step=0.01, key='supply')
        with col6:
            perc_cc = st.number_input("Percent CC", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key='perc_cc')
            perc_ncc = 1 - perc_cc
        with col7:
            zip_code = st.text_input("Zip Code", value="33401", max_chars=5, key='zip_code')
        with col8:  
            # Button to trigger prediction
            if st.button('Predict'):
                st.session_state.show_predictions = True
        zip_code = zip_code.zfill(5)
        seller_reference = 1
        user_inputs = pd.DataFrame({
            'Seller Reference #': [seller_reference],
            'first_occ_rate_per_sqft': [first_occ_rate_per_sqft],
            'MHHI': [MHHI],
            'Pop': [Pop],
            'Office': [office],
            'Supply': [Supply],
            'perc_cc': [perc_cc],
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
        # Ensure Zip Code is string type in both dataframes
        user_data['Zip Code'] = user_data['Zip Code'].astype(str).str.zfill(5)
        zip_codes['Zip Code'] = zip_codes['Zip Code'].astype(str).str.zfill(5)
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

        for y in range(1, 11):
            df_pred = user_data.copy()
            df_pred['years_of_growth'] = y-0.5
            current_year = datetime.datetime.today().year
            df_pred['Year'] = current_year + y
            if 'occ_rate_per_sqft' in df_pred.columns:
                X_pred = df_pred.drop(columns=['occ_rate_per_sqft'])
            else:
                X_pred = df_pred
            y_pred = model.predict(X_pred)
            df_pred['occ_rate_per_sqft_pred'] = y_pred
            df_pred['growth'] = (df_pred['occ_rate_per_sqft_pred']/df_pred['first_occ_rate_per_sqft'])**(1/(y)) - 1
            occ_rate_preds.append(df_pred[['SellerReference#', 'Year', 'occ_rate_per_sqft_pred', 'growth']])

        occ_rate_pred_df = pd.concat(occ_rate_preds, ignore_index=True)
        occ_rate_pred_df = occ_rate_pred_df.merge(user_data, on='SellerReference#', how='left')
        
        occ_rate_pred_df = occ_rate_pred_df.sort_values(['SellerReference#', 'Year'])
        occ_rate_pred_df['growth YoY'] = occ_rate_pred_df.groupby('SellerReference#')['occ_rate_per_sqft_pred'].pct_change()
        
        return occ_rate_pred_df

    def create_waterfall_chart(pipeline_model, user_data):
        st.subheader("Prediction Breakdown (Vertical Waterfall)")
        current_year = datetime.datetime.today().year
        year = st.slider("Year", min_value=current_year+1, max_value=current_year +10, value=current_year+4, step=1)
        user_data['Year'] = year
        user_data['years_of_growth'] = year - current_year - 0.5
        preprocessor = pipeline_model.named_steps['preprocessor']
        model = pipeline_model.named_steps['regressor']
        feature_names = preprocessor.transformers_[0][2]

        X_user = user_data[feature_names].astype(float)
        X_scaled = preprocessor.transform(X_user)

        coef = model.coef_
        contributions = X_scaled[0] * coef
        intercept = model.intercept_
        prediction = intercept + np.sum(contributions)

        sorted_idx = np.argsort(np.abs(contributions))
        top_features = [feature_names[i] for i in sorted_idx]
        top_contributions = contributions[sorted_idx]

        axis_label_map = {
            'Property_#': 'Property #',
            'occ_rate_per_sqft': 'Occ Rate/PSF',
            'first_occ_rate_per_sqft': 'First Occ Rate/PSF',
            'years_of_growth': 'Years of Growth',
            'perc_cc': '% CC',
            'Estimate_VALUE_Owner_occupiedunits_Mediandollars': 'Median House Price',
            'Estimate_ROOMS_Totalhousingunits_Medianrooms': 'Median Rooms per Household',
            'Household_Size': 'Household Size'
        }
        top_features_renamed = [axis_label_map.get(f, f) for f in top_features]

        colors = ['green' if x > 0 else 'red' for x in top_contributions]

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.barh(top_features_renamed, top_contributions, color=colors, alpha=0.8)
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Contribution to Prediction')
        ax.set_ylabel('Feature')
        ax.set_title(f'Top Factors Influencing Prediction\nPredicted Value: {prediction:.3f}')
        plt.tight_layout()
        st.pyplot(fig)

    def get_similar_training_rows(pipeline_model, user_data, training_data, n_neighbors=10):
        preprocessor = pipeline_model.named_steps['preprocessor']
        model = pipeline_model.named_steps['regressor']
        feature_names = preprocessor.transformers_[0][2]

        excluded_features = ['Year', 'years_of_growth','first_occ_rate_per_sqft']
        included_features = [f for f in feature_names if f not in excluded_features]

        X_train_all = preprocessor.transform(training_data[feature_names])
        X_user_all = preprocessor.transform(user_data[feature_names])

        X_train_df = pd.DataFrame(X_train_all, columns=feature_names)
        X_user_df = pd.DataFrame(X_user_all, columns=feature_names)

        X_train = X_train_df[included_features]
        X_user = X_user_df[included_features]

        coef_df = pd.DataFrame({'feature': feature_names, 'coef': np.abs(model.coef_)})
        coef_df = coef_df[coef_df['feature'].isin(included_features)]
        coef_weights = coef_df['coef'].values
        coef_weights = coef_weights / coef_weights.sum()

        X_train_weighted = X_train * coef_weights
        X_user_weighted = X_user * coef_weights

        distances = euclidean_distances(X_user_weighted, X_train_weighted)[0]

        training_data_copy = training_data.copy()
        training_data_copy['Similarity Score'] = distances
        nearest = training_data_copy.nsmallest(n_neighbors, 'Similarity Score')
        nearest['CAGR'] = (nearest['occ_rate_per_sqft']/nearest['first_occ_rate_per_sqft'])**(1/nearest['years_of_growth']) - 1
        return nearest

    tab1, tab2 = st.tabs(["Single Property", "Batch Upload"])
    with tab1:
        # Get user inputs
        user_data = get_user_inputs()

        # Show predictions if button was clicked
        if st.session_state.show_predictions:
            with st.spinner('Processing predictions...'):
                zip_codes = get_zip_codes()
                housing, housing_cols = get_housing_data()
                input_data = prepare_inputs(user_data, zip_codes, housing, housing_cols)
                linear_reg_occ_rate_model = joblib.load('linear_regression_occ_rate_model.pkl')
                pred_data = predict_occ_rate(input_data, linear_reg_occ_rate_model)
                
                avg_growth = pred_data.loc[pred_data['Year'] == pred_data['Year'].max(), 'growth'].mean()
                avg_growth_yoy = pred_data['growth YoY'].mean()
                col1, col2, _ = st.columns([1, 1, 8])
                with col1:
                    st.metric(label="Final CAGR", value=f"{avg_growth:.2%}")
                with col2:
                    st.metric(label="Average YoY Growth", value=f"{avg_growth_yoy:.2%}")
                
                display_df = pred_data[['Year', 'occ_rate_per_sqft_pred', 'growth', 'growth YoY']].copy()
                display_df['growth'] = display_df['growth'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                display_df['growth YoY'] = display_df['growth YoY'].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "")
                display_df = display_df.rename(columns={'occ_rate_per_sqft_pred':'In Place Rate/PSF', 'growth':'CAGR by Year', 'growth YoY':'YoY Growth'})
                st.write(display_df)
                create_waterfall_chart(linear_reg_occ_rate_model, input_data)
                training_data = pd.read_csv('occ_rate_model_train_data.csv')
                similar_rows = get_similar_training_rows(linear_reg_occ_rate_model, input_data, training_data)
                st.subheader("Most Similar Current Properties")
                st.write(similar_rows[['Property_#','Similarity Score','occ_rate_per_sqft','first_occ_rate_per_sqft','CAGR','years_of_growth','MHHI','Pop','Office','Supply','perc_cc',
                                        'Estimate_VALUE_Owner_occupiedunits_Mediandollars','Estimate_ROOMS_Totalhousingunits_Medianrooms','Household_Size']].head(10).rename(
                                        columns={'Property_#':'Property #','occ_rate_per_sqft':'Occ Rate/PSF','first_occ_rate_per_sqft':'First Occ Rate/PSF',
                                        'years_of_growth':'Years of Growth','perc_cc':'% CC','Estimate_VALUE_Owner_occupiedunits_Mediandollars':'Median House Price',
                                        'Estimate_ROOMS_Totalhousingunits_Medianrooms':'Median Rooms per Household','Household_Size':'Household Size'}))
    with tab2:
        
        #Example File
        example = pd.DataFrame({
                    'Seller Reference #': ['A', 'B'],
                    'In Place SS Rent/PSF': [1.0, 1.2],
                    '5 Mile HHI': [65000, 75000],
                    '5 Mile Population': [25000, 75000],
                    'Office ': [True, False],
                    '5 Mile SS RSF/Capita': [5.0, 15.0],
                    'Percent CC': [0.0, 0.75],
                    'Zip Code': [33401, 76501]
                })
        # Option to download results
        excel = example.to_csv(index=False)
        col1, col2, _ = st.columns([1, 2, 2])
        with col1:
            st.download_button(
                label="Download Example Format as CSV",
                data=excel,
                file_name="example.csv",
                mime="text/csv"
            )
        with col2:
            uploaded_file = st.file_uploader("Upload CSV file with Property Data", type=['csv'])
        
        if uploaded_file is not None:
            try:
                # Read the Excel file
                batch_data = pd.read_csv(uploaded_file)
                
                st.write("**Uploaded Data Preview:**")
                st.write(batch_data.head())
                
                batch_data = batch_data.rename(columns={
                    'In Place SS Rent/PSF': 'first_occ_rate_per_sqft',
                    '5 Mile HHI': 'MHHI',
                    '5 Mile Population': 'Pop',
                    '5 Mile SS RSF/Capita': 'Supply',
                    'Percent CC': 'perc_cc'
                })

                if st.button('Process Batch Predictions'):
                    with st.spinner('Processing batch predictions...'):
                        # Load necessary data
                        zip_codes = get_zip_codes()
                        housing, housing_cols = get_housing_data()
                        linear_reg_occ_rate_model = joblib.load('linear_regression_occ_rate_model.pkl')
                        # Prepare input data
                        batch_input_data = prepare_inputs(batch_data, zip_codes, housing, housing_cols)
                        # Get predictions for each row
                        all_predictions = []
                        for idx, row in batch_input_data.iterrows():
                            single_row = batch_input_data.iloc[[idx]]
                            pred_data = predict_occ_rate(single_row, linear_reg_occ_rate_model)
                            all_predictions.append(pred_data)
                        # Combine all predictions
                        combined_pred_data = pd.concat(all_predictions, ignore_index=True)
                        combined_pred_data = combined_pred_data.rename(columns={
                            'occ_rate_per_sqft_pred':'Future In Place SS Rate/PSF',
                            'first_occ_rate_per_sqft':'In Place SS Rent/PSF',
                            'growth':'CAGR by Year',
                            'growth YoY':'YoY Growth',
                            'MHHI': '5 Mile HHI',
                            'Pop': '5 Mile Population',
                            'Supply': '5 Mile SS RSF/Capita',
                            'perc_cc': 'Percent CC',
                            'ZipCode': 'Zip Code'})
                        combined_pred_data['CAGR by Year'] = (combined_pred_data['CAGR by Year']*100).map('{:.1f}%'.format)
                        combined_pred_data['YoY Growth'] = (combined_pred_data['YoY Growth']*100).map('{:.1f}%'.format)
                        # Display results
                        st.write(combined_pred_data[['SellerReference#', 'Year', 'Future In Place SS Rate/PSF','CAGR by Year','YoY Growth', 'In Place SS Rent/PSF', '5 Mile HHI', '5 Mile Population', '5 Mile SS RSF/Capita', 'Percent CC', 'Zip Code']])
                        
                        # # Option to download results
                        # csv = result_df.to_csv(index=False)
                        # st.download_button(
                        #     label="Download Results as CSV",
                        #     data=csv,
                        #     file_name="batch_predictions.csv",
                        #     mime="text/csv"
                        # )
            except Exception as e:
                st.error(f"Error processing file: {str(e)}")

elif st.session_state["authentication_status"] is False:
    st.error('Username/password is incorrect')
elif st.session_state["authentication_status"] is None:
    st.warning('Please enter your username and password')

# cd 'C:\Users\jrandall\Storage Rentals of America\Operations - Revenue Management\231117 - Move-In-Projections'
# python -m streamlit run occ_rate_proj.py