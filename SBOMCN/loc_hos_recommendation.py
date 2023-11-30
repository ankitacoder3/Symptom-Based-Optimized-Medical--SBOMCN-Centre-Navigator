import pandas as pd
from geopy.distance import geodesic

def recommend_hospital(disease_name, option, place_name=None, pincode=None):
    # Load the dataset
    df = pd.read_csv('hospital_data_new_merged_15_11.csv')  # Replace with your dataset file name

    if option == '1':
        try:
            import geocoder
            location = geocoder.ip('me')
            user_lat, user_lon = location.latlng

            # Check if the user's location is within India
            if not (6.5 <= user_lat <= 35.5 and 68.7 <= user_lon <= 97.25):
                return []

        except Exception as e:
            print(f"Error getting user's location: {e}")
            return []

        # Calculate distance
        df['Distance'] = df.apply(
            lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).kilometers,
            axis=1
        )

    elif option == '2':
        # Validate pincode (you may need to add more checks based on your data)
        if not (pincode.isdigit() and len(pincode) == 6):
            return []

        user_lat, user_lon = None, None

        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            return []

        try:
            # Get the latitude and longitude from the dataset
            location_data = df.loc[(df['Address'] == place_name) & (df['PIN'] == int(pincode)),
                                    ['Latitude', 'Longitude']].dropna()

            if location_data.empty:
                return []

            user_lat, user_lon = location_data.iloc[0]['Latitude'], location_data.iloc[0]['Longitude']

            # Check if the entered place name is in India
            if pd.isna(user_lat) or pd.isna(user_lon) or not (6.5 <= user_lat <= 35.5 and 68.7 <= user_lon <= 97.25):
                return []

        except Exception as e:
            print(f"Error getting location data: {e}")
            return []

        # Calculate distance
        df['Distance'] = df.apply(
            lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).kilometers,
            axis=1
        )

    else:
        return []

    # Filter hospitals based on disease name and pincode within a certain distance
    max_distance_km = 1000  # You can adjust this distance as needed
    nearest_hospitals = df[
        (df['Disease'].str.contains(disease_name, case=False)) &
        (df['Distance'] <= max_distance_km)
    ].nsmallest(5, 'Distance')

    return nearest_hospitals.to_dict('records')