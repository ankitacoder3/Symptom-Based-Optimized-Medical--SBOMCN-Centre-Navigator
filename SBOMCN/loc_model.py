import pandas as pd
from geopy.distance import geodesic

def recommend_hospital():
    # Load the dataset
    df = pd.read_csv('hospital_data_new_merged_15_11.csv')  # Replace with your dataset file name
    print("After loading data")
    print(df)

    # Step 2: Get user input (disease name and option)
    disease_name = input("Enter the disease name: ")
    print("Select option:")
    print("1. Use current location")
    print("2. Enter place name and pincode")

    while True:
        option = input("Option: ")
        if option in ['1', '2']:
            break
        else:
            print("Invalid option. Please enter 1 or 2.")

    if option == '1':
        try:
            import geocoder
            location = geocoder.ip('me')
            user_lat, user_lon = location.latlng
            user_place = f"Current Location (Latitude: {user_lat}, Longitude: {user_lon})"

            # Check if the user's location is within India (using a rough bounding box)
            if not (6.5 <= user_lat <= 35.5 and 68.7 <= user_lon <= 97.25):
                print("This application is applicable for Indian residents only.")
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
        while True:
            place_name = input("Enter the place name: ")
            pincode = input("Enter the pincode: ")

            # Validate pincode (you may need to add more checks based on your data)
            if pincode.isdigit() and len(pincode) == 6:
                break
            else:
                print("Invalid pincode. Please enter a valid 6-digit pincode.")

        user_lat, user_lon = None, None
        user_place = f"{place_name} (PIN: {pincode})"

        if 'Latitude' not in df.columns or 'Longitude' not in df.columns:
            print("Latitude and Longitude columns not found in the dataset.")
            return []

        try:
            # Get the latitude and longitude from the dataset
            location_data = df.loc[(df['Address'] == place_name) & (df['PIN'] == int(pincode)),
                                    ['Latitude', 'Longitude']].dropna()

            if location_data.empty:
                print(f"Location not found for {place_name} (PIN: {pincode}).")
                return []

            user_lat, user_lon = location_data.iloc[0]['Latitude'], location_data.iloc[0]['Longitude']

            # Check if the entered place name is in India
            if pd.isna(user_lat) or pd.isna(user_lon) or not (6.5 <= user_lat <= 35.5 and 68.7 <= user_lon <= 97.25):
                print("This application is applicable for Indian residents only.")
                return []

        except Exception as e:
            print(f"Error getting location data: {e}")
            return []

        # Calculate distance
        df['Distance'] = df.apply(
            lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).kilometers,
            axis=1
        )
        print("After Calculating distances")

    else:
        print("Invalid option selected. Exiting...")
        return []

    # Filter hospitals based on disease name and pincode within a certain distance
    max_distance_km = 1000  # You can adjust this distance as needed
    nearest_hospitals = df[
        (df['Disease'].str.contains(disease_name, case=False)) &
        (df['Distance'] <= max_distance_km)
    ].nsmallest(5, 'Distance')

    print(f"User's Location: {user_place}")
    print(" ")
    print("-------------------------------------------------------")

    print(f"Recommended Hospitals for {disease_name} in {user_place}:")
    for index, row in nearest_hospitals.iterrows():
        print(f"Hospital Name: {row['Hospital Name']}")
        print(f"Latitude: {row['Latitude']}")
        print(f"Longitude: {row['Longitude']}")
        print(f"Distance: {row['Distance']} km")

        # Assuming the address is in the 'Address' column
        print(f"Address: {row['Address']}")
        print("-" * 30)

    # Return the list of recommended hospitals
    return nearest_hospitals.to_dict('records')

# Call the function
result = recommend_hospital()

# Print the result
print("Function Result:")
print(result)
