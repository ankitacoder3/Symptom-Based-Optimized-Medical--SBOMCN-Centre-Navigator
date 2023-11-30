import pandas as pd
from opencage.geocoder import OpenCageGeocode
from geopy.distance import geodesic

# Step 1: Load the dataset
df = pd.read_csv('hospital_data.csv')  # Replace with your dataset file name

# Step 2: Get user input (disease name, place name, and pincode)
disease_name = input("Enter the disease name: ")
place_name = input("Enter your location (place name): ")
print(" ")
print("-------------------------------------------------------")

# Step 3: Convert place name to coordinates using geocoding
def get_coordinates_from_PIN(PIN):
    key = '017b9a6a6b6a412eba07fda48452578c'  # Replace with your OpenCageData API key
    geocoder = OpenCageGeocode(key)

    results = geocoder.geocode(f'PIN {PIN}')

    if results and len(results):
        return results[0]['geometry']['lat'], results[0]['geometry']['lng']
    else:
        print(f"Could not find coordinates for PIN {PIN}")
        return None, None

def get_coordinates_from_place(place_name):
    key = '017b9a6a6b6a412eba07fda48452578c'  # Replace with your OpenCageData API key
    geocoder = OpenCageGeocode(key)

    results = geocoder.geocode(place_name)

    if results and len(results):
        lat = results[0]['geometry']['lat']
        lon = results[0]['geometry']['lng']
        country = results[0]['components'].get('country')
        return lat, lon, country
    else:
        print(f"Could not find coordinates for {place_name}")
        return None, None, None

# Assuming 'PIN' is the column name in your dataset
df['Latitude'], df['Longitude'] = zip(*df['PIN'].apply(lambda x: get_coordinates_from_PIN(x)))

# Step 4: Get coordinates for user's location
def get_coordinates(input_value):
    if input_value.isnumeric():
        return get_coordinates_from_PIN(input_value)
    else:
        lat, lon, country = get_coordinates_from_place(input_value)
        if country and country != 'India':
            return None, None, country
        return lat, lon, country

user_lat, user_lon, user_country = get_coordinates(place_name)

if user_lat is None or user_lon is None:
    if user_country and user_country != 'India':
        print(f"Sorry, this application is applicable for Indian residents only.")
    else:
        print(f"Could not find coordinates for {place_name}")
    exit()

# Step 5: Calculate distance
df['Distance'] = df.apply(lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).kilometers, axis=1)

# Step 6: Filter hospitals based on disease name and pincode (if applicable)
if user_country == 'India':
    pincode = input("Enter your pincode: ")
    nearest_hospitals = df[(df['Disease'] == disease_name) & (df['PIN'] == int(pincode))].nsmallest(5, 'Distance')
else:
    nearest_hospitals = df[df['Disease'] == disease_name].nsmallest(5, 'Distance')

print(f"Recommended Hospitals for {disease_name} in {place_name}:")
for index, row in nearest_hospitals.iterrows():
    print(f"Hospital Name: {row['Hospital Name']}")
    print(f"Latitude: {row['Latitude']}")
    print(f"Longitude: {row['Longitude']}")
    print(f"Distance: {row['Distance']} km")
    print("-" * 30)
