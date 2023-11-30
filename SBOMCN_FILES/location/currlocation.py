import pandas as pd
from opencage.geocoder import OpenCageGeocode
from geopy.distance import geodesic

# Load the ground truth dataset (assuming it's in CSV format)
#ground_truth_df = pd.read_csv('ground_truth.csv')  # Replace with your ground truth file name

# Step 1: Load the dataset
df = pd.read_csv('hospital_data.csv')  # Replace with your dataset file name

# Get user input (disease name and option)
disease_name = input("Enter the disease name: ")
print("Select option:")
print("1. Use current location")
print("2. Enter place name and pincode")
option = input("Option: ")

if option == '1':
    try:
        import geocoder
        location = geocoder.ip('me')
        user_lat, user_lon = location.latlng
        user_place = f"Current Location (Latitude: {user_lat}, Longitude: {user_lon})"
    except Exception as e:
        print(f"Error getting user's location: {e}")
        exit()
elif option == '2':
    place_name = input("Enter your location (place name): ")
    pincode = input("Enter your pincode: ")
    user_lat, user_lon = None, None
    user_place = f"{place_name} (PIN: {pincode})"
else:
    print("Invalid option selected. Exiting...")
    exit()

print(f"User's Location: {user_place}")
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
        return results[0]['geometry']['lat'], results[0]['geometry']['lng']
    else:
        print(f"Could not find coordinates for {place_name}")
        return None, None

# Assuming 'PIN' is the column name in your dataset
df['Latitude'], df['Longitude'] = zip(*df['PIN'].apply(lambda x: get_coordinates_from_PIN(x)))

# Step 4: Get coordinates for user's location
def get_coordinates(input_value):
    if input_value.isnumeric():
        return get_coordinates_from_PIN(input_value)
    else:
        return get_coordinates_from_place(input_value)

user_lat, user_lon = get_coordinates(place_name)

if user_lat is None or user_lon is None:
    print(f"Could not find coordinates for {place_name}")
    exit()

# Step 5: Calculate distance
df['Distance'] = df.apply(lambda row: geodesic((user_lat, user_lon), (row['Latitude'], row['Longitude'])).kilometers, axis=1)

# Step 6: Filter hospitals based on disease name and pincode
nearest_hospitals = df[(df['Disease'] == disease_name) & (df['PIN'] == int(pincode))].nsmallest(5, 'Distance')


# Compare recommended hospitals with ground truth

#ground_truth_hospitals = ground_truth_df[(ground_truth_df['Disease'] == disease_name) & (ground_truth_df['Location'] == place_name)]

#correct_recommendations = 0
#for index, row in nearest_hospitals.iterrows():
    #if any((row['Hospital Name'] == gt_row['Recommended Hospital']) for _, gt_row in ground_truth_hospitals.iterrows()):
        #correct_recommendations += 1

#accuracy = (correct_recommendations / min(len(nearest_hospitals), len(ground_truth_hospitals))) * 100


print(f"Recommended Hospitals for {disease_name} in {place_name} (PIN: {pincode}):")
for index, row in nearest_hospitals.iterrows():
    print(f"Hospital Name: {row['Hospital Name']}")
    print(f"Latitude: {row['Latitude']}")
    print(f"Longitude: {row['Longitude']}")
    print(f"Distance: {row['Distance']} km")
    print("-" * 30)

#print(f"Accuracy: {accuracy:.2f}%")