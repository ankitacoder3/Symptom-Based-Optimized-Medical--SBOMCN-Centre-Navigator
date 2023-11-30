# hospital_recommendation.py
from loc_hos_recommendation import recommend_hospital

# Example 1: User's current location
result1 = recommend_hospital(disease_name='Cystic Fibrosis', option='1')
print("Example 1 Result:")
print(result1)

# Example 2: User's specified location
result2 = recommend_hospital(disease_name='Cystic Fibrosis', option='2', place_name='Chennai', pincode='600100')
print("Example 2 Result:")
print(result2)