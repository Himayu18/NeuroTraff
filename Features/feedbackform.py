from flask import request
from pymongo import MongoClient


client = MongoClient("mongodb+srv://himayudhoke:1X5idC51cKy8EntW@cluster0.lufwdkw.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
database = client['feedbacks']
collection = database['user_feedback']

def get_response(data):
    Name = data.get('name')
    Roadname = data.get('road')
    Traffic_Condition = data.get('trafficCondition')
    Delay = data.get('delay')
    Weather = data.get('weather')
    Description = data.get('description')
    Rating = data.get('rating')

    feedback_data = {
        "UserName":Name,
        "Roadname":Roadname,
        "Traffic_condition":Traffic_Condition,
        "Delay":Delay+"mins",
        "Weather":Weather,
        "Description":Description,
        "Rating":Rating+"star"
    }
    result = collection.insert_one(feedback_data)
    print(result.inserted_id)
