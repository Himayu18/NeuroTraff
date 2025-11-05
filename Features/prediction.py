from src.components.data_transformation import ColumnRemover,LabelEncoderTransformer,TimestampTransformer,DelayTransformer,CoordinatesTransformer,TrafficLevelTransformer,DataTransformer

from flask import Flask, request, jsonify, render_template
import requests
from datetime import datetime, timezone
import os
import json
import pandas as pd
import joblib
import uuid

API_KEY = 'cpZXnWw9uzrumJ7scFawXV328QM7F8NP'
if not API_KEY:
    raise Exception("ERROR: Missing API_KEY environment variable.")

base_url = "https://api.tomtom.com"
version_number = "4"
style = "absolute"
zoom = "10"
response_format = "json"
unit = "KMPH"
thickness = "2"
open_lr = "false"

road_points = {
   "SH-42 (Ghodbunder Road)": [
        "19.2500,73.0500",  # Kapurbawdi approx.
        "19.2500,72.9800"   # Near Ghodbunder junction
    ],
    "Eastern Express Highway": [
        "19.2183,72.9781",  # Thane approx.
        "19.0760,72.8777"   # Mumbai approx.
    ],
    "LBS Marg": [
        "19.1960,72.9600",  # Thane start
        "19.0500,72.8700"   # Sion approx.
    ],
    "Thane-Belapur Road": [
        "19.1800,73.0100",  # Kalwa
        "19.0450,73.0150"   # Navi Mumbai (Turbhe)
    ],
    "Sion-Panvel Expressway": [
        "19.0600,72.8850",  # Sion
        "18.9800,73.1100"   # Kalamboli
    ],
    "Mumbai-Nashik Expressway (NH 3)": [
        "19.2183,72.9781",  # Thane start
        "20.0110,73.7900"   # Nashik approx.
    ],
    "MDR 64": [
        "19.4100,73.1800",  # Murbad
        "19.4300,73.4800"   # Shahapur
    ],
    "SH 40": [
        "19.3200,73.1000",  # Shilphata
        "19.3000,73.1300"   # Bhiwandi
    ],
    "MDR 62": [
        "19.2200,73.0000"   # Local villages approx.
    ],
    "Ring Road (Kalyan-Dombivli)": [
        "19.2400,73.1300",  # Kalyan
        "19.2200,73.1000"   # Dombivli
    ],
    "Multi-Modal Corridor": [
        "18.6500,72.8700",  # Alibaug
        "19.4200,72.8400"   # Virar
    ],
     "Mumbai-Agra Road (NH3/NH160)": [
        "19.2183,72.9781",  # Thane approx.
        "20.0110,73.7900"   # Nashik approx.
    ],
    "Kalyan-Shilphata Road": [
        "19.2500,73.0500",  # Kalyan
        "19.2500,72.9800"   # Near Shilphata
    ],
    "Kalyan-Bhiwandi Road": [
        "19.2300,73.0800",  # Kalyan
        "19.2500,73.0100"   # Bhiwandi
    ],
    "Kalyan-Badlapur Road": [
        "19.2200,73.0500",  # Kalyan
        "19.3000,73.0800"   # Badlapur
    ],
    "Dombivli-Manpada Road": [
        "19.2200,73.0700",  # Dombivli
        "19.2400,73.1000"   # Manpada
    ],
    "Bhiwandi Bypass Road": [
        "19.2500,73.0500",  # Bhiwandi
        "19.2500,73.0100"   # Near bypass
    ],
    "Airoli-Thane Creek Bridge Road": [
        "19.1800,72.9800",  # Airoli
        "19.2200,72.9500"   # Thane
    ],
    "Kalyan-Karjat Road": [
        "19.2100,73.0600",  # Kalyan
        "19.0400,73.2400"   # Karjat
    ],
    "Mumbra Bypass Road": [
        "19.2500,73.0300",  # Mumbra
        "19.2400,73.0100"   # Ghodbunder Road junction
    ],
    "Dombivli-Kalyan Link Road": [
        "19.2300,73.0800",  # Dombivli
        "19.2200,73.0700"   # Kalyan
    ]
}




def fetch_traffic_data(point):
    url = (
        f"{base_url}/traffic/services/{version_number}/flowSegmentData/"
        f"{style}/{zoom}/{response_format}?"
        f"key={API_KEY}&point={point}&unit={unit}&thickness={thickness}"
        f"&openLr={open_lr}&jsonp="
    )
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def predict_real_time_traffic(road):
    now = datetime.now(timezone.utc)
    hour = now.hour
    if not (5 <= hour < 24):
        return jsonify({"error": "Outside allowed hours for fetching traffic data"}), 403

    road = road.strip() if road else None
    points = road_points.get(road)
    if not points:
        return jsonify({"error": "Unknown road selected"}), 400
    
    results = []
    try:
        for pt in points:
            response_data = fetch_traffic_data(pt)
            flow_segment = response_data.get('flowSegmentData', {})

            record = {
                "_id": str(uuid.uuid4()),
                "road": road,
                "point": pt,
                "timestamp": now.isoformat(),
                "roadName": flow_segment.get('roadName'),
                "frc": flow_segment.get('frc'),
                "currentSpeed": flow_segment.get('currentSpeed'),
                "freeFlowSpeed": flow_segment.get('freeFlowSpeed'),
                "currentTravelTime": flow_segment.get('currentTravelTime'),
                "freeFlowTravelTime": flow_segment.get('freeFlowTravelTime'),
                "confidence": flow_segment.get('confidence'),
                "roadClosure": flow_segment.get('roadClosure'),
            }
            results.append(record)
            # print(results)
        df = pd.DataFrame(results)

        transformer = joblib.load(r"artifacts\Piplines\traffic_pipeline.pkl")
        model = joblib.load(r"artifacts\Piplines\best_model.pkl")
        X_transformed = transformer.transform(df)
        delay_values = X_transformed["Delay"].values
        average_delay = delay_values.mean()
        X_transformed = X_transformed.drop(columns=["delay ratio","Delay","Traffic level"])
        
        prediction = model.predict(X_transformed)


        label_encoder = transformer.named_steps['label_encode_traffic'].le
        decoded_predictions = label_encoder.inverse_transform(prediction)

        
        priority = {"low": 1, "medium": 2, "high": 3}
        decoded_predictions = [val.lower() for val in decoded_predictions]
        worst = max(decoded_predictions, key=lambda x: priority.get(x, 0))

        traffic_level = worst.capitalize()
        # print(traffic_level)

        # print(average_delay)
        if traffic_level == "Medium":
    # Add about +1 minute (60s) and scale up a bit
            average_delay = average_delay * 1.25 + 60  

        elif traffic_level == "High":
    # Ensure a minimum baseline delay (e.g. at least 3 minutes)
            if average_delay < 180:  # 180 seconds
                average_delay = 180
    # Amplify and add a bit more
            average_delay = average_delay * 1.45 + 120  # add 2 minutes extra

        final_delay_minutes = max(1, round(average_delay / 60))  
        return traffic_level, final_delay_minutes



    except Exception as e:
        print(f"Error fetching data for {road}: {e}")
        return jsonify({"error": str(e)}), 500