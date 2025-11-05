import requests
import pandas as pd
import uuid
import joblib
from datetime import datetime, timezone
from flask import Flask, request, jsonify, render_template
from src.components.data_transformation import ColumnRemover,LabelEncoderTransformer,TimestampTransformer,DelayTransformer,CoordinatesTransformer,TrafficLevelTransformer,DataTransformer

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

def analyze_whatif(data):
    Roadname = data.get('road')
    time_of_day = data.get('time_of_day')  # Hour (0-23)
    is_rain = data.get('is_rain')  # Boolean or None
    vehicle_volume = data.get('vehicle_volume')  # 50-200% or None
    has_accident = data.get('has_accident')  # Boolean or None
    accident_severity = data.get('accident_severity')  # 'minor', 'moderate', 'severe' or None
    day_type = data.get('day_type')  # 'weekday', 'weekend', 'holiday' or None

    Roadname = Roadname.strip() if Roadname else None
    points = road_points.get(Roadname)

    # Handle time_of_day - default to current hour if None
    if time_of_day is None:
        time_of_day = datetime.now(timezone.utc).hour
    else:
        time_of_day = max(0, min(23, int(time_of_day)))

    results = []
    try:
        for pt in points:
            response_data = fetch_traffic_data(pt)
            flow_segment = response_data.get('flowSegmentData', {})

            record = {
                "_id": str(uuid.uuid4()),
                "road": Roadname,
                "point": pt,
                "timestamp": datetime.now(timezone.utc).isoformat(),  # Will be updated
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
        
        df = pd.DataFrame(results)
        
        # ===== MODIFY TIMESTAMP =====
        base_date = datetime.now(timezone.utc).date()
        specified_time = datetime.combine(base_date, datetime.min.time()).replace(
            hour=time_of_day, 
            minute=0, 
            second=0, 
            microsecond=0,
            tzinfo=timezone.utc
        )
        df['timestamp'] = specified_time.isoformat()
        
        # ===== CALCULATE SPEED REDUCTION FACTOR =====
        speed_reduction_factor = 1.0
        
        # 1. RAIN: 25% reduction
        if is_rain is not None and is_rain:
            speed_reduction_factor *= 0.75
        
        # 2. VEHICLE VOLUME: 50% to 200%
        if vehicle_volume is not None:
            vehicle_volume = max(50, min(200, int(vehicle_volume)))
            
            if vehicle_volume <= 100:
                # Below capacity: slight speed increase
                volume_factor = (vehicle_volume - 50) / 50
                speed_adjustment = 1.1 - (0.1 * volume_factor)
                speed_reduction_factor *= speed_adjustment
            else:
                # Above capacity: congestion
                excess_volume = vehicle_volume - 100
                if excess_volume <= 50:
                    reduction = 0.07 * (excess_volume / 10)
                else:
                    base_reduction = 0.35
                    additional_excess = excess_volume - 50
                    additional_reduction = 0.35 * (additional_excess / 50)
                    reduction = base_reduction + additional_reduction
                
                speed_reduction_factor *= (1 - reduction)
                speed_reduction_factor = max(speed_reduction_factor, 0.15)
        
        # 3. ACCIDENT: severity-based reduction
        if has_accident is not None and has_accident:
            if accident_severity == 'severe':
                speed_reduction_factor *= 0.40
            elif accident_severity == 'moderate':
                speed_reduction_factor *= 0.65
            elif accident_severity == 'minor':
                speed_reduction_factor *= 0.80
            else:
                speed_reduction_factor *= 0.65
        
        # 4. DAY TYPE
        if day_type is not None:
            if day_type == 'weekend':
                speed_reduction_factor *= 0.92
            elif day_type == 'holiday':
                speed_reduction_factor *= 0.85
        
        # ===== MODIFY currentSpeed =====
        df['currentSpeed'] = df['currentSpeed'] * speed_reduction_factor
        df['currentSpeed'] = df['currentSpeed'].clip(lower=5).round(1)
        
        # ===== MODIFY currentTravelTime =====
        # Time = Distance / Speed, so if speed reduces, time increases proportionally
        df['currentTravelTime'] = (df['currentTravelTime'] / speed_reduction_factor).round().astype(int)
        
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
        return traffic_level
  
    
    except Exception as e:
        print(f"Error in analyze_whatif: {str(e)}")
        return None