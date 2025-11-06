import os
import sys
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

# ----------------- Navigation routes -----------------
@app.route('/')
def home():
    try:
        return render_template('home.html')
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(sys, e)

@app.route('/predictions')
def predictions():
    try:
        return render_template('predictions.html')
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(sys, e)

@app.route('/analytics')
def analytics():
    try:
        return render_template('analytics.html')
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(sys, e)

@app.route('/about')
def about():
    try:
        return render_template('about.html')
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(sys, e)

@app.route('/feedback')
def feedback():
    try:
        return render_template('feedback.html')
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(sys, e)

# ----------------- API routes -----------------
@app.route('/analyze_whatif', methods=['POST'])
def analyze_what_if():
    try:
        data = request.get_json()
        import Features.whatif as whatif

        traffic_level = whatif.analyze_whatif(data)
        if traffic_level is None:
            return jsonify({"error": "Failed to analyze traffic"}), 500

        return jsonify({"traffic_level": traffic_level})
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(sys, e)

@app.route('/analytics_data', methods=['POST'])
def traffic_analytics():
    try:
        data = request.get_json()
        roadname = data.get('road')

        # Lazy imports
        import pandas as pd
        import Features.traffic_insights as traffic_insights

        peak_hours_df = traffic_insights.peak_hours(roadname)
        if isinstance(peak_hours_df, pd.DataFrame):
            peak_hours_data = peak_hours_df.to_dict(orient='records')
        else:
            peak_hours_data = str(peak_hours_df)

        print(peak_hours_data)
        carbon_emission = traffic_insights.co2_emission_rate(roadname)
        travel_variability = traffic_insights.travel_time_variability(roadname)
        congestion_rate = traffic_insights.congestionrate(roadname)
        delay_ratio_data = traffic_insights.delay_ratio(roadname)

        return jsonify({
            "status": "success",
            "peak_hours_data": peak_hours_data,
            "carbon_emission": carbon_emission,
            "travel_variability": travel_variability,
            "congestion_rate": congestion_rate,
            "delay_ratio_data": delay_ratio_data
        })
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(sys, e)

@app.route('/submit_feedback', methods=['POST'])
def get_feedbackformdata():
    try:
        data = request.get_json()
        import Features.feedbackform as feedbackform

        feedbackform.get_response(data)
        return jsonify({'status': 'success', 'message': 'Feedback received'})
    except Exception as e:
        from src.exception import CustomException
        raise CustomException(sys, e)

@app.route('/selected_road', methods=['POST'])
def selected_road():
    try:
        data = request.get_json()
        selected = data.get('road')
        import Features.prediction as prediction

        traffic_level, clear_time_estimate = prediction.predict_real_time_traffic(selected)

        return jsonify({
            "status": "success",
            "traffic_level": traffic_level,
            "clear_time_estimate": clear_time_estimate
        })
    except Exception as e:
        print("Error in /selected_road:", e)
        # Instead of raising CustomException:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500


# ----------------- Run app -----------------
if __name__ == '__main__':
    print("Starting Flask app...")
    # app.run(debug=True)
