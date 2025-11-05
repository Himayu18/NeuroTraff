import pandas as pd 
import datetime

trainset = pd.read_csv(r"artifacts\TransformedData\train_data_transformed.csv")
testset = pd.read_csv(r"artifacts/TransformedData/test_data_transformed.csv")
rawdf = pd.read_csv(r"artifacts/RawData/raw_data.csv")
new_df = pd.concat([trainset,testset])
new_df['road'] = rawdf['road']
year = 2025
month = 8
new_df['Date_time'] = pd.to_datetime({'year':year,'month':month,'day':new_df['Day'],'hour':new_df['Hour'],'minute':new_df['minute']})
new_df['Day_name'] = new_df['Date_time'].dt.day_name()




def peak_hours(roadname):
    today_day = datetime.datetime.now().strftime('%A')
    peak = (new_df.groupby(['road', 'Day_name', 'Hour'])['currentSpeed'].mean().reset_index(name='mean_currentSpeed'))
    filtered = peak.loc[
        (peak['road'] == roadname) & (peak['Day_name'] == today_day),
        ['Hour', 'mean_currentSpeed']]
    if filtered.empty:
        return f"No data available for {roadname} on {today_day}."
    else:
        return filtered.sort_values(by='mean_currentSpeed', ascending=False)

def congestionrate(roadname):
    congestion_rate_df = (
        new_df.groupby(['road'])['delay ratio']
        .mean()
        .mul(100)
        .reset_index(name='congestionrate')
    )
    result = congestion_rate_df.loc[
        congestion_rate_df['road'] == roadname, 'congestionrate'
    ]
    if not result.empty:
        return round(result.iloc[0], 2)  # round to 2 decimal places
    else:
        return None


def co2_emission_rate(roadname):
    df = new_df.copy()
    df['travel_time_h'] = df['currentTravelTime'] / 3600
    df['distance_km'] = df['currentSpeed'] * df['travel_time_h']

    def emission_factor(speed):
        if speed <= 20:
            return 300
        elif speed <= 50:
            return 200
        elif speed <= 80:
            return 150
        elif speed <= 120:
            return 180
        else:
            return 250

    df['emission_factor'] = df['currentSpeed'].apply(emission_factor)
    df['CO2_emission_g'] = df['emission_factor'] * df['distance_km']
    df['CO2_emission_kg'] = df['CO2_emission_g'] / 1000


    emissions_by_road = df.groupby('road', as_index=False)['CO2_emission_kg'].sum()
    result = emissions_by_road.loc[emissions_by_road['road'] == roadname, 'CO2_emission_kg']
    if not result.empty:
        return round(result.iloc[0], 2)  # rounded
    else:
        return None


def travel_time_variability(roadname):
    grouped = new_df.groupby('road')['currentTravelTime']
    travel_time_variability = (grouped.std() / grouped.mean()).reset_index(name='travel_time_variability')
    result = travel_time_variability.loc[
        travel_time_variability['road'] == roadname, 'travel_time_variability'
    ]
    if not result.empty:
        return round(result.iloc[0], 3)
    else:
        return None


def delay_ratio(roadname):
    delay_ratio_df = (new_df.groupby('road')['delay ratio'].mean().reset_index(name='delayrato'))
    result = delay_ratio_df.loc[
        delay_ratio_df['road'] == roadname, 'delayrato'
    ]
    if not result.empty:
        return result.iloc[0]
    else:
        return None


    