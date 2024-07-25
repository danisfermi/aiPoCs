import csv
import random
from datetime import datetime, timedelta

def generate_random_data(start_time, num_points, interval_minutes=5, max_packet_rate=100):
    timestamps = []
    packet_rates = []
    
    current_time = start_time
    for _ in range(num_points):
        timestamps.append(current_time)
        packet_rates.append(random.randint(0, max_packet_rate))
        current_time += timedelta(minutes=interval_minutes)
    
    return timestamps, packet_rates

def write_to_csv(timestamps, packet_rates, filename='additional_packet_rate_data.csv'):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['5 Minutes', 'Rate'])
        for time, rate in zip(timestamps, packet_rates):
            formatted_time = time.strftime('%m/%d/%Y %H:%M')
            writer.writerow([formatted_time, rate])

def main():
    start_time = datetime.strptime('04/01/2016 0:00', '%m/%d/%Y %H:%M')  # Start from a specific date and time
    num_points = 1000  # Number of data points
    interval_minutes = 5  # Interval in minutes
    max_packet_rate = 100  # Maximum packet rate value
    
    timestamps, packet_rates = generate_random_data(start_time, num_points, interval_minutes, max_packet_rate)
    write_to_csv(timestamps, packet_rates)

if __name__ == "__main__":
    main()
