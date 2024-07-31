import pandas as pd
from scapy.all import rdpcap
from datetime import datetime, timedelta

def read_pcap(file_path):
    packets = rdpcap(file_path)
    timestamps = [float(pkt.time) for pkt in packets]
    return timestamps

def generate_csv(timestamps, output_file):
    # Convert timestamps to datetime
    datetimes = [datetime.fromtimestamp(ts) for ts in timestamps]

    # Round to the nearest second
    rounded_datetimes = [dt.replace(microsecond=0) for dt in datetimes]

    # Count packets in each second
    counts = {}
    for dt in rounded_datetimes:
        counts[dt] = counts.get(dt, 0) + 1

    # Create a DataFrame
    df = pd.DataFrame(list(counts.items()), columns=['Second', 'Rate'])

    # Sort by datetime
    df = df.sort_values('Second')

    # Write to CSV
    df.to_csv(output_file, index=False)

# Example usage
pcap_file = 'sample.pcap'
output_csv = 'output_packet_rate_data.csv'

timestamps = read_pcap(pcap_file)
generate_csv(timestamps, output_csv)

