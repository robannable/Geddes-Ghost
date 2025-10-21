#!/usr/bin/env python3
"""
Quick diagnostic script to check if temperature data exists in CSV logs
"""
import pandas as pd
import os
import glob

logs_dir = "logs"
csv_files = glob.glob(os.path.join(logs_dir, "*_response_log.csv"))

if not csv_files:
    print("❌ No CSV files found in logs/ directory")
    print("   Generate some responses first!")
    exit(1)

print(f"Found {len(csv_files)} CSV file(s)\n")

for csv_file in csv_files:
    print(f"📄 Checking: {os.path.basename(csv_file)}")

    try:
        df = pd.read_csv(csv_file, encoding='utf-8', on_bad_lines='skip')
        print(f"   Total rows: {len(df)}")

        # Check for temperature columns
        has_actual_temp = 'actual_temperature' in df.columns
        has_temp_source = 'temperature_source' in df.columns
        has_detected_mode = 'detected_mode' in df.columns

        print(f"   Columns: actual_temperature={has_actual_temp}, "
              f"temperature_source={has_temp_source}, detected_mode={has_detected_mode}")

        if has_actual_temp:
            non_null = df['actual_temperature'].notna().sum()
            print(f"   Rows with temperature data: {non_null} / {len(df)}")

            if non_null > 0:
                print(f"   ✅ Temperature data found!")
                print(f"      Sample temperatures: {df['actual_temperature'].dropna().head(3).tolist()}")
                if has_temp_source:
                    sources = df['temperature_source'].dropna().value_counts().to_dict()
                    print(f"      Sources: {sources}")
            else:
                print(f"   ⚠️  Column exists but all values are None/NaN")
                print(f"      This CSV was created before temperature tracking was added")
        else:
            print(f"   ⚠️  Temperature columns missing (old CSV format)")

        print()

    except Exception as e:
        print(f"   ❌ Error reading file: {str(e)}\n")

print("\n💡 To generate temperature data:")
print("   1. Make sure you have the latest code (git pull)")
print("   2. Use the chatbot to generate new responses")
print("   3. Temperature data will be recorded for new responses")
print("\n💡 To start fresh:")
print(f"   rm {logs_dir}/*.csv  # Delete old CSV files")
print("   # Then generate new responses")
