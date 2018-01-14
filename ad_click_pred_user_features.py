import numpy as np
import pandas as pd
from datetime import date
from datetime import time
from datetime import datetime
import matplotlib.pyplot as plt
import sys
import os

if __name__ == "__main__":
    # Read the data
    input_file_location = sys.argv[1]
    df = pd.read_csv(input_file_location)
    df["ts"] = df.ts.apply(lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S'))
    #1# highly_active Feature
    all_events_per_user = df.groupby(["uuid"]).size().sort_values(ascending=False).reset_index(name='count')
    val_90_percentile = all_events_per_user.quantile(0.90)["count"]
    all_events_per_user["highly_active"] = all_events_per_user["count"].apply(lambda x: x >= val_90_percentile)
    highly_active_df = all_events_per_user.loc[:, ["uuid", "highly_active"]]
    #2# weekday_biz Feature
    weekdays = [0, 1, 2, 3, 4]
    bizhours = [9, 18]  # 9 am to 6 pm
    df["weekday_biz_hours"] = df.ts.apply(
        lambda x: 1 if (bizhours[0] <= x.hour <= bizhours[1]) and (x.weekday() in weekdays) else 0)
    weekday_biz_df = df.loc[:, ["uuid", "weekday_biz_hours"]].groupby("uuid").sum().reset_index()
    weekday_biz_df["weekday_biz"] = weekday_biz_df.weekday_biz_hours.apply(lambda x: x > 0)
    weekday_biz_final_df = weekday_biz_df.loc[:, ["uuid", "weekday_biz"]]
    #3# Recency feature
    end_date = "2017-08-01 00:00:00"  # The date from which the look-back is to be done. Taking 1st August here to look back for July Events
    end_ts = datetime.strptime(end_date, '%Y-%m-%d %H:%M:%S')
    df_gb = df.loc[:, ["uuid", "ts"]].groupby("uuid")
    df_last_event = df_gb["ts"].agg({'last_event': np.max}).reset_index()
    df_last_event["days_since_last_event"] = df_last_event.last_event.apply(lambda x: (end_ts - x).days)
    df_last_event_final = df_last_event.loc[:, ["uuid", "days_since_last_event"]]
    ## Combine all features
    df_final = highly_active_df.merge(weekday_biz_final_df, on="uuid").merge(df_last_event_final, on="uuid")
    ## Write the output
    df_final.to_csv(sys.stdout, sep=',')
    print("output written")