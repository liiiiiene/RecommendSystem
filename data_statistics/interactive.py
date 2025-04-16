from load_data import getOriginalData,SMALL_MATRIX
from plot import plot_density,plot_bar
import pandas as pd

class Interactive:
    def __init__(self):
        self.small_matrix = getOriginalData(SMALL_MATRIX)
    
    def get_watch_ratio(self):
        watch_ratio = self.small_matrix["watch_ratio"]
        watch_ratio = watch_ratio[watch_ratio < 5]
        plot_density(watch_ratio,"Watch ratio","Density","Watch ratio distribution")

    def get_play_times(self):
        play_times = self.small_matrix["user_id"].value_counts()
        plot_density(play_times,"Play times","Density","Play times distribution")
    
    def get_play_datetime(self):
        play_datetime = self.small_matrix["time"]
        date_time_format = "%Y-%m-%d %H:%M:%S.%f"
        play_datetime = play_datetime[pd.notna(play_datetime)]
        play_datetime = pd.to_datetime(play_datetime, format=date_time_format)
        play_datetime = play_datetime.dt.hour
        play_datetime = [i for i in play_datetime.value_counts().items()]
        plot_bar(play_datetime,"Play hour everyday","Count","Play hour distribution",rotation=45,bottom_margin=0.2)
    

if __name__ == "__main__":
    interactive = Interactive()
    interactive.get_watch_ratio()
    interactive.get_play_times()
    interactive.get_play_datetime()
