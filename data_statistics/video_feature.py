from load_data import getOriginalData,ITEM_DAILY_FEATURES,ITEM_CATEGORIES,SMALL_MATRIX
from plot import plot_bar,plot_density
from collections import defaultdict
import pandas as pd

class VideoFeature:
    def __init__(self):
        self.video_feature_df = getOriginalData(ITEM_DAILY_FEATURES)
        self.video_tag_df = getOriginalData(ITEM_CATEGORIES)
        self.video_interaction_df = getOriginalData(SMALL_MATRIX)

    def __milliseconds_to_minutes(self,milliseconds):
        return round(milliseconds/60000,2)
    
    def __duration_degree(self,duration):
        duration = self.__milliseconds_to_minutes(duration)
        if duration < 1:
            return "short video"
        elif duration < 5:
            return "middle video"
        elif duration < 10:
            return "long video"
        elif duration >= 10:
            return "very long video"
        else:
            return "UNKOWN"
        
    def vedio_duration(self):
        duration = self.video_interaction_df["video_duration"]

        duration_degree = duration.apply(self.__duration_degree)
        duration_degree = sorted(duration_degree.value_counts().items(),key=lambda x:x[1],reverse=True)
        plot_bar(duration_degree,"Video duration(minutes)","Count","Video duration distribution")

        duration_density = duration[duration < 150000]
        plot_density(duration_density,"Video duration(milliseconds)","Density","Video duration density")
        

    def video_upload_type(self):
        upload_type = self.video_feature_df["upload_type"].value_counts()
        upload_type = sorted(upload_type.items(),key=lambda x:x[1],reverse=True)
        plot_bar(upload_type,"Video upload type","Count","Video upload type distribution",rotation=45,bottom_margin=0.2)

    def __convert_to_list(self,data):
        try:
            return eval(data)
        except:
            return []
        
    def get_video_tag(self):
        video_tag = self.video_tag_df["feat"].apply(self.__convert_to_list)
        video_tag_dict = defaultdict(int)
        for tag_list in video_tag:
            for tag in tag_list:
                video_tag_dict[tag] += 1
        plot_bar(video_tag_dict.items(),"Video tag","Count","Video tag distribution",rotation=45,bottom_margin=0.2)

    def get_video_tag_count(self):
        video_tag = self.video_tag_df["feat"].apply(lambda x:str(len(self.__convert_to_list(x))))
        video_tag = sorted(video_tag.value_counts().items(),key=lambda x:x[1],reverse=True)
        plot_bar(video_tag,"Video tag count","Count","Video tag count distribution")

if __name__ == "__main__":
    video_feature = VideoFeature()
    video_feature.vedio_duration()
    video_feature.video_upload_type()
    video_feature.get_video_tag()
    video_feature.get_video_tag_count()

