from load_data import getOriginalData,USER_FEATURES
from plot import plot_bar

# 获取用户特征
class UserFeature:
    def __init__(self):
        self.user_feature_df = getOriginalData(USER_FEATURES)

    def get_activate(self):
        activate = sorted(self.user_feature_df.groupby("user_active_degree")["user_id"].count().items(),key=lambda x:x[1],reverse=True)
        plot_bar(activate,"User activate degree","Count","User activity distribution")

    def get_register_time(self):
        register_time = sorted(self.user_feature_df.groupby("register_days_range")["user_id"].count().items(),key=lambda x:x[1],reverse=True)
        plot_bar(register_time,"User register time","Count","User register time distribution")

if __name__ == "__main__":
    user_feature = UserFeature()
    user_feature.get_activate()
    user_feature.get_register_time()

