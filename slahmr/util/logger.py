import datetime


class Logger(object):
    """
    Static logging class
    """

    log_file = None

    @staticmethod
    def init(log_path):
        Logger.log_file = log_path

    @staticmethod
    def log(write_str, to_stdout=True):
        if to_stdout:
            print(write_str)
        if not Logger.log_file:
            print("Logger must be initialized before logging!")
            return
        time_str = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
        with open(Logger.log_file, "a") as f:
            f.write(time_str + "  ")
            f.write(str(write_str) + "\n")


def log_cur_stats(stats_dict, iter=None, to_stdout=True):
    loss = stats_dict.pop("total", 0)
    Logger.log(f"LOSS: {loss:.04f}", to_stdout=to_stdout)
    for k, v in stats_dict.items():
        Logger.log(f"{k}: {v:.04f}", to_stdout=to_stdout)
    if to_stdout:
        if iter is not None:
            print("======= iter %d =======" % iter)
        else:
            print("========")
