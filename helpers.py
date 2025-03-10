from datetime import datetime, timedelta
import time

def convert_hour_to_datetime(hour: int) -> datetime:
    return datetime.now().replace(hour=hour, minute=0, second=0, microsecond=0)

def dt_to_unix(dt):
    return int(dt.timestamp())

def dt_to_unix_seconds(dt_obj):
    return int(time.mktime(dt_obj.timetuple())) 