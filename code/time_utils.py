# core/time_utils.py
def parse_period(period_str):
    if period_str is None:
        return None, None

    period_str = str(period_str).replace('–', '-').replace('—', '-')
    parts = period_str.split('-')
    if len(parts) == 2:
        try:
            return int(parts[0]), int(parts[1])
        except ValueError:
            return None, None
    return None, None


def climatology_time(start_year, end_year):
    from datetime import datetime
    if not start_year or not end_year:
        return None
    mid_year = (start_year + end_year) // 2
    return (datetime(mid_year, 7, 1) - datetime(1970, 1, 1)).days
