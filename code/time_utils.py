import re
from datetime import datetime


def parse_year_period(period_text):
    """
    Parse source period text into start/end years.

    Supported examples:
    - "1957-2017"
    - "1957–2017"
    - "(1957-2017)"
    - "1957"
    """
    if period_text is None:
        return None, None

    text = str(period_text).strip()
    if not text:
        return None, None

    m = re.search(r"(\d{4})\s*[-–]\s*(\d{4})", text)
    if m:
        start_year = int(m.group(1))
        end_year = int(m.group(2))
        if start_year <= end_year:
            return start_year, end_year
        return end_year, start_year

    m = re.fullmatch(r"\s*(\d{4})\s*", text)
    if m:
        year = int(m.group(1))
        return year, year

    return None, None


def climatology_mid_datetime(start_year, end_year):
    """
    Return representative climatology timestamp.

    Convention:
    - middle year of source period
    - July 1 of that middle year
    - integer midpoint uses floor division, matching ALi_De_Boer and HMA
    """
    if start_year is None or end_year is None:
        return None

    start_year = int(start_year)
    end_year = int(end_year)
    if start_year > end_year:
        start_year, end_year = end_year, start_year

    mid_year = (start_year + end_year) // 2
    return datetime(mid_year, 7, 1)


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
