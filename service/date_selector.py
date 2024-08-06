from datetime import datetime
from llm.chat_llm import chat
import re
from config import elastic_seach_month_list, qdrant_month_list, date_selector_model

prompt = """You need to determine the time range to be retrieved based on the user's query and the current system time (e.g., the user's query may require retrieving data from the past year). 
The unit is in months, for example, January 2023 should be represented as 2301. 
A time range from March 2022 to February 2024 should be represented as 2203-2402. 
If it is now July 2022, then the range for the past half year should be 2201-2207.
If the user's query does not involve a retrieval time, simply and ONLY reply "NONE"; otherwise, only reply with the time range in the format "XXXX-XXXX".

Current Time:
{current_time}

User Query:
{query_str}

Query Time Range:
"""


def validate_and_process_time_range(llm_output, month_list):
    pattern = re.compile(r'^\d{4}-\d{4}$')
    if not pattern.match(llm_output):
        return None
    start, end = llm_output.split('-')
    start_year = int(start[:2])
    start_month = int(start[2:])
    end_year = int(end[:2])
    end_month = int(end[2:])
    result = []
    year = start_year
    month = start_month

    while (year < end_year) or (year == end_year and month <= end_month):
        cur_month = f'{year:02d}{month:02d}'
        if cur_month in month_list:
            result.append(cur_month)
        month += 1
        if month > 12:
            month = 1
            year += 1

    return result

def date_selector(query_str, range_type="qdrant"):
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt.format(current_time=current_time, query_str=query_str)}
    ]
    completion = chat(messages, model = date_selector_model)
    response = ""
    for chunk in completion:
        response += chunk
    if range_type == "qdrant":
        return validate_and_process_time_range(response, qdrant_month_list)
    elif range_type == "elastic search":
        return validate_and_process_time_range(response, elastic_seach_month_list)
    return validate_and_process_time_range(response, qdrant_month_list), validate_and_process_time_range(response, elastic_seach_month_list)