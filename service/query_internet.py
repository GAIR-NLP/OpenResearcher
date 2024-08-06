from config import bing_search_end_point, bing_search_key
import requests

def bing_search(query):
    subscription_key = bing_search_key
    endpoint = bing_search_end_point
    mkt = 'en-US'
    params = { 'q': query, 'mkt': mkt }
    headers = { 'Ocp-Apim-Subscription-Key': subscription_key }
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()
        pages = response.json()['webPages']['value']
        result_list = []
        for page in pages:
            result_list.append(f"""Page name: {page['name']}
Page url: {page['url']}
Page snippet: {page['snippet']}""")    
        return "\n\n".join(result_list)
    except Exception as e:
        print(e)
        return "No search result yet"

def query_internet(query_str):
    return bing_search(query_str)