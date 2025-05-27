'''
Bu dosya, belirli bir kategorideki belgeleri yüklemek, işlemek ve yanıt almak için kullanılan bir Python betiğidir.
'''
##########BU KISIMDA BİR DEĞİŞİKLİK YAPMANA GEREK YOK

from chat_history_manager import add_to_history, get_history
from typing import List, Dict
from base_processor import DocumentQAProcessor  

def update_chat_history(user_message: str, assistant_message: str) -> None:
    add_to_history("user", user_message)
    add_to_history("assistant", assistant_message)

def fetch_chat_history() -> List[Dict[str, str]]:
    return get_history()
