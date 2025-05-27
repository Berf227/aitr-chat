''' 
1) Her soru cevap döngüsünde, önceki etkileşimleri saklayan bir global “history” sözlüğü oluştur
2) bunu prompt içerisine ekle

'''
##########BU KISIMDA BİR DEĞİŞİKLİK YAPMANA GEREK YOK

from typing import List, Dict


history: List[Dict[str, str]] = []

def add_to_history(role: str, content: str) -> None:
    if role not in ("user", "assistant"):
        raise ValueError(f"Invalid role: {role}")
    history.append({"role": role, "content": content})
    max_msgs = 20 
    if len(history) > max_msgs:
        del history[0 : len(history) - max_msgs]

def get_history() -> List[Dict[str, str]]:
    return history
