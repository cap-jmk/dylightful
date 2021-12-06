import json


def save_dict(d, save_path, name=None):
    if name is None: 
        name = "_time_series.json"
    else: 
        name = "_"+name
        name +="_time_series.json"
    
    with open(save_path + name , "w") as fp:
        json.dump(d, fp)
    print("Saved file to", save_path)
