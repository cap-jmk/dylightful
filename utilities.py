import json


def save_dict(d, path):
    with open(path + "_time_series.json", "w") as fp:
        json.dump(d, fp)
    print("Saved file to", path)
