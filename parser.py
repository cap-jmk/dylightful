import os
import xml.etree.ElementTree as ET
import numpy as np
import json
from pathlib import Path

from utilities import save_dict


def get_time_series(pml_path):
    """ets the time_series of the dynophore from the pml file

    Args:
        pml_path (str): path to the pml file containing the Dynophore trajectory

    Returns:
        [dictionary, JSON]: returns the time series for each superfeature as a JSON file
    """

    tree = ET.parse(pml_path)
    root = tree.getroot()
    time_series = {}
    for child in root:
        i = 0
        frames = []
        for attributes in child:
            if i > 0:  # first entry does not provide frameIndex information
                frame_idx = int(attributes.get("frameIndex"))
                frames.append(frame_idx)
                if i == 1:  # get the value of the last frameIndex
                    max_index = frame_idx + 1  # counting in python starts at 0
                elif max_index < frame_idx +1:
                    max_index = frame_idx + 1
            i += 1
        time_series[child.get("id")] = frames
    time_series["num_frames"] = max_index
    print("Max features is:", max_index)
    time_series = rewrites_time_series(time_series)
    save_path = parse_file_path(pml_path)
    save_dict(time_series, save_path)
    return time_series


def rewrites_time_series(feature_series):
    """Convertes to a sparse time series to be ready for the HMM processing

    Args:
        feature_series (np.array): 

    Returns:
        dictionionary, JSON: JSON with the time series per superfeature
    """

    max_frames = feature_series["num_frames"]
    keys = list(feature_series.keys())
    for i in range(len(keys) - 1):
        time_ser_feat = feature_series[keys[i]]
        new_time_ser = np.zeros(int(max_frames))
        try:
            for frame_index in time_ser_feat:
                try:
                    if frame_index<len(new_time_ser):
                        new_time_ser[int(frame_index)] = 1
                        if max_frames < frame_index: 
                            max_frames = int(frame_index) #if something with the frame_index is wrong set it here
                            print("Set max frames to", frame_index)
                    else:
                        tmp = np.zeros(int(frame_index+50)) #free new memory
                        tmp[:len(new_time_ser)] = new_time_ser
                        tmp[int(frame_index)] = 1
                        new_time_ser = tmp 
                except:
                    print(
                        "Error parsing into new time series in superfeature, ",
                        keys[i],
                        "in frame", 
                        frame_index,
                        "but the memory was only",
                        len(new_time_ser),
                        "time points"
                    )
                    continue
        except: 
            raise RuntimeError("Fatal error while parsing superfeature",keys[i])
        new_time_ser = new_time_ser[:max_frames]
        assert len(new_time_ser) == max_frames, "Lengths of parsed time series does not match the maximum number of frames. Length was"+str(len(new_time_ser))
        feature_series[keys[i]] = new_time_ser.astype(np.int32).tolist()
    return feature_series


def parse_file_path(path):
    """Automatically generates an output path for the time trajectory

    Args:
        path ([type]): Dynophore input path
    """
    return get_dir(path) + "/" + get_name(path)


def get_dir(path):
    """Automatically extracts the path to the dynophore trajectory

    Args:
        path (str): File path to the dynophore trajectory

    Returns:
        str: /some/file/path
    """
    dir_path = os.path.dirname(os.path.realpath(path))
    return dir_path


def get_name(path):
    """ Gets the name of the dynophore trajectory without the .pml extension

    Args:
        path (str): File path to the dynophore trajectory

    Returns:
        str: dynophore_pml
    """
    file = Path(path).stem
    return file


if __name__ == "__main__":
    #get_time_series("../Trajectories/Dominique/1KE7_dynophore.json")
    get_time_series("../dynophores-master/dynophores/tests/data/out/1KE7_dynophore.pml")
