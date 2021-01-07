import cv2
from datetime import datetime, timedelta
from dateutil import parser
from dateutil.parser._parser import ParserError
import os
from glob import glob
from sklearn.model_selection import train_test_split
import numpy as np
import ffmpeg


def check_rotation(path_video_file):
    # this returns meta-data of the video file in form of a dictionary
    meta_dict = ffmpeg.probe(path_video_file)

    # from the dictionary, meta_dict['streams'][0]['tags']['rotate'] is the key
    # we are looking for
    rotateCode = None
    if "tags" in meta_dict['streams'][0] and "rotate" in meta_dict['streams'][0]['tags']:
        if int(meta_dict['streams'][0]['tags']['rotate']) == 90:
            rotateCode = cv2.ROTATE_90_CLOCKWISE
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 180:
            rotateCode = cv2.ROTATE_180
        elif int(meta_dict['streams'][0]['tags']['rotate']) == 270:
            rotateCode = cv2.ROTATE_90_COUNTERCLOCKWISE

    return rotateCode


def correct_rotation(frame, rotateCode):
    return cv2.rotate(frame, rotateCode)


def extract_frame_from_movie(source_file, target_folder, clip_sec=2):
    """
    Given a video, directly generated images and count of image,
    save 1. images and 2. annotation for inference purpose
    one video per time
    split by 2s
    folder structure :/data/ucf101/rawframes/throw/name/jpgs
    :param source_file:
    :param target_folder:
    :return:
    """
    # target_folder = "original/throw_up_down_0"
    # source_file = ["source_data/video/throw/throw_up_down_0.mp4"]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    generated_video = []
    if not isinstance(source_file, list):
        source_file = [source_file]
    if not os.path.exists(os.path.join(target_folder, "inference_video", "inference")):
        os.makedirs(os.path.join(target_folder, "inference_video", "inference"), exist_ok=False)

    # print("sn", source_file)
    for video_name in source_file:
        v_name, _ = video_name.split(".")
        v_affix = "avi"
        v_name = v_name.split("/")[-1]
        cameraCapture = cv2.VideoCapture(video_name)
        assert cameraCapture.isOpened(), "failed to load video!"
        rotateCode = check_rotation(video_name)
        size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if rotateCode in [0, 2]:
            size = size[1], size[0]
        video_count = 0
        fps = cameraCapture.get(cv2.CAP_PROP_FPS)
        print("FPS for video {} is: {}".format(video_name, fps))
        success, frame = cameraCapture.read()
        if rotateCode is not None:
            print("Need rotation!")
            frame = correct_rotation(frame, rotateCode)
        idx = 1
        outVideo = cv2.VideoWriter(
            os.path.join(target_folder, "inference_video", "inference", v_name + "_" + str(video_count) + "." + v_affix),
            fourcc, fps, size)
        start_time = parser.parse("{}:{}:{}".format(0, 0, 0))
        end_time = parser.parse("{}:{}:{}".format(0, 0, clip_sec))
        while success:
            milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)
            if idx == 1 and milliseconds < 0:
                # Then enforce millisecond as 0, only show on mts file
                milliseconds = 0
            minutes = 0
            hours = 0
            seconds = milliseconds // 1000

            if seconds >= 60:
                minutes = seconds // 60
                seconds = seconds % 60

            if minutes >= 60:
                hours = minutes // 60
                minutes = minutes % 60

            try:
                current_time = parser.parse("{}:{}:{}".format(hours, minutes, seconds))
                print(current_time)
            except ParserError:
                print("Failed to parse date! quit program! check {}:{}:{}".format(hours, minutes, seconds))
                assert False, "Quit program!"

            if start_time <= current_time <= end_time:
                idx += 1
                outVideo.write(frame)

            elif end_time < current_time:
                # Release old and create new video
                outVideo.release()
                generated_video.append(
                    [os.path.join("inference", v_name + "_" + str(video_count)), idx - 1, start_time, end_time])
                start_time = end_time
                end_time = end_time + timedelta(seconds=clip_sec)
                video_count += 1
                outVideo = cv2.VideoWriter(
                    os.path.join(target_folder, "inference_video", "inference",
                                 v_name + "_" + str(video_count) + "." + v_affix),
                    fourcc, fps, size)
                idx = 1
                continue

            success, frame = cameraCapture.read()
            if rotateCode is not None:
                frame = correct_rotation(frame, rotateCode)

        cameraCapture.release()
    with open("ucf101/inference_list.txt", 'w') as file_writer:
        for ele in generated_video:
            if ele[1]>0:
                file_writer.writelines(" ".join(map(str, ele[:2])) + " -1\n")
    with open("ucf101/inference_result.txt", 'w') as file_writer:
        for ele in generated_video:
            if ele[1]>0:
                file_writer.writelines(" ".join(map(str, ele)) + " -1\n")

    return generated_video

def prepare_movie_with_timestamp(source_folder, target_folder, action_name, clip_len=64):
    """
    Cut to multiple small movie, each contains exactly  clips, drop if <16
    save to multiple videos
    :param source_folder:
    :param target_folder:
    :return:
    """
    source_file = glob(os.path.join(source_folder, "video", action_name, "*.*"))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    assert len(source_file) > 0, "Movie files are not available."
    if not os.path.exists(os.path.join(target_folder, "videos", action_name)):
        os.makedirs(os.path.join(target_folder, "videos", action_name), exist_ok=False)
    generated_video = []

    for video_name in source_file:
        print("procesing file {}".format(video_name))
        v_name, _ = video_name.split(".")
        v_affix = "avi"
        v_name = v_name.split("/")[-1]
        annotation_name = video_name.replace("video", "gt_timestamp").replace(video_name.split(".")[-1], "txt")
        cameraCapture = cv2.VideoCapture(video_name)
        rotateCode = check_rotation(video_name)

        assert cameraCapture.isOpened(), "failed to load video!"
        size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print(size)
        if rotateCode in [0, 2]:
            size = size[1], size[0]
        # print(size,rotateCode)
        video_count = 0
        fps = cameraCapture.get(cv2.CAP_PROP_FPS)
        print("FPS for video {} is: {}".format(video_name, fps))

        time_string_array = []
        with open(annotation_name, 'r') as file_loader:
            for time_string in file_loader:
                time_string = time_string.strip("\n")
                if len(time_string) > 0:
                    start_time, end_time = parser.parse(time_string.split("-")[0]), parser.parse(
                        time_string.split("-")[-1])
                    # assert start_time < end_time, "start date larger than end date, that's wrong!"
                    time_string_array.append([start_time, end_time])

        success, frame = cameraCapture.read()
        if rotateCode is not None:
            print("Need rotation!")
            frame = correct_rotation(frame, rotateCode)
        idx = 1

        if len(time_string_array) > 0:
            start_time, end_time = time_string_array.pop(0)
        else:
            print("This file {} has no target to extract!".format(video_name))
            continue
        # print("The size of the video is: ", size,frame.shape[:-1], video_name)
        outVideo = cv2.VideoWriter(
            os.path.join(target_folder, "videos", action_name, v_name + "_" + str(video_count) + "." + v_affix),
            fourcc, fps, size)
        while success:
            milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)
            if idx == 1 and milliseconds < 0:
                # Then enforce millisecond as 0, only show on mts file
                milliseconds = 0
            minutes = 0
            hours = 0
            seconds = milliseconds // 1000

            if seconds >= 60:
                minutes = seconds // 60
                seconds = seconds % 60

            if minutes >= 60:
                hours = minutes // 60
                minutes = minutes % 60

            try:
                current_time = parser.parse("{}:{}:{}".format(hours, minutes, seconds))
            except ParserError:
                print("Failed to parse date! quit program! ")
                print(hours, minutes, seconds)

            if start_time <= current_time <= end_time:
                if idx % (clip_len + 1) == 0:
                    # Release old and create new video
                    idx = 0
                    outVideo.release()
                    generated_video.append(
                        os.path.join(action_name, v_name + "_" + str(video_count) + "." + v_affix))
                    video_count += 1
                    outVideo = cv2.VideoWriter(
                        os.path.join(target_folder, "videos", action_name,
                                     v_name + "_" + str(video_count) + "." + v_affix),
                        fourcc, fps, size)
                idx += 1
                outVideo.write(frame)

            elif end_time < current_time:
                prev_end_time = end_time
                if len(time_string_array) > 0:
                    start_time, end_time = time_string_array.pop(0)
                    # assert prev_end_time < end_time, "Issues with labeling file! Is it correct?"
                else:
                    # All target of interest have been collected
                    if idx >= 32:
                        outVideo.release()
                        generated_video.append(
                            os.path.join(action_name,
                                         v_name + "_" + str(video_count) + "." + v_affix))
                    cameraCapture.release()
                    break

            success, frame = cameraCapture.read()
            if rotateCode is not None:
                frame = correct_rotation(frame, rotateCode)

        if idx >= 32:
            outVideo.release()
            generated_video.append(
                os.path.join(action_name, v_name + "_" + str(video_count) + "." + v_affix))
        cameraCapture.release()
    return generated_video


def prepare_movie_with_timestamp_bkrd(source_folder, target_folder, clip_len=64):
    """
    Cut to multiple small movie, each contains exactly  clips, drop if <16
    save to multiple videos
    :param source_folder:
    :param target_folder:
    :return:
    """
    action_name = "background"
    source_file = glob(os.path.join(source_folder, "video", "udstairs", "*.*"))
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    assert len(source_file) > 0, "Movie files are not available."
    if not os.path.exists(os.path.join(target_folder, "videos", action_name)):
        os.makedirs(os.path.join(target_folder, "videos", action_name), exist_ok=False)
    generated_video = []

    for video_name in source_file:
        print("procesing file {}".format(video_name))
        v_name, _ = video_name.split(".")
        v_affix = "avi"
        v_name = v_name.split("/")[-1]
        annotation_name = video_name.replace("video", "gt_timestamp").replace(video_name.split(".")[-1], "txt")
        cameraCapture = cv2.VideoCapture(video_name)
        rotateCode = check_rotation(video_name)

        assert cameraCapture.isOpened(), "failed to load video!"
        size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        # print(size)
        if rotateCode in [0, 2]:
            size = size[1], size[0]
        # print(size,rotateCode)
        video_count = 0
        fps = cameraCapture.get(cv2.CAP_PROP_FPS)
        print("FPS for video {} is: {}".format(video_name, fps))

        time_string_array = []
        with open(annotation_name, 'r') as file_loader:
            for time_string in file_loader:
                time_string = time_string.strip("\n")
                if len(time_string) > 0:
                    start_time, end_time = parser.parse(time_string.split("-")[0]), parser.parse(
                        time_string.split("-")[-1])
                    # assert start_time < end_time, "start date larger than end date, that's wrong!"
                    time_string_array.append([start_time, end_time])

        success, frame = cameraCapture.read()
        if rotateCode is not None:
            print("Need rotation!")
            frame = correct_rotation(frame, rotateCode)
        idx = 1

        if len(time_string_array) > 0:
            start_time, end_time = time_string_array.pop(0)
        else:
            print("This file {} has no target to extract!".format(video_name))
            continue
        # print("The size of the video is: ", size,frame.shape[:-1], video_name)
        print("path: ",os.path.join(target_folder, "videos", action_name, v_name + "_" + str(video_count) + "." + v_affix))
        outVideo = cv2.VideoWriter(
            os.path.join(target_folder, "videos", action_name, v_name + "_" + str(video_count) + "." + v_affix),
            fourcc, fps, size)
        while success:
            milliseconds = cameraCapture.get(cv2.CAP_PROP_POS_MSEC)
            if idx == 1 and milliseconds < 0:
                # Then enforce millisecond as 0, only show on mts file
                milliseconds = 0
            minutes = 0
            hours = 0
            seconds = milliseconds // 1000

            if seconds >= 60:
                minutes = seconds // 60
                seconds = seconds % 60

            if minutes >= 60:
                hours = minutes // 60
                minutes = minutes % 60

            try:
                current_time = parser.parse("{}:{}:{}".format(hours, minutes, seconds))
            except ParserError:
                print("Failed to parse date! quit program! ")
                print(hours, minutes, seconds)

            if not (start_time <= current_time <= end_time):
                if end_time < current_time:
                    prev_end_time = end_time
                    if len(time_string_array) > 0:
                        start_time, end_time = time_string_array.pop(0)
                        # assert prev_end_time < end_time, "Issues with labeling file! Is it correct?"
                    else:
                        # All target of interest have been collected
                        if idx >= 32:
                            outVideo.release()
                            generated_video.append(
                                os.path.join(action_name,
                                             v_name + "_" + str(video_count) + "." + v_affix))
                        cameraCapture.release()
                        break
                if idx % (clip_len + 1) == 0:
                    # Release old and create new video
                    idx = 0
                    outVideo.release()
                    generated_video.append(
                        os.path.join(action_name, v_name + "_" + str(video_count) + "." + v_affix))
                    video_count += 1
                    outVideo = cv2.VideoWriter(
                        os.path.join(target_folder, "videos", action_name,
                                     v_name + "_" + str(video_count) + "." + v_affix),
                        fourcc, fps, size)
                idx += 1
                outVideo.write(frame)

            success, frame = cameraCapture.read()
            if rotateCode is not None:
                frame = correct_rotation(frame, rotateCode)

        if idx >= 32:
            outVideo.release()
            generated_video.append(
                os.path.join(action_name, v_name + "_" + str(video_count) + "." + v_affix))
        cameraCapture.release()
    return generated_video


def generate_classidx(target_folder, class_idx):
    if not os.path.exists(os.path.join(target_folder, "annotations")):
        os.makedirs(os.path.join(target_folder, "annotations"), exist_ok=False)
    with open(os.path.join(target_folder, "annotations", "classInd.txt"), 'w') as file_writer:
        for action_name_key in class_idx:
            file_writer.writelines("{} {}\n".format(class_idx[action_name_key], action_name_key))


def generate_annotation(target_folder, class_idx, video_list, seed=42, test_ratio=0.2):
    """
    Generate annotation after you obtain video gt
    build one list for train and test, prefix 01
    :param class_idx: type dict, indicate the class to train
    :param video_list: the list of video to generate as gt
    :return:
    """
    if not os.path.exists(os.path.join(target_folder, "annotations")):
        os.makedirs(os.path.join(target_folder, "annotations"), exist_ok=False)
    X_trainval, X_test = train_test_split(video_list, test_size=test_ratio, random_state=seed)
    for i in range(1, 4):
        with open(os.path.join(target_folder, "annotations", "trainlist{:02d}.txt".format(i)), 'w') as file_writer:
            for path in X_trainval:
                action_name = path.split("/")[0]
                file_writer.writelines(path + " " + str(class_idx[action_name]) + "\n")
        with open(os.path.join(target_folder, "annotations", "testlist{:02d}.txt".format(i)), 'w') as file_writer:
            for path in X_test:
                action_name = path.split("/")[0]
                file_writer.writelines(path + " " + str(class_idx[action_name]) + "\n")


if __name__ == "__main__":
    """
    Use this script to organize data in ucf101 format:
    This script can achieve the following objectives:
    1. Generate video
    2. Generate annotation
        i. Generate class index
        ii. Generate train/test annotation file
    Copy source data folder into mmaction/data before running script
    Make sub-folder by action name for each raw video
    """
    root = ""
    source_folder, target_folder = os.path.join(root, "source_data"), os.path.join(root, "ucf101")
    clip_len = 64
    action_name = ["udstairs"]
    seed = 42
    test_ratio = 0.2
    class_idx = {x: i + 2 for i, x in enumerate(action_name)}
    class_idx["background"] = 1
    mode = "train"  # train or test
    if mode == "test":
        bkrd_list = prepare_movie_with_timestamp_bkrd(source_folder, target_folder, clip_len=clip_len)
        generate_classidx(target_folder, class_idx)
        for action in action_name:
            video_list = prepare_movie_with_timestamp(source_folder, target_folder, action, clip_len=clip_len)
            generate_annotation(target_folder, class_idx, bkrd_list + video_list, seed=seed,
                                test_ratio=test_ratio)
    else:
        # Root : data
        source_file = "source_data/video/udstairs/1027_ok_1p_part4.mp4"
        target_folder = os.path.join("ucf101")
        extract_frame_from_movie(source_file, target_folder, clip_sec=0.8)
