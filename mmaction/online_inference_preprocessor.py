import cv2
import os
import json
import time
import datetime
import argparse


def save_metadata(data, json_path):
    """
    Save inference result by video
    :param data:
    :return:
    """
    with open(json_path, "w") as write_file:
        json.dump(data, write_file)
    return


def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessor for online inference')
    parser.add_argument('stream_link', help='the url for source stream, required')
    parser.add_argument('type', help='Indicate type of inference to conduct, required')
    parser.add_argument('--save_second', type=int, default=2, help='Indicate type of inference to conduct, required')
    parser.add_argument('--video_prefix', default=".avi")
    parser.add_argument('--output_folder', default=r"/Volumes/External drive/stream data")
    parser.add_argument('--skip_frame', type=int, default=-1, help="set number if you want to sample video")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Collect data from flv
    args = parse_args()
    stream_link = args.stream_link
    stream_affix = "." + stream_link.split(".")[-1]
    output_affix = args.video_prefix
    output_list = ["device_id", "job_id", "start_timestamp", "end_timestamp"]
    data = {x: None for x in output_list}
    data["type"] = args.type
    cap = cv2.VideoCapture(stream_link)
    # Gather status
    assert cap.isOpened(), "Unable to obtain video!"
    stream_name = stream_link.split("/")[-1].replace(stream_affix, "")
    assert len(stream_name.split(".")) == 2, "cannot locate device id and job id, is the url correct?"
    data["device_id"], data["job_id"] = stream_name.split(".")
    # Collect FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Current fps of the video is: {}".format(fps))
    # get height, width of video
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("The size of the video is: {}".format(str((size))))
    save_folder = args.output_folder
    save_video_folder = os.path.join(save_folder, "videos")
    save_meta_folder = os.path.join(save_folder, "inference")
    stream_id = 1
    if not os.path.exists(save_folder):
        os.makedirs(save_folder, exist_ok=False)
        os.makedirs(save_video_folder, exist_ok=False)
        os.makedirs(save_meta_folder, exist_ok=False)

    fourcc = cv2.VideoWriter_fourcc(*"MJPG")  # MP42/VP6f
    success, frame = cap.read()

    tot = 0
    # Set skip_frame here for sampling purpose, disable by setting to -1
    skip_frame = args.skip_frame
    save_seconds = args.save_second
    timedelta = datetime.timedelta(seconds=save_seconds)
    total_frames = save_seconds * fps

    outVideo = cv2.VideoWriter(os.path.join(save_video_folder, stream_name + "_" + str(stream_id) + output_affix),
                               fourcc, fps,
                               size)

    while success:
        if tot < total_frames:
            success, frame = cap.read()
            if tot == 0:
                # set start time here, need to do conversion
                start_time = datetime.datetime.now()
                data["start_timestamp"] = time.mktime(start_time.timetuple())

            # save frame only after given skip_frame count
            if skip_frame != -1 and tot % skip_frame == 0:
                cv2.imwrite('cut/' + 'cut_' + str(tot // skip_frame) + '.jpg', frame)
            tot += 1
            outVideo.write(frame)
            cv2.waitKey(25)
        else:
            outVideo.release()
            tot = 0  # reset frame count and save a new video
            stream_id += 1
            output_video_name = os.path.join(save_video_folder, stream_name + "_" + str(stream_id))
            output_metadata_name = os.path.join(save_meta_folder, stream_name + "_" + str(stream_id))
            outVideo = cv2.VideoWriter(output_video_name + output_affix, fourcc,
                                       fps, size)
            # save json info here. Add inference result later.
            end_time = start_time + timedelta
            data["end_timestamp"] = time.mktime(end_time.timetuple())
            save_metadata(data, output_metadata_name + ".json")
        print(
            "{} / {} frames gathered for video {}!".format(tot, int(total_frames),
                                                           stream_name + "_" + str(stream_id) + output_affix))
    outVideo.release()
    cap.release()
