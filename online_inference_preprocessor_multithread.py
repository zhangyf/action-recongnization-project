import cv2
import os
import json
import time
import datetime
import argparse
import multiprocessing


def save_metadata(data, json_path):
    """
    Save inference result by video
    :param data:
    :return:
    """
    with open(json_path, "w") as write_file:
        json.dump(data, write_file)
    return

# change save_sec to 0.5
# change output_list
# change data/video_name and data/prob

def parse_args():
    parser = argparse.ArgumentParser(
        description='Preprocessor for online inference')
    parser.add_argument('--type', default = "inference", help='Indicate type of inference to conduct, required')
    parser.add_argument('--save_second', type=float, default=0.5, help='Indicate type of inference to conduct, required')
    parser.add_argument('--video_prefix', default=".avi")
    parser.add_argument('--output_folder', default=r"/Volumes/External drive/stream data")
    parser.add_argument('--skip_frame', type=int, default=-1, help="set number if you want to sample video")
    parser.add_argument('--inference_count', type=int, default=5, help="inference each 5 videos")
    args = parser.parse_args()
    return args


def main(url, n):
    # Collect data from flv
    args = parse_args()
    stream_link = url
    stream_affix = "." + stream_link.split(".")[-1]
    output_affix = args.video_prefix
    output_list = ["video_name", "prob", "device_id", "job_id", "start_timestamp", "end_timestamp"]
    data = {x: None for x in output_list}
    data["type"] = args.type
    cap = cv2.VideoCapture(stream_link)
    video_list = []
    # Gather status
    assert cap.isOpened(), "Unable to obtain video!"
    stream_name = stream_link.split("/")[-1].replace(stream_affix, "")
    assert len(stream_name.split(".")) == 2, "cannot locate device id and job id, is the url correct?"
    data["device_id"], data["job_id"] = stream_name.split(".")
    stream_name = stream_name.replace(".", "_")
    args.output_folder = os.path.join(args.output_folder, stream_name)
    # Collect FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    print("Current fps of the video is: {}".format(fps))
    # get height, width of video
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("The size of the video is: {}".format(str((size))))
    save_folder = args.output_folder
    save_video_folder = os.path.join(save_folder, "inference_video", args.type)
    save_meta_folder = os.path.join(save_folder, "inference_video", "inference_json")
    video_id = 1
    if not os.path.exists(save_video_folder):
        os.makedirs(save_video_folder, exist_ok=False)
        os.makedirs(save_meta_folder, exist_ok=False)

    fourcc = cv2.VideoWriter_fourcc(*"XVID")  # MP42/VP6f
    success, frame = cap.read()

    tot = 0
    inference_id = 0
    # Set skip_frame here for sampling purpose, disable by setting to -1
    skip_frame = args.skip_frame
    save_seconds = args.save_second
    timedelta = datetime.timedelta(seconds=save_seconds)
    total_frames = save_seconds * fps

    output_video_name = os.path.join(save_video_folder, stream_name + "_" + str(video_id))
    output_metadata_name = os.path.join(save_meta_folder, stream_name + "_" + str(video_id))
    outVideo = cv2.VideoWriter(output_video_name + output_affix, fourcc,
                               fps, size)

    while success:
        if tot < total_frames-1:
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
            # cv2.waitKey(25)
        else:
            outVideo.release()
            # save json info here. Add inference result later.
            end_time = start_time + timedelta
            data["end_timestamp"] = time.mktime(end_time.timetuple())
            data["video_name"] = os.path.join(args.type, stream_name + "_" + str(video_id) + output_affix)
            data["prob"] = 0
            save_metadata(data, output_metadata_name + ".json")
            video_list.append([os.path.join(args.type, stream_name + "_" + str(video_id)), tot, start_time, end_time])
            tot = 0  # reset frame count and save a new video
            video_id += 1
            output_video_name = os.path.join(save_video_folder, stream_name + "_" + str(video_id))
            output_metadata_name = os.path.join(save_meta_folder, stream_name + "_" + str(video_id))
            outVideo = cv2.VideoWriter(output_video_name + output_affix, fourcc,
                                       fps, size)
            if video_id%args.inference_count==0:
                with open(os.path.join(args.output_folder, "inference_list_{}.txt".format(inference_id)), 'w') as file_writer:
                    for ele in video_list:
                        if ele[1] > 0:
                            file_writer.writelines(" ".join(map(str, ele[:2])) + " -1\n")
                inference_id += 1
                del video_list[:]

        print(
            "{} / {} frames gathered for video {}!".format(tot, int(total_frames),
                                                           stream_name + "_" + str(video_id) + output_affix))
    outVideo.release()
    cap.release()


def wrapper(n):
    main(urls[n], n)


def pooled(n):
    # By default, our pool will have
    # numproc slots
    with multiprocessing.Pool() as pool:
       pool.map(wrapper, range(n))


if __name__ == "__main__":
    urls = []
    with open("streaming_urls.txt", 'r') as file_loader:
        for path in file_loader:
            urls.append(path.replace("\n", ""))
            #urls.append(path.replace("\n", "").split("/")[-1].replace(".flv", "").replace(".", "_"))
    #thread_count = len(urls)
    thread_count = 2
    pooled(thread_count)

