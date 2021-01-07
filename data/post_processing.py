import os
import cv2
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


def process_result(ann_file):
    results = []
    with open(ann_file, "r") as file_reader:
        for line in file_reader:
            raw_data = line.split(" ")
            path, pred = raw_data[0], int(raw_data[-1])
            results.append([path, pred])
    return results


def generate_inference_video(target_folder, results, video_affix, out_name):
    if not results:
        print("No inference video found!")
        return
    video_name = os.path.join(target_folder, results[0][0]+"."+video_affix)
    cameraCapture = cv2.VideoCapture(video_name)
    assert cameraCapture.isOpened(), "failed to load video!"
    rotateCode = check_rotation(video_name)
    size = (int(cameraCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cameraCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    if rotateCode in [0, 2]:
        size = size[1], size[0]
    fps = cameraCapture.get(cv2.CAP_PROP_FPS)
    print("FPS for video {} is: {}".format(video_name, fps))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    outVideo = cv2.VideoWriter(
        os.path.join(target_folder, out_name + "." + video_affix),
        fourcc, fps, size)
    for ele in results:
        path, pred = ele
        cameraCapture = cv2.VideoCapture(os.path.join(target_folder, path + "." + video_affix))
        assert cameraCapture.isOpened(), "failed to load video!"
        success, frame = cameraCapture.read()
        if rotateCode is not None:
            frame = correct_rotation(frame, rotateCode)
        while success:
            if pred>0:
                # It is positive
                cv2.putText(frame, "False", (100, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 0, 255), 3)
            outVideo.write(frame)
            success, frame = cameraCapture.read()
            if rotateCode is not None:
                frame = correct_rotation(frame, rotateCode)
        cameraCapture.release()
    outVideo.release()
    return

if __name__ == "__main__":
    """
    Given 1. inference video and 2. inference result
    merge video together for visualization purpose
    """
    ann_file = "ucf101/inference_result.txt"
    target_folder = "ucf101/inference_video"
    video_affix = "avi"
    out_name = "inference_video"
    results = process_result(ann_file)
    generate_inference_video(target_folder, results, video_affix, out_name)
