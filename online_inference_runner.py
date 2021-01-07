import os
from glob import glob
import argparse
import subprocess
from timeit import default_timer as timer

def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('root', help='location for root folder', default='/Volumes/external\ drive/stream\ data/')
    parser.add_argument('--stream_name', help='the name of the stream', default='YZ0010202005_096c8e2a684b4c879726cc51bebe4712/')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    print("The path for inference file : {}".format(os.path.join(args.root, args.stream_name)))
    cur_count, inference_list = 0, glob(os.path.join(args.root, args.stream_name, "*.txt"))
    inference_count = len(inference_list)
    print("Found {} inference files".format(inference_count))
    assert inference_count>0, "No inference files found!"
    while cur_count<inference_count:
        # run extract_rgb shell
        start = timer()
        subprocess.call(['python', 'build_rawframes.py',
                         os.path.join(args.root, args.stream_name, "inference_video"),
                         os.path.join(args.root, args.stream_name, "rawframes"),
                         "--level", "2", "--ext", "avi", "--online_inference", "--inference_id", str(cur_count)])
        # run inference shell
        with open("test_configs/TSN/ucf101/tsn_test.py", 'r') as file_loader:
            raw_code = file_loader.readlines()
            for idx, code in enumerate(raw_code):
                if "data_root = " in code:
                    raw_code[idx] = "data_root = "+ "'" + os.path.join(os.path.join(args.root, args.stream_name, "rawframes")) \
                            + "'" + "\n"
                if "ann_file" in code:
                    ann_loc_group = code.split("ann_file=")
                    ann_loc = ann_loc_group.pop(-1)
                    ann_loc = ann_loc.replace("'","").replace(",","").replace(".txt","").replace("\n","").split("/")[-1]+"_{}.txt".format(cur_count)
                    ann_loc = os.path.join(args.root, args.stream_name, ann_loc)
                    ann_loc = "'{}',\n".format(ann_loc)
                    ann_loc_group.append(ann_loc)
                    code = "ann_file=".join(ann_loc_group)
                    raw_code[idx] = code

        with open("test_configs/TSN/ucf101/tsn_test_{}.py".format(cur_count), 'w') as file_writer:
            for code in raw_code:
                file_writer.writelines(code)
        start = timer()
        subprocess.call(['bash', 'tools/dist_online_inference_recognizer.sh', 'test_configs/TSN/ucf101/tsn_test_{}.py'.format(cur_count), 'work_dirs/tsn_2d_rgb_bninception_seg_3_f1s1_b32_g8/latest.pth', '1', '--use_softmax', "--json_out", os.path.join(args.root, args.stream_name,"inference_video", "inference_json"), "--root", os.path.join(args.root, args.stream_name), "--inference_id", str(cur_count)])

        # Keep going...
        inference_list = glob(os.path.join(args.root, args.stream_name, "*.txt"))
        inference_count = len(inference_list)
        cur_count += 1
        print(cur_count, inference_count)
        end = timer()
        print("Time cost for conduct online batch inference: {} seconds".format(end - start))
    print("Online inference complete!")
    
    
if __name__ == "__main__":
    main()
