import argparse
import os
from datetime import datetime

import cv2
import numpy as np

from tracker import faces_tracker
from tracker import re3_tracker

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


def track_faces(
    source: str = None,
    output: str = None,
    each_frame: int = 1,
    face_tracker: faces_tracker.FacesTracker = None,
    screen: bool = True,
):

    if source is None:
        source_is_file = False
        source = 0
    elif os.path.isfile(source):
        source_is_file = True
    else:
        raise RuntimeError(f"source video {source} is not found")

    src = cv2.VideoCapture(source)
    cnt = 0
    screen_init = True
    font = cv2.FONT_HERSHEY_SIMPLEX
    black = (0, 0, 0)
    green = (0, 255, 0)

    video_writer = None
    if source_is_file:
        if output is not None:
            if os.path.exists(output):
                os.remove(output)
            fps = src.get(cv2.CAP_PROP_FPS)
            fourcc = cv2.VideoWriter_fourcc(*"MP4V")
            width = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
            video_writer = cv2.VideoWriter(
                output, fourcc, fps / each_frame, frameSize=(width, height)
            )

    try:

        while True:

            cnt += 1
            ret_val, img = src.read()

            if cnt % 10 == 0:
                print(f"{datetime.utcnow()}: processed {cnt} frames")


            if source_is_file and cnt % each_frame > 0:
                continue

            if img is None:
                print("video capturing finished")
                break

            if not source_is_file:
                img = cv2.flip(img, 1)

            faces = face_tracker.track(img)

            img_avg = (img.shape[1] + img.shape[0]) / 2
            f_size = img_avg / 2400
            for face in faces:
                box = face.bbox
                color = green if face.confirmed else (0, 127, 0)
                cv2.rectangle(
                    img,
                    (int(box[0]), int(box[1])),
                    (int(box[2]), int(box[3])),
                    color,
                    2 if face.just_detected else 1,
                )
                lbl = f"Track {face.id}" if face.confirmed else "---"
                if face.class_id:
                    lbl = f"{lbl}, class {face.class_id}"
                if face.prob:
                    lbl = "{}, prb {:.2f}".format(lbl, face.prob)
                if face.confirm_count > 0:
                    lbl = f"{lbl}, cf {face.confirm_count}"
                if face.to_remove:
                    lbl = f"{lbl}, rm {face.remove_count}"
                crd = (int(box[0]), int(box[1] - img_avg / 100))
                cv2.putText(
                    img, lbl, crd, font, f_size, black, thickness=3,
                )
                cv2.putText(
                    img, lbl, crd, font, f_size, color, thickness=1,
                )

            for i, l in enumerate(tracker.log):
                crd = (30, 50 + i * 22)
                cv2.putText(img, l, crd, font, f_size, black, thickness=3)
                cv2.putText(img, l, crd, font, f_size, green, thickness=1)

            if screen:
                if screen_init:
                    cv2.namedWindow("Output", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Output", int(img.shape[1] / 2), int(img.shape[0] / 2))
                    screen_init = False
                cv2.imshow("Output", img)

            if video_writer is not None:
                video_writer.write(img)

            key_pressed = cv2.waitKey(1)
            if key_pressed == 27 or key_pressed == 1048603:
                break  # esc to quit

    except (KeyboardInterrupt, SystemExit) as e:
        print("Caught %s: %s" % (e.__class__.__name__, e))

    finally:
        if screen:
            cv2.destroyAllWindows()
        if video_writer is not None:
            print("Written video to %s." % output)
            video_writer.release()

        profiling = tracker.profiler.get_and_reset_dict()
        print("Profiling:")
        for k in profiling:
            print(f" - {k}: {profiling[k]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-source",
        type=str,
        default=None,
        help="Video source (web cam if not set)",
    )
    parser.add_argument(
        "--video-output", type=str, default=None, help="Video target",
    )
    parser.add_argument(
        "--video-frame-freq",
        type=int,
        default=1,
        help="Video processing frames frequency: process each N frame (for video file only)",
    )
    parser.add_argument(
        "--video-detect-freq",
        type=int,
        default=10,
        help="Faces detection frequency: detect for each N processed frame (for video file only)",
    )
    parser.add_argument(
        "--re3-checkpoint-dir",
        type=str,
        default="./models/re3-tracker/checkpoints",
        help="re3 model checkpoints dir",
    )
    parser.add_argument(
        "--face-detection-path",
        type=str,
        default=None,
        help="face detection model path (default openvino face-detection-adas-0001)",
    )
    parser.add_argument(
        "--facenet-path", type=str, default=None, help="facenet model path",
    )
    parser.add_argument(
        "--screen", action="store_true", help="Show result on screen",
    )
    args = parser.parse_args()

    re3_tracker.SPEED_OUTPUT = False
    tracker = faces_tracker.FacesTracker(
        re3_checkpoint_dir=args.re3_checkpoint_dir,
        face_detection_path=args.face_detection_path,
        facenet_path=args.facenet_path,
        detect_each=args.video_detect_freq,
    )
    track_faces(
        source=args.video_source,
        output=args.video_output,
        each_frame=args.video_frame_freq,
        face_tracker=tracker,
        screen=args.screen,
    )
