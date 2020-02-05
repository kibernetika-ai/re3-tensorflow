import argparse
import os

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
    tracker: faces_tracker.FacesTracker = None,
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

            if cnt % 100 == 0:
                print(f"processed {cnt} frames")

            if source_is_file and cnt % each_frame > 0:
                continue

            if img is None:
                print("video capturing finished")
                break

            if not source_is_file:
                img = cv2.flip(img, 1)

            faces = tracker.track(img)

            img_avg = (img.shape[1] + img.shape[0]) / 2
            for face in faces:
                box = face.bbox
                color = (0, 255, 0) if face.confirmed else (0, 127, 0)
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
                if face.confirm_count > 0:
                    lbl = f"{lbl}, cf {face.confirm_count}"
                if face.to_remove:
                    lbl = f"{lbl}, rm {face.remove_count}"
                crd = (int(box[0]), int(box[1] - img_avg / 100))
                cv2.putText(
                    img,
                    lbl,
                    crd,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    img_avg / 2400,
                    (0, 0, 0),
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    lbl,
                    crd,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    img_avg / 2400,
                    color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            for i, l in enumerate(tracker.get_log()):
                crd = (30, 50 + i * 22)
                cv2.putText(
                    img,
                    l,
                    crd,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    img_avg / 2400,
                    (0, 0, 0),
                    thickness=3,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    img,
                    l,
                    crd,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    img_avg / 2400,
                    (0, 255, 0),
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

            if screen:
                cv2.imshow("Webcam", img)

            if video_writer is not None:
                video_writer.write(img)

            keyPressed = cv2.waitKey(1)
            if keyPressed == 27 or keyPressed == 1048603:
                break  # esc to quit

    except (KeyboardInterrupt, SystemExit) as e:
        print("Caught %s: %s" % (e.__class__.__name__, e))

    finally:
        if screen:
            cv2.destroyAllWindows()
        if video_writer is not None:
            print("Written video to %s." % output)
            video_writer.release()


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
        "--video-each-frame",
        type=int,
        default=1,
        help="Process each N frame (for video file only)",
    )
    parser.add_argument(
        "--re3-checkpoint-dir",
        type=str,
        default="./models/re3-tracker/checkpoints",
        help="re3 model checkpoints dir",
    )
    parser.add_argument(
        "--facenet-path",
        type=str,
        default="./models/model-facenet-pretrained-1.0.0-vgg-openvino-cpu/facenet.xml",
        help="facenet model path",
    )
    parser.add_argument(
        "--screen", action="store_true", help="Show result on screen",
    )
    args = parser.parse_args()

    re3_tracker.SPEED_OUTPUT = False
    tracker = faces_tracker.FacesTracker(
        re3_checkpoint_dir=args.re3_checkpoint_dir,
        facenet_path=args.facenet_path,
    )
    track_faces(
        source=args.video_source,
        output=args.video_output,
        each_frame=args.video_each_frame,
        tracker=tracker,
        screen=args.screen,
    )
