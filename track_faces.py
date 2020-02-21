import argparse
import base64
import json
import os
from datetime import datetime

import cv2
import jinja2
import numpy as np

from report import db
from tracker import faces_tracker
from tracker import re3_tracker
from utils import utils

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-source",
        default=None,
        help="Video source (web cam if not set)",
    )
    parser.add_argument(
        "--video-output", help="Video target",
    )
    parser.add_argument(
        "--report-dir", default='reports', help="Report output dir",
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
        "--log-each-frame",
        type=int,
        default=100,
        help="StdOut record after each N frame",
    )
    parser.add_argument(
        "--re3-checkpoint-dir",
        default="./models/re3-tracker/checkpoints",
        help="re3 model checkpoints dir",
    )
    parser.add_argument(
        "--face-detection-path",
        help="face detection model path (default openvino face-detection-adas-0001)",
    )
    parser.add_argument(
        "--age_model_path",
        help="path to the age model (default openvino age-gender-estimation",
    )
    parser.add_argument(
        "--head_pose_model_path",
        help="path to the head-pose model (default openvino head-pose-estimation",
    )
    parser.add_argument(
        "--facenet-path", help="facenet model path",
    )
    parser.add_argument(
        "--screen", action="store_true", help="Show result on screen",
    )
    return parser.parse_args()


def track_faces(source: str = None,
                output: str = None,
                each_frame: int = 1,
                face_tracker: faces_tracker.FacesTracker = None,
                screen: bool = True,
                log_each_frame: int = 100, **kwargs):
    if source is None:
        source_is_file = False
        source = 0
    elif os.path.isfile(source):
        source_is_file = True
    # elif os.path.isdir(source):
    #     source_is_file = True
    else:
        raise RuntimeError(f"source video {source} is not found")

    src = cv2.VideoCapture(source)
    cnt = 0
    screen_init = True
    font = cv2.FONT_HERSHEY_SIMPLEX
    black = (0, 0, 0)
    green = (0, 255, 0)

    video_writer = None
    fps = None
    if source_is_file:
        fps = src.get(cv2.CAP_PROP_FPS)
        if output is not None:
            if os.path.exists(output):
                os.remove(output)
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

            if cnt % log_each_frame == 0:
                msg = f"{datetime.utcnow()}: processed {cnt} frames"
                if fps is not None:
                    msg = f"{msg}, {cnt / fps:.2f} sec"
                print(msg)

            if source_is_file and cnt % each_frame > 0:
                continue

            if img is None:
                print("video capturing finished")
                break

            if not source_is_file:
                img = cv2.flip(img, 1)

            faces = face_tracker.track(img.copy())

            img_avg = (img.shape[1] + img.shape[0]) / 2
            f_size = img_avg / 2400
            if screen or video_writer is not None:
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

                for i, msg in enumerate(tracker.log):
                    crd = (30, 50 + i * 22)
                    cv2.putText(img, msg, crd, font, f_size, black, thickness=3)
                    cv2.putText(img, msg, crd, font, f_size, green, thickness=1)

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

        profiling = face_tracker.profiler.get_and_reset_dict()
        print("Profiling:")
        for k in profiling:
            print(f" - {k}: {profiling[k]}")

        # print("report data: ", face_tracker.report(fps=fps))

        save_report(kwargs['report_dir'], face_tracker, fps=fps)


def save_report(report_dir, face_tracker, fps=30):
    intervals, class_images = face_tracker.report(fps=fps)

    tpl = jinja2.Template(utils.template)
    # Group by classes
    by_classes = {}
    for interval in intervals:
        class_id = interval['class_id']
        if class_id is None:
            continue

        if class_id in by_classes:
            by_classes[class_id].append(interval)
        else:
            by_classes[class_id] = [interval]

    def value_func(xy):
        imax = 0
        x = xy[1]
        for i in x:
            if i['end'] - i['start'] > imax:
                imax = i['end'] - i['start']
        return imax

    # Sort by duration
    by_classes = sorted(by_classes.items(), key=value_func, reverse=True)

    # Ensure report dir
    if not os.path.isdir(report_dir):
        os.makedirs(report_dir)

    db_api = db.ReportDB(os.path.join(report_dir, 'report.db'), clear_report=True)
    db_api.begin_transaction()

    report = []
    for class_id, v in by_classes:
        age_average = int(round(np.average([i['metadata']['age'] for i in v])))
        gender_average = np.average([i['metadata']['gender'] for i in v])
        gender = 'male' if gender_average >= 0.5 else 'female'

        name = f'Person_{class_id}'

        # Insert into DB
        db_data = {
            'id': str(class_id),
            'name': name,  # just person for now
            'meta': json.dumps({'age': age_average, 'gender': gender}),
            'age': age_average,
            'gender': gender,
        }
        db_api.insert_data('classes', db_data, commit=False)

        name = f'{name}; age={age_average}'
        name = f'{name}; gender={gender}'

        for interval in v:
            db_data = {
                'class_id': str(class_id),
                'start': interval['start'],
                'end': interval['end']
            }
            db_api.insert_data('intervals', db_data, commit=False)

        # TODO: insert attention intervals

        intervals = '; '.join(f'{i["start"]:.1f}-{i["end"]:.1f}' for i in v)
        duration = max([i['end'] - i['start'] for i in v])
        images = []
        for image in class_images[class_id]:
            encoded = cv2.imencode('.jpg', image[:, :, ::-1])[1].tostring()
            encoded = base64.standard_b64encode(encoded).decode()
            images.append(encoded)

        report.append([
            name, None, images, intervals, duration
        ])

    db_api.end_transaction()

    html = tpl.render(data=report)
    report_file = os.path.join(report_dir, 'top100_faces.html')
    with open(report_file, 'w') as f:
        f.write(html)

    print(f'Report is saved to {report_file}')


if __name__ == "__main__":
    args = parse_args()

    kwargs = vars(args)

    utils.configure_logging()

    re3_tracker.SPEED_OUTPUT = False
    tracker = faces_tracker.FacesTracker(
        detect_each=args.video_detect_freq,
        **kwargs
    )
    track_faces(
        source=args.video_source,
        output=args.video_output,
        each_frame=args.video_frame_freq,
        face_tracker=tracker,
        **kwargs,
    )
