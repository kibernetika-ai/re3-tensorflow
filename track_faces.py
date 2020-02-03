import argparse
import os

import cv2
import numpy as np

from tracker import faces_tracker

np.set_printoptions(precision=6)
np.set_printoptions(suppress=True)


def track_faces(source: str = None, output: str = None, each_frame: int = 1, tracker: faces_tracker.FacesTracker = None):

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
        fps = src.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'MP4V')
        width = int(src.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(src.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if output is not None:
            video_writer = cv2.VideoWriter(output, fourcc, fps / each_frame, frameSize=(width, height))

    try:

        while True:

            cnt += 1
            ret_val, img = src.read()

            if source_is_file and cnt % each_frame > 0:
                continue

            if img is None:
                print('video capturing finished')
                break

            if not source_is_file:
                img = cv2.flip(img, 1)

            faces, tracked = tracker.track(img)

            img_avg = (img.shape[1] + img.shape[0]) / 2
            for face in faces:
                box = face.bbox
                cv2.rectangle(img,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        (0, 255, 0), 2)
                cv2.putText(img, face.label(), (int(box[0]), int(box[1] - img_avg / 100)),
                            cv2.FONT_HERSHEY_SIMPLEX, img_avg / 1600, (0, 255, 0),
                            thickness=1, lineType=cv2.LINE_AA)

            for tr in tracked:
                cv2.rectangle(img,
                        (int(tr[0]), int(tr[1])),
                        (int(tr[2]), int(tr[3])),
                        (0, 180, 0), 1)

            cv2.imshow('Webcam', img)

            if video_writer is not None:
                video_writer.write(img)

            keyPressed = cv2.waitKey(1)
            if keyPressed == 27 or keyPressed == 1048603:
                break  # esc to quit

    except (KeyboardInterrupt, SystemExit) as e:
        print('Caught %s: %s' % (e.__class__.__name__, e))

    finally:
        cv2.destroyAllWindows()
        if video_writer is not None:
            print('Written video to %s.' % output)
            video_writer.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video-source',
        type=str,
        default=None,
        help='Video source (web cam if not set)',
    )
    parser.add_argument(
        '--video-output',
        type=str,
        default=None,
        help='Video target',
    )
    parser.add_argument(
        '--video-each-frame',
        type=int,
        default=1,
        help='Process each N frame (for video file only)',
    )
    args = parser.parse_args()

    tracker = faces_tracker.FacesTracker()
    track_faces(
        source=args.video_source,
        output=args.video_output,
        each_frame=args.video_each_frame,
        tracker=tracker
    )
