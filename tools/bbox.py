import typing


def box_intersection(box_a: typing.List[int], box_b: typing.List[int]):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    d = float(box_a_area + box_b_area - inter_area)
    if d == 0:
        return 0
    iou = inter_area / d
    return iou


def box_includes(outer: typing.List[int], inner: typing.List[int]):
    return outer[0] <= inner[0] and \
           outer[1] <= inner[1] and \
           outer[2] >= inner[2] and \
           outer[3] >= inner[3]

