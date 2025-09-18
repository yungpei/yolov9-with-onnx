import os
import cv2
from pathlib import Path

from yolov9 import YOLOv9


def get_detector(args):
    weights_path = args.weights
    classes_path = args.classes
    source_path = args.source

    assert os.path.isfile(weights_path), f"There's no weight file with name {weights_path}"
    assert os.path.isfile(classes_path), f"There's no classes file with name {classes_path}"

    # 改這裡：source 可以是檔案或資料夾，不要用 isfile 強制檢查
    assert os.path.exists(source_path), f"There's no source at {source_path}"

    if args.image:
        if os.path.isfile(source_path):
            image = cv2.imread(source_path)
            h, w = image.shape[:2]
        elif os.path.isdir(source_path):
            # 先隨便讀一張圖來抓寬高
            import glob
            img_files = glob.glob(os.path.join(source_path, "*.jpg")) + glob.glob(os.path.join(source_path, "*.png"))
            if len(img_files) == 0:
                raise ValueError(f"No images found in {source_path}")
            image = cv2.imread(img_files[0])
            h, w = image.shape[:2]
        else:
            raise ValueError("Source must be an image file or directory")
    elif args.video:
        cap = cv2.VideoCapture(source_path)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    detector = YOLOv9(
        model_path=weights_path,
        class_mapping_path=classes_path,
        original_size=(w, h),
        score_threshold=args.score_threshold,
        conf_thresold=args.conf_threshold,
        iou_threshold=args.iou_threshold,
        device=args.device
    )
    return detector

def inference_on_images(args):
    print("[INFO] Initialize Model")
    detector = get_detector(args)

    source_path = Path(args.source)
    if source_path.is_dir():
        img_paths = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))
    else:
        img_paths = [source_path]

    os.makedirs("output", exist_ok=True)
    os.makedirs("plates", exist_ok=True)

    for img_path in img_paths:
        print(f"[INFO] Inference Image: {img_path}")
        image = cv2.imread(str(img_path))
        detections = detector.detect(image)

        # 畫框
        detector.draw_detections(image, detections=detections)
        output_path = f"output/{img_path.name}"
        cv2.imwrite(output_path, image)
        print(f"[INFO] Saved result to {output_path}")

        # 裁切車牌
        for i, det in enumerate(detections):
            # 如果 det 是 dict
            if isinstance(det, dict):
                x1 = int(det.get('x1', 0))
                y1 = int(det.get('y1', 0))
                x2 = int(det.get('x2', 0))
                y2 = int(det.get('y2', 0))
                conf = det.get('confidence', 1.0)
                cls_id = det.get('class_index', 0)
            # 如果 det 是 list 或 tuple
            elif isinstance(det, (list, tuple)):
                if len(det) >= 4:
                    x1, y1, x2, y2 = map(int, det[:4])
                    conf = det[4] if len(det) > 4 else 1.0
                    cls_id = det[5] if len(det) > 5 else 0
                else:
                    continue  # 跳過不完整的檢測
            else:
                continue  # 其他類型跳過

            # 裁切車牌
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            plate = image[y1:y2, x1:x2]
            if plate.size > 0:
                plate_filename = f"plates/{img_path.stem}_plate{i}.jpg"
                cv2.imwrite(plate_filename, plate)
                print(f"[INFO] Saved cropped plate to {plate_filename}")

        if args.show:
            cv2.imshow("Result", image)
            cv2.waitKey(500)  # 每張停 0.5 秒
    cv2.destroyAllWindows()


def inference_on_video(args):
    print("[INFO] Intialize Model")
    detector = get_detector(args)

    cap = cv2.VideoCapture(args.source)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    os.makedirs("output", exist_ok=True)
    writer = cv2.VideoWriter('output/result.avi', cv2.VideoWriter_fourcc(*'MJPG'), video_fps, (w, h))

    print("[INFO] Inference on Video")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detections = detector.detect(frame)
        detector.draw_detections(frame, detections=detections)
        writer.write(frame)
        cv2.imshow("Result", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    print("[INFO] Finish. Saving result to output/result.avi")

if __name__=="__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Argument for YOLOv9 Inference using ONNXRuntime")

    parser.add_argument("--source", type=str, required=True, help="Path to image or video file")
    parser.add_argument("--weights", type=str, required=True, help="Path to yolov9 onnx file")
    parser.add_argument("--classes", type=str, required=True, help="Path to list of class in yaml file")
    parser.add_argument("--score-threshold", type=float, required=False, default=0.1)
    parser.add_argument("--conf-threshold", type=float, required=False, default=0.4)
    parser.add_argument("--iou-threshold", type=float, required=False, default=0.4)
    parser.add_argument("--image", action="store_true", required=False, help="Image inference mode")
    parser.add_argument("--video", action="store_true", required=False)
    # parser.add_argument("--show", required=False, type=bool, default=True, help="Show result on pop-up window")
    parser.add_argument("--show", action="store_true", help="Show result on pop-up window")
    parser.add_argument("--device", type=str, required=False, help="Device use (cpu or cuda)", choices=["cpu", "cuda"], default="cpu")

    args = parser.parse_args()

    if args.image:
        inference_on_images(args=args)
    elif args.video:
        inference_on_video(args=args)
    else:
        raise ValueError("You can't process the result because you have not define the source type (video or image) in the argument")
    