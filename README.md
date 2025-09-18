# YOLOv9 with ONNX & ONNXRuntime 練習
[ONNX 練習](https://github.com/danielsyahputra/yolov9-onnx)

## Anaconda 新建環境
```cmd=
conda create -n onnx python=3.10 -y
conda activate onnx
````

## 取得 YOLOv9 with ONNX 專案並安裝需求
若沒安裝 Git：
```cmd=
conda install -c anaconda git -y
git --version
```

### Github 檔案下載 - [danielsyahputra/yolov9-onnx](https://github.com/danielsyahputra/yolov9-onnx.git)
``` cmd=
git clone https://github.com/danielsyahputra/yolov9-onnx.git

cd yolov9-onnx

pip install -r requirements.txt
```
### ONNX Runtime
For Nvidia GPU computers: 
```cmd=
pip install onnxruntime-gpu
```

#### 遇到的問題
1. numpy 與 python 版本不相容：
```cmd=
pip install --upgrade pip setuptools wheel
pip install --upgrade numpy
```

2. `openvino-dev` 和 `onnx / scipy / numpy / networkx / protobuf` 等版本之間有相依性衝突。
>先分開安裝，再裝剩下的套件
```cmd=
pip install numpy==1.23.5 scipy==1.10.1 onnx==1.13.1 protobuf==3.20.3 networkx==2.8.8

pip install openvino-dev[onnx]==2023.0.0

pip install -r requirements.txt --no-deps
```
> `--no-deps` 可以避免 pip 又去回溯版本，把依賴衝突壓下來

### 推論（Inference on images）
如果要辨識整個資料集，要改 main.py：
```python=
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
```
```python=
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
```
再執行：
```anaconda=
python main.py --image --source C:\Users\user\yolov9-onnx\carplate\train --weights yolov9-c.onnx --classes metadata.yaml
```

