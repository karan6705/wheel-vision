from ultralytics import YOLO

# 1) Load your fine-tuned model
model = YOLO('runs/detect/train2/weights/best.pt')

# 2) Point to any video, adjust 'source=' to your filename
results = model.predict(
    source='test/drive.mp4',  # replace with your file
    conf=0.25,                     # keep detections ≥25% confidence
    save=True,                     # write out an annotated video
    show=False                     # set True to preview live
)

print('✅ Done.  See annotated clip at:', results[0].path)
