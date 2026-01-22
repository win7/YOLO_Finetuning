import onnx
# model = onnx.load("yolov10n_git.onnx")
model = onnx.load("yolov10n.onnx")
print(model.opset_import)
