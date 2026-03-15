import cv2
import torch
import torch.onnx
import onnx
from models import AutoDriveNet

print("🚀 代码开始运行...")

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ 设备设置完成: {device}")

# 加载训练好的 PyTorch 模型
checkpoint_path = "./veps_0414.pth"
print(f"🔍 正在加载模型: {checkpoint_path}")

try:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = AutoDriveNet().to(device)
    model.load_state_dict(checkpoint['model'], strict=False)
    model.eval()
    print("✅ PyTorch 模型加载完成")
except Exception as e:
    print(f"❌ 加载模型失败: {e}")
    exit()

# 加载测试图像
img_path = "./results/2.jpg"
print(f"🔍 正在加载测试图片: {img_path}")

try:
    img = cv2.imread(img_path)
    img = cv2.resize(img, (160, 120))  # 假设输入大小是 160x120
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    print("✅ 测试图片加载成功")
except Exception as e:
    print(f"❌ 读取图片失败: {e}")
    exit()

# 预处理
try:
    img = torch.from_numpy(img.copy()).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0).to(device)  # (B, C, H, W)
    print("✅ 预处理完成")
except Exception as e:
    print(f"❌ 预处理失败: {e}")
    exit()

# 导出 ONNX
onnx_path = "results/veps_0414.onnx"
print(f"🔍 开始导出 ONNX: {onnx_path}")

try:
    torch.onnx.export(
        model, img, onnx_path,
        export_params=True,
        opset_version=11,  # ONNX 版本
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"✅ ONNX 模型已保存到 {onnx_path}")
except Exception as e:
    print(f"❌ ONNX 导出失败: {e}")
