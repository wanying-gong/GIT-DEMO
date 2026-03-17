import torch
import torchvision.transforms as transforms
from torchvision.models import resnet50, ResNet50_Weights
from PIL import Image
import requests
from io import BytesIO

def predict_image(image_path_or_url):
    # 1. 加载预训练模型 (ResNet50)
    # 默认使用 Imagenet 数据集预训练的权重
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()  # 切换到评估模式

    # 2. 定义图片预处理 (调整大小、归一化等)
    preprocess = weights.transforms()

    # 3. 加载图片
    try:
        if image_path_or_url.startswith('http'):
            response = requests.get(image_path_or_url)
            img = Image.open(BytesIO(response.content)).convert('RGB')
        else:
            img = Image.open(image_path_or_url).convert('RGB')
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # 4. 预处理图片并增加 batch 维度 (1, 3, H, W)
    batch = preprocess(img).unsqueeze(0)

    # 5. 模型推理
    with torch.no_grad():
        output = model(batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        
        # 获取 Top-5 结果
        top5_prob, top5_catid = torch.topk(probabilities, 5)

    print("-" * 30)
    print(f"图片识别结果 (Top-5):")
    for i in range(top5_prob.size(0)):
        category_name = weights.meta["categories"][top5_catid[i]]
        score = top5_prob[i].item()
        print(f"{i+1}. {category_name:20} | 置信度: {100 * score:.2f}%")
    print("-" * 30)

if __name__ == "__main__":
    # 示例用法:
    # 1. 使用网络图片链接 (这是一张猫的图片)
    test_url = "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba?auto=format&fit=crop&q=80&w=400"
    print(f"正在识别网络图片...")
    predict_image(test_url)

    # 2. 如果你有本地图片，可以取消下面这一行的注释并替换路径
    # predict_image("cat.jpg")
