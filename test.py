import os
import argparse
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model
from dataset import denorm
import warnings
warnings.filterwarnings("ignore")   

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.RandomCrop(512),
                            transforms.ToTensor(),
                            normalize])

def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer')
    parser.add_argument('-c', '--content', required=True, help='Content image path')
    parser.add_argument('-s', '--style', required=True, help='Style image path')
    parser.add_argument('-o', '--output_name', default=None, help='Output filename prefix')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='Style blend factor')
    parser.add_argument('-g', '--gpu', type=int, default=0, help='GPU ID, negative for CPU')
    parser.add_argument('-m', '--model_state_path', default='adain_best.pth', help='Path to model weights')
    
    args = parser.parse_args()

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device(f'cuda:{args.gpu}')
    else:
        print("CUDA not available, using CPU")
        device = torch.device('cpu')


    model = Model(device).to(device)
    model.load_state_dict(torch.load(args.model_state_path, map_location=device))
    model.eval()

    content_img = Image.open(args.content).convert('RGB')
    style_img = Image.open(args.style).convert('RGB')

    c_tensor = trans(content_img).unsqueeze(0).to(device)
    s_tensor = trans(style_img).unsqueeze(0).to(device)
    
    os.makedirs('content_resized', exist_ok=True)
    os.makedirs('style_resized', exist_ok=True)
    save_image(denorm(c_tensor, device='cpu'), os.path.join('content_resized', os.path.basename(args.content)))
    save_image(denorm(s_tensor, device='cpu'), os.path.join('style_resized', os.path.basename(args.style)))

    with torch.no_grad():
        output = model.generate(c_tensor, s_tensor, args.alpha)

    output_denorm = denorm(output, device)

    if args.output_name is None:
        c_base = os.path.splitext(os.path.basename(args.content))[0]
        args.output_name = c_base[-4:]

    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    output_path = os.path.join(save_dir, f"{args.output_name}.jpg")
    save_image(output_denorm, output_path)
    
    print(f"Result saved at: {output_path}")

if __name__ == "__main__":
    main()
