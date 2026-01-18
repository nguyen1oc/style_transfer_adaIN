import os
import io
import uuid
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from model import Model
from dataset import denorm
import warnings
warnings.filterwarnings("ignore")

app = FastAPI(title="AdaIN Style Transfer")

UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
UPLOAD_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

if Path("static").exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")
    
USE_CPU = os.getenv("USE_CPU", "true").lower() == "true"

if USE_CPU:
    device = torch.device("cpu")
    print("Using CPU mode (forced)")
else:
    try:
        if torch.cuda.is_available():
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            device = torch.device("cuda")
            print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        else:
            device = torch.device("cpu")
            print("CUDA not available, using CPU")
    except Exception as e:
        device = torch.device("cpu")
        print(f"CUDA error detected, falling back to CPU: {str(e)}")

model = Model(device).to(device)
model_path = "weights/adain_best.pth"

if os.path.exists(model_path):
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        print(f"Model loaded successfully on {device}")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise
else:
    print(f"Warning: Model file not found at {model_path}")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

trans = transforms.Compose([transforms.RandomCrop(512),
                            transforms.ToTensor(),
                            normalize])


MAX_FILE_SIZE = 10 * 1024 * 1024  
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png'}

def process_image(image_bytes: bytes, filename: str) -> Image.Image:
    """Convert uploaded image bytes to PIL Image with validation"""
    file_ext = os.path.splitext(filename.lower())[1]
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file format. Only JPG, JPEG, and PNG are allowed. Got: {file_ext}"
        )
    
    if len(image_bytes) > MAX_FILE_SIZE:
        size_mb = len(image_bytes) / (1024 * 1024)
        max_mb = MAX_FILE_SIZE / (1024 * 1024)
        raise HTTPException(
            status_code=400,
            detail=f"File size too large: {size_mb:.2f}MB. Maximum allowed: {max_mb}MB"
        )
    
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image: {str(e)}")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = """
<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AdaIN Style Transfer</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            padding: 40px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }
        
        h1 {
            text-align: center;
            color: #333;
            margin-bottom: 10px;
            font-size: 2.5em;
        }
        
        .subtitle {
            text-align: center;
            color: #666;
            margin-bottom: 40px;
            font-size: 1.1em;
        }
        
        .upload-section {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .upload-box {
            border: 3px dashed #667eea;
            border-radius: 15px;
            padding: 30px;
            text-align: center;
            transition: all 0.3s;
            background: #f8f9fa;
            position: relative;
        }
        
        .upload-box:hover {
            border-color: #764ba2;
            background: #f0f0f0;
        }
        
        .upload-box.dragover {
            border-color: #764ba2;
            background: #e8e0f0;
        }
        
        .upload-label {
            display: block;
            cursor: pointer;
            color: #667eea;
            font-weight: bold;
            margin-bottom: 15px;
            font-size: 1.2em;
            position: relative;
            z-index: 5;
        }
        
        .file-input {
            display: none;
        }
        
        .preview-container {
            margin-top: 20px;
            min-height: 200px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-direction: column;
            position: relative;
        }
        
        .upload-icon {
            width: 80px;
            height: 80px;
            cursor: pointer;
            color: #667eea;
            transition: all 0.3s;
            pointer-events: auto;
            z-index: 2;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .upload-icon:hover {
            color: #764ba2;
            transform: scale(1.1);
        }
        
        .upload-icon svg {
            width: 100%;
            height: 100%;
        }
        
        .upload-hint-text {
            text-align: center;
            color: #999;
            font-size: 0.9em;
            margin-top: 30px;
            margin-bottom: 20px;
        }
        
        .preview-img {
            max-width: 100%;
            max-height: 300px;
            border-radius: 10px;
            box-shadow: 0 5px 15px rgba(0,0,0,0.2);
        }
        
        .alpha-control {
            margin: 30px 0;
            text-align: center;
        }
        
        .alpha-control label {
            display: block;
            margin-bottom: 10px;
            font-weight: bold;
            color: #333;
            font-size: 1.1em;
        }
        
        .alpha-control input[type="range"] {
            width: 100%;
            max-width: 400px;
            height: 8px;
            border-radius: 5px;
            background: #ddd;
            outline: none;
            -webkit-appearance: none;
        }
        
        .alpha-control input[type="range"]::-webkit-slider-thumb {
            -webkit-appearance: none;
            appearance: none;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            background: #667eea;
            cursor: pointer;
        }
        
        .alpha-value {
            display: inline-block;
            margin-top: 10px;
            padding: 5px 15px;
            background: #667eea;
            color: white;
            border-radius: 20px;
            font-weight: bold;
        }
        
        .btn-submit {
            display: block;
            width: 100%;
            max-width: 400px;
            margin: 30px auto;
            padding: 15px 40px;
            font-size: 1.2em;
            font-weight: bold;
            color: white;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            border-radius: 50px;
            cursor: pointer;
            transition: all 0.3s;
            box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
        }
        
        .btn-submit:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 20px rgba(102, 126, 234, 0.6);
        }
        
        .btn-submit:disabled {
            opacity: 0.6;
            cursor: not-allowed;
            transform: none;
        }
        
        .result-section {
            margin-top: 40px;
            padding-top: 40px;
            border-top: 2px solid #eee;
            text-align: center;
        }
        
        .result-img {
            max-width: 100%;
            max-height: 600px;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
            margin-top: 20px;
        }
        
        .loading {
            display: none;
            text-align: center;
            margin: 20px 0;
        }
        
        .loading.active {
            display: block;
        }
        
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        .error {
            color: #e74c3c;
            background: #ffeaea;
            padding: 15px;
            border-radius: 10px;
            margin: 20px 0;
            display: none;
        }
        
        .error.active {
            display: block;
        }
        
        .preview-wrapper {
            position: relative;
            display: inline-block;
            width: 100%;
        }
        
        .btn-remove {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(231, 76, 60, 0.9);
            color: white;
            border: none;
            border-radius: 50%;
            width: 35px;
            height: 35px;
            cursor: pointer;
            font-size: 18px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: all 0.3s;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            z-index: 10;
        }
        
        .btn-remove:hover {
            background: rgba(231, 76, 60, 1);
            transform: scale(1.1);
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>AdaIN Style Transfer</h1>
        <p class="subtitle">Transform your images with AI-powered style transfer</p>
        
        <form id="transferForm" enctype="multipart/form-data">
            <div class="upload-section">
                <div class="upload-box" id="contentBox">
                    <label for="contentImage" class="upload-label">Content Image</label>
                    <input type="file" id="contentImage" name="content_image" accept=".jpg,.jpeg,.png,image/jpeg,image/png" class="file-input" required>
                    <div class="preview-container">
                        <div class="upload-icon" id="contentUploadIcon">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                        </div>
                        <div class="preview-wrapper" style="display: none;">
                            <img id="contentPreview" class="preview-img">
                            <button type="button" class="btn-remove" id="removeContent" style="display: none;">×</button>
                        </div>
                    </div>
                </div>
                
                <div class="upload-box" id="styleBox">
                    <label for="styleImage" class="upload-label">Style Image</label>
                    <input type="file" id="styleImage" name="style_image" accept=".jpg,.jpeg,.png,image/jpeg,image/png" class="file-input" required>
                    <div class="preview-container">
                        <div class="upload-icon" id="styleUploadIcon">
                            <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                            </svg>
                        </div>
                        <div class="preview-wrapper" style="display: none;">
                            <img id="stylePreview" class="preview-img">
                            <button type="button" class="btn-remove" id="removeStyle" style="display: none;">×</button>
                        </div>
                    </div>
                </div>
            </div>
            
            <p class="upload-hint-text">Drag and drop or click to select image (JPG, PNG, max 10MB)</p>
            
            <div class="alpha-control">
                <label for="alpha">Style Strength (Alpha):</label>
                <input type="range" id="alpha" name="alpha" min="0" max="1" step="0.1" value="1.0">
                <div class="alpha-value" id="alphaValue">1.0</div>
            </div>
            
            <button type="submit" class="btn-submit" id="submitBtn">Transfer Style</button>
            
            <div class="loading" id="loading">
                <div class="spinner"></div>
                <p style="margin-top: 15px; color: #667eea; font-weight: bold;">Processing image...</p>
            </div>
            
            <div class="error" id="error"></div>
        </form>
        
        <div class="result-section" id="resultSection" style="display: none;">
            <h2 style="color: #333; margin-bottom: 20px;">Result</h2>
            <img id="resultImage" class="result-img" alt="Result">
            <div style="margin-top: 20px;">
                <a id="downloadLink" download="style_transfer_result.jpg" class="btn-submit" style="display: inline-block; text-decoration: none; width: auto; padding: 12px 30px; margin: 0;">
                    Download
                </a>
            </div>
        </div>
    </div>
    
    <script>
        // Preview images
        const contentInput = document.getElementById('contentImage');
        const styleInput = document.getElementById('styleImage');
        const contentPreview = document.getElementById('contentPreview');
        const stylePreview = document.getElementById('stylePreview');
        const contentPreviewWrapper = contentPreview.closest('.preview-wrapper');
        const stylePreviewWrapper = stylePreview.closest('.preview-wrapper');
        const contentUploadIcon = document.getElementById('contentUploadIcon');
        const styleUploadIcon = document.getElementById('styleUploadIcon');
        const alphaSlider = document.getElementById('alpha');
        const alphaValue = document.getElementById('alphaValue');
        const form = document.getElementById('transferForm');
        const loading = document.getElementById('loading');
        const errorDiv = document.getElementById('error');
        const resultSection = document.getElementById('resultSection');
        const submitBtn = document.getElementById('submitBtn');
        
        // Update alpha value display
        alphaSlider.addEventListener('input', (e) => {
            alphaValue.textContent = parseFloat(e.target.value).toFixed(1);
        });
        
        // Remove buttons
        const removeContentBtn = document.getElementById('removeContent');
        const removeStyleBtn = document.getElementById('removeStyle');
        
        // Function to reset content image
        function resetContentImage() {
            contentInput.value = '';
            contentPreviewWrapper.style.display = 'none';
            contentUploadIcon.style.display = 'flex';
            removeContentBtn.style.display = 'none';
        }
        
        // Function to reset style image
        function resetStyleImage() {
            styleInput.value = '';
            stylePreviewWrapper.style.display = 'none';
            styleUploadIcon.style.display = 'flex';
            removeStyleBtn.style.display = 'none';
        }
        
        // Upload icon click handlers
        contentUploadIcon.addEventListener('click', () => {
            contentInput.click();
        });
        
        styleUploadIcon.addEventListener('click', () => {
            styleInput.click();
        });
        
        removeContentBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetContentImage();
        });
        
        removeStyleBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            resetStyleImage();
        });
        
        // File size limit: 10MB
        const MAX_FILE_SIZE = 10 * 1024 * 1024; // 10MB in bytes
        const ALLOWED_TYPES = ['image/jpeg', 'image/jpg', 'image/png'];
        
        // Function to validate file
        function validateFile(file, inputName) {
            // Check file type
            if (!ALLOWED_TYPES.includes(file.type)) {
                const ext = file.name.split('.').pop().toLowerCase();
                throw new Error(`${inputName}: Invalid file format. Only JPG, JPEG, and PNG are allowed. Got: ${ext}`);
            }
            
            // Check file size
            if (file.size > MAX_FILE_SIZE) {
                const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
                const maxMB = (MAX_FILE_SIZE / (1024 * 1024)).toFixed(0);
                throw new Error(`${inputName}: File size too large (${sizeMB}MB). Maximum allowed: ${maxMB}MB`);
            }
            
            return true;
        }
        
        // Content image preview
        contentInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    validateFile(file, 'Content image');
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        contentPreview.src = event.target.result;
                        contentUploadIcon.style.display = 'none';
                        contentPreviewWrapper.style.display = 'block';
                        removeContentBtn.style.display = 'flex';
                    };
                    reader.readAsDataURL(file);
                    errorDiv.classList.remove('active');
                } catch (error) {
                    showError(error.message);
                    contentInput.value = '';
                    resetContentImage();
                }
            }
        });
        
        // Style image preview
        styleInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) {
                try {
                    validateFile(file, 'Style image');
                    const reader = new FileReader();
                    reader.onload = (event) => {
                        stylePreview.src = event.target.result;
                        styleUploadIcon.style.display = 'none';
                        stylePreviewWrapper.style.display = 'block';
                        removeStyleBtn.style.display = 'flex';
                    };
                    reader.readAsDataURL(file);
                    errorDiv.classList.remove('active');
                } catch (error) {
                    showError(error.message);
                    styleInput.value = '';
                    resetStyleImage();
                }
            }
        });
        
        // Prevent clicking on box from triggering file input
        const contentBox = document.getElementById('contentBox');
        const styleBox = document.getElementById('styleBox');
        
        // Prevent box click from opening file picker - only label and icon should trigger
        contentBox.addEventListener('click', (e) => {
            // Allow clicks on label or upload icon to trigger file picker
            if (e.target === contentBox.querySelector('label') || e.target.closest('label') || 
                e.target.closest('.upload-icon')) {
                // Allow these clicks to proceed
                return;
            }
            // Prevent other clicks from propagating
            e.stopPropagation();
        });
        
        styleBox.addEventListener('click', (e) => {
            // Allow clicks on label or upload icon to trigger file picker
            if (e.target === styleBox.querySelector('label') || e.target.closest('label') || 
                e.target.closest('.upload-icon')) {
                // Allow these clicks to proceed
                return;
            }
            // Prevent other clicks from propagating
            e.stopPropagation();
        });
        
        // Drag and drop
        [contentBox, styleBox].forEach((box, index) => {
            const input = index === 0 ? contentInput : styleInput;
            const preview = index === 0 ? contentPreview : stylePreview;
            const previewWrapper = index === 0 ? contentPreviewWrapper : stylePreviewWrapper;
            const uploadIcon = index === 0 ? contentUploadIcon : styleUploadIcon;
            
            box.addEventListener('dragover', (e) => {
                e.preventDefault();
                box.classList.add('dragover');
            });
            
            box.addEventListener('dragleave', () => {
                box.classList.remove('dragover');
            });
            
            box.addEventListener('drop', (e) => {
                e.preventDefault();
                box.classList.remove('dragover');
                const file = e.dataTransfer.files[0];
                const inputName = index === 0 ? 'Content image' : 'Style image';
                
                if (file) {
                    try {
                        // Validate file
                        if (!ALLOWED_TYPES.includes(file.type)) {
                            throw new Error(`${inputName}: Invalid file format. Only JPG, JPEG, and PNG are allowed.`);
                        }
                        if (file.size > MAX_FILE_SIZE) {
                            const sizeMB = (file.size / (1024 * 1024)).toFixed(2);
                            const maxMB = (MAX_FILE_SIZE / (1024 * 1024)).toFixed(0);
                            throw new Error(`${inputName}: File size too large (${sizeMB}MB). Maximum allowed: ${maxMB}MB`);
                        }
                        
                        // Set file to input element
                        try {
                            const dataTransfer = new DataTransfer();
                            dataTransfer.items.add(file);
                            input.files = dataTransfer.files;
                        } catch (err) {
                            // Fallback for older browsers
                            input.files = e.dataTransfer.files;
                        }
                        
                        const reader = new FileReader();
                        reader.onload = (event) => {
                            preview.src = event.target.result;
                            uploadIcon.style.display = 'none';
                            previewWrapper.style.display = 'block';
                            const removeBtn = index === 0 ? removeContentBtn : removeStyleBtn;
                            removeBtn.style.display = 'flex';
                        };
                        reader.readAsDataURL(file);
                        errorDiv.classList.remove('active');
                    } catch (error) {
                        showError(error.message);
                        if (index === 0) {
                            resetContentImage();
                        } else {
                            resetStyleImage();
                        }
                    }
                }
            });
        });
        
        // Form submission
        form.addEventListener('submit', async (e) => {
            e.preventDefault();
            
            const contentFile = contentInput.files[0];
            const styleFile = styleInput.files[0];
            
            if (!contentFile || !styleFile) {
                showError('Please select both content and style images!');
                return;
            }
            
            const formData = new FormData();
            formData.append('content_image', contentFile);
            formData.append('style_image', styleFile);
            formData.append('alpha', alphaSlider.value);
            
            loading.classList.add('active');
            errorDiv.classList.remove('active');
            resultSection.style.display = 'none';
            submitBtn.disabled = true;
            
            try {
                const response = await fetch('/transfer', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const error = await response.json();
                    throw new Error(error.detail || 'An error occurred');
                }
                
                const result = await response.json();
                const resultImage = document.getElementById('resultImage');
                const downloadLink = document.getElementById('downloadLink');
                
                resultImage.src = `/results/${result.filename}`;
                downloadLink.href = `/results/${result.filename}`;
                resultSection.style.display = 'block';
                
            } catch (error) {
                showError(error.message);
            } finally {
                loading.classList.remove('active');
                submitBtn.disabled = false;
            }
        });
        
        function showError(message) {
            errorDiv.textContent = message;
            errorDiv.classList.add('active');
        }
    </script>
</body>
</html>
    """
    return html_content


@app.post("/transfer")
async def transfer_style(
    content_image: UploadFile = File(...),
    style_image: UploadFile = File(...),
    alpha: float = Form(1.0)
):
    try:
       
        content_bytes = await content_image.read()
        style_bytes = await style_image.read()
        
        content_img = process_image(content_bytes, content_image.filename)
        style_img = process_image(style_bytes, style_image.filename)
        
        c_tensor = trans(content_img).unsqueeze(0).to(device)
        s_tensor = trans(style_img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model.generate(c_tensor, s_tensor, alpha)
        
        output_denorm = denorm(output, device)
        
        output_denorm = output_denorm.cpu()
        
        result_filename = f"{uuid.uuid4().hex[:8]}.jpg"
        output_path = RESULTS_DIR / result_filename
        save_image(output_denorm, output_path)
        
        return JSONResponse({
            "success": True,
            "filename": result_filename,
            "message": "Style transfer thành công!"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi xử lý: {str(e)}")


@app.get("/results/{filename}")
async def get_result(filename: str):
    """Lấy ảnh kết quả"""
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File không tồn tại")
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
