import io, base64, json, re, requests, numpy as np, cv2, itertools
from flask import Flask, request, render_template, jsonify
from PIL import Image, ImageDraw, ImageFont
from rembg import remove

# ---------- НАСТРОЙКИ ----------
OLLAMA_API = "http://localhost:11434/api/generate"
LLM_MODEL = "gemma3:4b"

SD_PORTS = [7860, 7861, 7862, 7863, 7864, 7865]
SD_BASE_URL = "http://localhost:{}/sdapi/v1/img2img"
port_cycle = itertools.cycle(SD_PORTS)
app = Flask(__name__)
# Попытаемся импортировать easyocr для автоматического удаления текста
try:
    import easyocr
    easyocr_reader = easyocr.Reader(['ru', 'en'], gpu=True)  # русский и английский
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

# ---------- УТИЛИТЫ ----------
def enhance_prompt(raw_prompt: str) -> str:
    system = "You are a Stable Diffusion prompt expert. Expand a short user request into a detailed, artistic English prompt. Output ONLY the prompt."
    payload = {"model": LLM_MODEL, "prompt": f"{system}\nUSER: {raw_prompt}\nASSISTANT:", "stream": False}
    try:
        resp = requests.post(OLLAMA_API, json=payload, timeout=60)
        return resp.json().get("response", "").strip() or raw_prompt
    except:
        return raw_prompt

def call_sd_img2img(init_b64: str, prompt: str, width=512, height=512) -> str:
    port = next(port_cycle)
    url = SD_BASE_URL.format(port)
    payload = {
        "init_images": [init_b64],
        "prompt": prompt,
        "denoising_strength": 0.6,
        "steps": 25, "cfg_scale": 7.5,
        "width": width, "height": height,
        "sampler_name": "Euler a", "seed": -1
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        return resp.json()["images"][0]
    except Exception as e:
        print(f"SD error: {e}")
        return ""

# ---------- ИНСТРУМЕНТЫ ----------
def tool_style_transfer(image_b64, prompt, w, h):
    enhanced = enhance_prompt(prompt)
    result = call_sd_img2img(image_b64, enhanced, w, h)
    return {"type": "image", "content": f"data:image/png;base64,{result}" if result else image_b64,
            "message": f"✅ Стилизация: {prompt}"}
def call_sd_inpaint(image_b64: str, mask_b64: str, width=512, height=512) -> str:
    """Инпейнтинг через Stable Diffusion – восстанавливает фон аккуратно."""
    port = next(port_cycle)
    url = f"http://localhost:{port}/sdapi/v1/inpaint"
    payload = {
        "init_images": [image_b64],
        "mask": mask_b64,
        "prompt": "seamless background, clean fill, high resolution",
        "negative_prompt": "blurry, distorted, noise",
        "denoising_strength": 0.45,
        "steps": 20,
        "cfg_scale": 7,
        "width": width,
        "height": height,
        "sampler_name": "Euler a",
        "seed": -1,
        "mask_blur": 8,           # слегка размываем края
        "inpainting_fill": 1,     # заполняет соседними пикселями перед генерацией
        "inpaint_full_res": True  # обрабатывает область маски в оригинальном разрешении
    }
    try:
        resp = requests.post(url, json=payload, timeout=300)
        if resp.status_code == 200:
            return resp.json()["images"][0]
        else:
            print(f"Inpaint API error {resp.status_code}: {resp.text}")
    except Exception as e:
        print(f"Inpaint API exception: {e}")
    return ""
def tool_qr_extract(image_b64):
    img_data = base64.b64decode(image_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if points is not None and data:
        pts = points[0].astype(np.int32)
        x, y, w, h = cv2.boundingRect(pts)
        cropped = img[y:y+h, x:x+w]
        _, buffer = cv2.imencode('.png', cropped)
        cropped_b64 = base64.b64encode(buffer).decode()
        return {"type": "image", "content": f"data:image/png;base64,{cropped_b64}",
                "message": f"✅ QR-код вырезан, данные: {data}"}
    return {"type": "text", "content": "❌ QR-код не найден"}

def remove_qr_opencv(image_b64: str) -> str:
    """Удаляет QR-код, закрашивая его средним цветом фона."""
    img_data = base64.b64decode(image_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if points is None:
        return ""
    pts = points[0].astype(np.int32)
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    border = cv2.dilate(mask, kernel, iterations=5) - cv2.erode(mask, kernel, iterations=5)
    border_pixels = img[border == 255]
    if len(border_pixels) > 0:
        mean_color = np.mean(border_pixels, axis=0).astype(int)
    else:
        mean_color = (128, 128, 128)
    img[mask == 255] = mean_color
    img = cv2.GaussianBlur(img, (7,7), 0)
    _, buffer = cv2.imencode('.png', img)
    return base64.b64encode(buffer).decode()

def tool_inpaint_qr(image_b64, w, h):
    """Удаляет QR-код: пытается нейросеть, при неудаче – OpenCV inpaint."""
    # Создаём маску QR (как ранее)
    img_data = base64.b64decode(image_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(img)
    if points is None:
        return {"type": "text", "content": "❌ QR-код не найден"}

    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    pts = points[0].astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (15,15), 0)
    _, mask_buffer = cv2.imencode('.png', mask)
    mask_b64 = base64.b64encode(mask_buffer).decode()

    # 1. Нейросетевой инпейнтинг
    result_b64 = call_sd_inpaint(image_b64, mask_b64, w, h)
    if result_b64:
        return {"type": "image", "content": f"data:image/png;base64,{result_b64}",
                "message": "✅ QR-код удалён (нейросеть)"}
    
    # 2. Fallback на OpenCV inpaint
    result_b64 = opencv_inpaint(image_b64, mask_b64)
    return {"type": "image", "content": f"data:image/png;base64,{result_b64}",
            "message": "✅ QR-код удалён (OpenCV inpaint)"}
def tool_inpaint_mask(image_b64, mask_b64, w, h, object_name="объект"):
    """Удаление произвольного объекта по маске."""
    if mask_b64 is None:
        return {"type": "text", "content": "❌ Маска не предоставлена"}
    # Нейросеть
    result_b64 = call_sd_inpaint(image_b64, mask_b64, w, h)
    if result_b64:
        return {"type": "image", "content": f"data:image/png;base64,{result_b64}",
                "message": f"✅ {object_name} удалён (нейросеть)"}
    # Fallback
    result_b64 = opencv_inpaint(image_b64, mask_b64)
    return {"type": "image", "content": f"data:image/png;base64,{result_b64}",
            "message": f"✅ {object_name} удалён (OpenCV inpaint)"}
def opencv_inpaint(image_b64: str, mask_b64: str) -> str:
    """Fallback: использует алгоритм Telea для заполнения области."""
    img_data = base64.b64decode(image_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    mask_data = base64.b64decode(mask_b64)
    mask_np = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    # Инпейнтинг Telea
    result = cv2.inpaint(img, mask_np, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    _, buffer = cv2.imencode('.png', result)
    return base64.b64encode(buffer).decode()
def tool_remove_text_auto(image_b64, w, h):
    """Удаляет обнаруженный текст через easyocr (маска + заливка)."""
    if not EASYOCR_AVAILABLE:
        return {"type": "text", "content": "❌ easyocr не установлен."}
    img_data = base64.b64decode(image_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    # EasyOCR детекция текстовых блоков
    results = easyocr_reader.readtext(img)
    if not results:
        return {"type": "text", "content": "❌ Текст не найден"}
    mask = np.zeros(img.shape[:2], dtype=np.uint8)
    for (bbox, text, confidence) in results:
        pts = np.array(bbox, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
    # Заливка средним цветом фона (как в remove_qr_opencv)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=3)
    border = cv2.dilate(mask, kernel, iterations=5) - cv2.erode(mask, kernel, iterations=5)
    border_pixels = img[border == 255]
    if len(border_pixels) > 0:
        mean_color = np.mean(border_pixels, axis=0).astype(int)
    else:
        mean_color = (128, 128, 128)
    img[mask == 255] = mean_color
    img = cv2.GaussianBlur(img, (7,7), 0)
    _, buffer = cv2.imencode('.png', img)
    return tool_inpaint_mask(image_b64, mask_b64, w, h, object_name="текст")

def tool_inpaint_generic(image_b64, mask_b64, w, h):
    """Удаляет произвольный объект по пользовательской маске (заливка)."""
    img_data = base64.b64decode(image_b64)
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    mask_data = base64.b64decode(mask_b64)
    mask_np = cv2.imdecode(np.frombuffer(mask_data, np.uint8), cv2.IMREAD_GRAYSCALE)
    mean_color = np.mean(img, axis=(0,1)).astype(int)
    img[mask_np > 127] = mean_color
    img = cv2.GaussianBlur(img, (7,7), 0)
    _, buffer = cv2.imencode('.png', img)
    return {"type": "image", "content": f"data:image/png;base64,{base64.b64encode(buffer).decode()}",
            "message": "✅ Объект удалён (по маске)"}

def tool_remove_bg(image_b64):
    img_data = base64.b64decode(image_b64)
    input_img = Image.open(io.BytesIO(img_data)).convert("RGBA")
    output_img = remove(input_img)
    buf = io.BytesIO()
    output_img.save(buf, format="PNG")
    return {"type": "image", "content": f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}",
            "message": "✅ Фон удалён"}

def tool_business_card(image_b64, name, title, phone, style_prompt):
    white_img = Image.new("RGB", (900, 500), color="white")
    buf = io.BytesIO()
    white_img.save(buf, format="PNG")
    white_b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = enhance_prompt(f"business card background, {style_prompt}, elegant, professional")
    bg_b64 = call_sd_img2img(white_b64, prompt, 900, 500)
    if not bg_b64:
        bg_b64 = white_b64
    bg_img = Image.open(io.BytesIO(base64.b64decode(bg_b64))).convert("RGBA")
    draw = ImageDraw.Draw(bg_img)
    try:
        font_name = ImageFont.truetype("arial.ttf", 40)
        font_small = ImageFont.truetype("arial.ttf", 30)
    except:
        font_name = ImageFont.load_default()
        font_small = ImageFont.load_default()
    draw.text((50, 80), name, font=font_name, fill="white")
    draw.text((50, 150), title, font=font_small, fill="white")
    draw.text((50, 220), phone, font=font_small, fill="white")
    out_buf = io.BytesIO()
    bg_img.save(out_buf, format="PNG")
    return {"type": "image", "content": f"data:image/png;base64,{base64.b64encode(out_buf.getvalue()).decode()}",
            "message": f"✅ Визитка создана: {name}, {title}"}

# ---------- ИНТЕРПРЕТАЦИЯ КОМАНД ----------
def interpret_command_llm(command, has_image):
    tools_desc = """
You are an image processing agent. Select the correct tool strictly.

Available tools:
1. style_transfer – Apply artistic style (prompt). Needs image.
2. qr_extract – CUT OUT / EXTRACT QR code from image. Use for "вырежи qr", "покажи QR отдельно".
3. remove_background – Remove background, make transparent.
4. business_card – Create business card (name, title, phone, style_prompt).
5. inpaint_qr – ERASE / REMOVE QR code automatically from image. Use ONLY for "удали qr код", "убери qr", "сотри qr", "замажь qr".
6. inpaint_generic – Remove an arbitrary object from image using a mask. **Requires a mask image.** If user says "удали текст" or "удали объект" and does NOT provide a mask, return {"action": "inpaint_generic", "params": {"need_mask": true}}.
7. remove_text_auto – Automatically find and delete text on image (using OCR). Use for "удали текст", "убери все надписи". Can work without mask.

CRITICAL RULES:
- "удали qr" → inpaint_qr
- "вырежи qr" → qr_extract
- "удали текст" or "убери надписи" → remove_text_auto
- "удали [что-то другое]" → inpaint_generic (if no mask, set need_mask=true)

Return ONLY JSON: {"action": "tool_name", "params": {...}}.
"""
    payload = {"model": LLM_MODEL, "prompt": f"{tools_desc}\nUser: {command}\nHas image: {has_image}", "stream": False}
    try:
        resp = requests.post(OLLAMA_API, json=payload)
        raw = resp.json().get("response", "").strip()
        json_match = re.search(r'\{[\s\S]*\}', raw)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        print(f"LLM error: {e}")
    return {"action": "unknown", "message": "Не удалось понять команду"}

# ---------- МАРШРУТЫ ----------
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/execute", methods=["POST"])
def execute():
    user_command = request.form.get("command", "").strip()
    if not user_command:
        return jsonify({"type": "text", "content": "❌ Введите команду."})

    file = request.files.get("photo")
    mask_file = request.files.get("mask")
    w, h = 512, 512
    image_b64 = None
    mask_b64 = None

    if file:
        image = Image.open(file).convert("RGB")
        image.thumbnail((1024, 1024))
        w, h = image.size
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        image_b64 = base64.b64encode(buf.getvalue()).decode()

    if mask_file:
        mask_img = Image.open(mask_file).convert("L")
        mask_img = mask_img.resize((w, h))
        mask_buf = io.BytesIO()
        mask_img.save(mask_buf, format="PNG")
        mask_b64 = base64.b64encode(mask_buf.getvalue()).decode()

    interpretation = interpret_command_llm(user_command, image_b64 is not None)
    action = interpretation.get("action")
    params = interpretation.get("params", {})

    if action == "unknown":
        return jsonify({"type": "text", "content": interpretation.get("message", "🤔 Не понимаю.")})

    result = None
    if action == "style_transfer":
        result = tool_style_transfer(image_b64, params.get("prompt", "stylish"), w, h)
    elif action == "qr_extract":
        result = tool_qr_extract(image_b64)
    elif action == "remove_background":
        result = tool_remove_bg(image_b64)
    elif action == "inpaint_qr":
        result = tool_inpaint_qr(image_b64, w, h)
    elif action == "remove_text_auto":
        result = tool_remove_text_auto(image_b64, w, h)
    elif action == "inpaint_generic":
        if not mask_b64:
            return jsonify({
                "type": "text",
                "content": "⚠️ Для удаления произвольного объекта загрузите маску (ч/б изображение, где белое – что удалить).",
                "action": "inpaint_generic"
            })
        result = tool_inpaint_generic(image_b64, mask_b64, w, h)
    elif action == "business_card":
        result = tool_business_card(
            image_b64,
            name=params.get("name", "Имя Фамилия"),
            title=params.get("title", ""),
            phone=params.get("phone", ""),
            style_prompt=params.get("style_prompt", "professional")
        )
    else:
        result = {"type": "text", "content": "🚫 Неизвестное действие."}

    return jsonify(result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)