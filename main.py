import os
import io
import re
import requests
import PIL.Image
import uvicorn
from collections import defaultdict
from fastapi import FastAPI, Request, Header, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Google GenAI SDK
from google import genai
from google.genai import types

# Line Bot
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage, ImageSendMessage, ImageMessage,
)

from langchain.agents import create_agent

# ==========================
#  環境設定與工具函式
# ==========================
google_api = os.environ.get("GOOGLE_API_KEY")
genai_client = genai.Client(api_key=google_api)

pixazo_api = os.environ.get("PIXAZO_API_KEY")

line_bot_api = LineBotApi(os.environ.get("CHANNEL_ACCESS_TOKEN"))
line_handler = WebhookHandler(os.environ.get("CHANNEL_SECRET"))

user_message_history = defaultdict(list)
app = FastAPI()

if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_image_url_from_line(message_id):
    try:
        message_content = line_bot_api.get_message_content(message_id)
        file_path = f"static/{message_id}.png"
        with open(file_path, "wb") as f:
            for chunk in message_content.iter_content():
                f.write(chunk)
        return file_path
    except Exception as e:
        print(f"❌ 圖片取得失敗：{e}")
        return None

def store_user_message(user_id, message_type, message_content):
    user_message_history[user_id].append({"type": message_type, "content": message_content})

def get_previous_message(user_id):
    if user_id in user_message_history and len(user_message_history[user_id]) > 0:
        return user_message_history[user_id][-1]
    return {"type": "text", "content": "No message!"}

# ==========================
#  LangChain 工具定義
# ==========================

def generate_and_upload_image(prompt: str) -> str:
    """根據文字提示生成圖片。"""
    
    # Free api - pixazo.ai/getImage/v1/getSDXLImage
    try:
        # --- 1. 設定 API 請求參數 ---
        url = "https://gateway.pixazo.ai/getImage/v1/getSDXLImage"
        headers = {
            "Content-Type": "application/json",
            "Cache-Control": "no-cache",
            "Ocp-Apim-Subscription-Key": pixazo_api  # 填入你的 API 金鑰
        }
        # 定義生成圖片的詳細設定
        data = {
            "prompt": prompt,
            "negative_prompt": "Low-quality, blurry image, with any other birds or animals. Avoid abstract or cartoonish styles, dark or gloomy atmosphere, unnecessary objects or distractions in the background, harsh lighting, and unnatural colors.",
            "height": 1024,
            "width": 1024,
            "num_steps": 20,      # 步數越高通常細節越多，但也越耗時
            "guidance_scale": 5,  # 數值越高，圖片越貼近提示詞
            "seed": 40            # 固定種子碼可確保相同提示詞生成相同圖片
        }
        # --- 2. 發送請求至 Pixazo API ---
        response = requests.post(url, json=data, headers=headers)
        print(response.json())  # 在控制台查看原始回應數據內容
        # 解析 JSON 回應
        resp_data = response.json()
        image_url = resp_data.get("imageUrl") # 從回應中取得圖片的外部連結
        if image_url:
            # --- 3. 下載生成的圖片 ---
            # 因為 API 只給 URL，我們需要再發送一次 GET 請求把圖片存下來
            image_download_response = requests.get(image_url)
            if image_download_response.status_code == 200:
                # 取得圖片的二進位數據 (Binary data)
                image_binary = image_download_response.content
                # --- 4. 處理並儲存圖片 ---
                # 使用 PIL 打開二進位數據並轉為圖片格式
                image = PIL.Image.open(io.BytesIO(image_binary))
                # 產生一個隨機的檔名（避免重複），存放在 static 資料夾下
                file_name = f"static/{os.urandom(8).hex()}.png"
                image.save(file_name, format="PNG")
                # --- 5. 回傳圖片連結 ---
                # 取得目前的伺服器基礎位址 (例如 Hugging Face Space 或本地端)
                base_url = os.getenv("HF_SPACE", "http://localhost:7860").rstrip("/")
                return f"{base_url}/{file_name}"
            else:
                return "從 URL 下載圖片失敗。"
        else:
            # 如果 JSON 裡沒拿到 imageUrl，回傳 API 給出的錯誤訊息
            return f"圖片生成失敗: {resp_data.get('message', 'API 未回傳 URL')}"
    except Exception as e:
        # 捕捉任何執行過程中的異常（如網路中斷、記憶體錯誤等）
        return f"程式執行出錯: {e}"

def analyze_image_with_text(image_path: str, user_text: str) -> str:
    """根據圖片路徑和文字提問進行分析。"""
    try:
        if not os.path.exists(image_path):
            return "錯誤：找不到該圖片檔案。"
        img_user = PIL.Image.open(image_path)
        response = genai_client.models.generate_content(
            model="gemini-3-flash-preview", # "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-2.5-flash-lite", "gemini-2.5-flash"
            contents=[img_user, user_text]
        )
        return response.text if response.text else "Gemini 沒答案！"
    except Exception as e:
        return f"分析出錯: {e}"

# ==========================
#  LangChain 代理人設定
# ==========================
tools = [generate_and_upload_image, analyze_image_with_text]

system_prompt = """
你是一個強大的影像生成及影像分析代理人。如果生成了圖片，請直接給出 imageUrl。
【重要指示】：如果使用者只是單純打招呼、閒聊或沒有提供明確的圖片需求，請直接用文字友善回覆，絕對不要呼叫任何工具。
"""

agent_executor = create_agent(
    model="google_genai:gemini-3.1-flash-lite",
    tools=tools,
    system_prompt=system_prompt,
)

# ==========================
#  FastAPI 路由
# ==========================
@app.get("/")
def root():
    return {"status": "running"}

@app.post("/webhook")
async def webhook(request: Request, background_tasks: BackgroundTasks, x_line_signature=Header(None)):
    body = await request.body()
    try:
        background_tasks.add_task(line_handler.handle, body.decode("utf-8"), x_line_signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400)
    return "ok"

@line_handler.add(MessageEvent, message=(ImageMessage, TextMessage))
def handle_message(event):
    user_id = event.source.user_id
    
    # 1. 處理圖片訊息
    if isinstance(event.message, ImageMessage):
        image_path = get_image_url_from_line(event.message.id)
        if image_path:
            store_user_message(user_id, "image", image_path)
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="收到圖片了！請問你想對這張圖做什麼分析？"))
    # 2. 處理文字訊息 (修正縮進，確保它與上面的 if 對齊)
    elif isinstance(event.message, TextMessage):
        user_text = event.message.text
        previous_message = get_previous_message(user_id)
        if previous_message["type"] == "image":
            image_path = previous_message["content"]
            user_text = f"請分析這張圖片 {image_path}，問題是：{user_text}"
            user_message_history[user_id].pop()

        agent_input = {"messages": [{"role": "user", "content": user_text}]} 
        try:
            response = agent_executor.invoke(agent_input)
            messages = response.get("messages", [])
            
            out = ""
            image_url = None

            # 1. 由後往前找，優先找出包含 imageUrl 的訊息
            for msg in reversed(messages):
                content = msg.content
                
                # 如果內容是字典 (工具回傳的格式)
                if isinstance(content, dict) and "imageUrl" in content:
                    image_url = content["imageUrl"]
                    break
                
                # 如果內容是字串，嘗試從中提取網址
                elif isinstance(content, str) and content.strip():
                    # 這裡用我們先前的 Regex 找找看
                    found_urls = re.findall(r'https?://[^\s<>"\)]+|www\.[^\s<>"\)]+', content)
                    img = next((u for u in found_urls if any(ext in u.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif'])), None)
                    if img:
                        image_url = img.split(')')[0]
                        out = content # 保留文字內容
                        break
                    elif not out: # 如果還沒找到圖片，先暫存文字內容
                        out = content

                # 如果內容是列表 (Gemini 2.5 有時會回傳 block list)
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            # 找圖片網址
                            text = block.get("text", "")
                            found_urls = re.findall(r'https?://[^\s<>"\)]+|www\.[^\s<>"\)]+', text)
                            img = next((u for u in found_urls if any(ext in u.lower() for ext in ['.png', '.jpg', '.jpeg', '.gif'])), None)
                            if img:
                                image_url = img.split(')')[0]
                            if text:
                                out += text

            # 2. 最終回覆邏輯
            if image_url:
                line_bot_api.reply_message(
                    event.reply_token,
                    [
                        TextSendMessage(text="這是我為你生成的圖片："),
                        ImageSendMessage(original_content_url=image_url, preview_image_url=image_url)
                    ]
                )
            elif out.strip():
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text=out.strip()))
            else:
                # 萬一真的什麼都沒有，回報一個預設訊息，避免 LINE 噴 400 錯誤
                line_bot_api.reply_message(event.reply_token, TextSendMessage(text="圖片已生成，但我暫時無法取得連結，請稍後再試。"))

        except Exception as e:
            print(f"Agent Error: {e}")
            line_bot_api.reply_message(event.reply_token, TextSendMessage(text="抱歉，我現在無法處理這個請求。"))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
