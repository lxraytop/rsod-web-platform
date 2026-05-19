import os
import uuid
from fastapi import UploadFile
from app.config import settings


def ensure_directories():
    """确保必要的目录存在"""
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.RESULT_DIR, exist_ok=True)


async def save_upload_file(upload_file: UploadFile, upload_dir: str) -> str:
    """保存上传文件"""
    filename = f"temp_{uuid.uuid4().hex}{os.path.splitext(upload_file.filename)[1]}"
    file_path = os.path.join(upload_dir, filename)
    with open(file_path, "wb") as f:
        content = await upload_file.read()
        f.write(content)
    return filename


def get_file_url(filename: str, base_dir: str) -> str:
    """获取文件访问URL"""
    return f"/{base_dir}/{filename}"
