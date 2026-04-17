import os

# 加载.env文件（如果存在）
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

__version__ = f"0.1.{os.getenv('BUILD_NUMBER', '0')}"
