# -*- coding: utf-8 -*-
"""
项目配置文件
包含API配置、数据库配置、检索配置等
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 加载 .env 文件
load_dotenv()

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# API 配置
# =============================================================================

# DeepSeek API配置
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")  # DeepSeek API密钥
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")  # DeepSeek API地址
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")  # DeepSeek聊天模型

# 阿里云百炼Embedding配置
ALIBABA_API_KEY = os.getenv("ALIBABA_API_KEY", "")  # 阿里云百炼API密钥
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-v4")  # 阿里云百炼Embedding模型

# =============================================================================
# 存储配置
# =============================================================================

# ChromaDB配置
CHROMA_DB_PATH = str(PROJECT_ROOT / "storage" / "chroma_db")
COLLECTION_NAME = "psychology_knowledge"

# =============================================================================
# 资源路径配置
# =============================================================================

# 知识库文档目录
KNOWLEDGE_DIR = str(PROJECT_ROOT / "resources" / "knowledge")

# 提示词目录
PROMPTS_DIR = str(PROJECT_ROOT / "resources" / "prompts")

# =============================================================================
# 文本处理配置
# =============================================================================

# 文本分块配置
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50

# =============================================================================
# 检索配置
# =============================================================================

TOP_K_RESULTS = 6
SIMILARITY_THRESHOLD = 0.15

# =============================================================================
# 对话配置
# =============================================================================

# 对话持续监控配置
MAX_NO_RAG_ROUNDS = 3

