#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心理咨询伴侣
基于RAG技术的智能心理健康助手，提供专业的心理咨询支持

作者: wink-wink-wink555
"""

import sys
from pathlib import Path

# 确保项目根目录在Python路径中
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.core.rag_system import RAGSystem
import argparse


def main():
    parser = argparse.ArgumentParser(description='心理咨询伴侣')
    parser.add_argument('--rebuild', action='store_true', help='重新构建心理咨询知识库')
    parser.add_argument('--info', action='store_true', help='显示知识库信息')
    parser.add_argument('--cli', action='store_true', help='使用命令行交互模式')
    
    args = parser.parse_args()
    
    # 显示系统信息
    print("="*60)
    print("💝 心理咨询伴侣")
    print("="*60)
    print("基于RAG技术的智能心理健康助手，提供专业的心理咨询支持")
    print("作者: wink-wink-wink555")
    print("="*60)
    
    # 处理命令行参数
    if args.info:
        rag = RAGSystem()
        info = rag.get_knowledge_base_info()
        print(f"知识库信息: {info}")
        return
    
    if args.rebuild:
        rag = RAGSystem()
        print("重新构建心理咨询知识库...")
        success = rag.build_knowledge_base(clear_existing=True)
        if success:
            print("心理咨询知识库重建完成")
        else:
            print("心理咨询知识库重建失败")
        return
    
    if args.cli:
        # 命令行交互模式
        from src.core.rag_system import main as cli_main
        cli_main()
        return
    
    # 默认启动Web界面
    print("🚀 启动Web界面...")
    try:
        from src.web.interface import main as web_main
        web_main()
    except ImportError as e:
        print(f"❌ 启动Web界面失败: {e}")
        print("请确保已安装FastAPI和uvicorn: pip install fastapi uvicorn[standard]")


if __name__ == "__main__":
    main()
