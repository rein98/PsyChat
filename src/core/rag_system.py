# -*- coding: utf-8 -*-
"""
RAG系统核心模块
整合Agent、向量存储、数据处理，提供完整的检索增强生成功能
"""

import requests
from typing import List, Dict, Any
from src.data.processor import DataProcessor
from src.core.vector_store import VectorStore
from src.agent.psychology_agent import PsychologyAgent
from src.config import *

class RAGSystem:
    def __init__(self):
        # 初始化组件
        self.data_processor = DataProcessor()
        self.vector_store = VectorStore()
        self.psychology_agent = PsychologyAgent()
        
        # 初始化DeepSeek LLM配置
        self.deepseek_api_key = DEEPSEEK_API_KEY
        self.llm_url = f"{DEEPSEEK_BASE_URL}/chat/completions"
        
        # 对话历史
        self.conversation_history = []
        
        # 对话持续监控Agent：追踪连续无RAG的轮数
        self.no_rag_counter = 0  # 连续无RAG的轮数计数器
        self.last_retrieval_docs = []  # 上次检索的文档缓存
        
        print("心理咨询RAG系统初始化完成")
        print(f"📊 对话持续监控Agent已启用：连续{MAX_NO_RAG_ROUNDS}轮无RAG将强制检索")
    
    def build_knowledge_base(self, use_psychology_qa: bool = True, use_header_splitting: bool = True, clear_existing: bool = False) -> bool:
        """构建心理咨询知识库"""
        try:
            print("开始构建心理咨询知识库...")
            
            # 清空现有数据（如果需要）
            if clear_existing:
                self.vector_store.clear_collection()
            
            # 处理文档
            documents = self.data_processor.process_documents(use_psychology_qa, use_header_splitting)
            
            if not documents:
                print("没有找到可处理的文档")
                return False
            
            # 添加到向量存储
            success = self.vector_store.add_documents(documents)
            
            if success:
                # 显示知识库信息
                info = self.vector_store.get_collection_info()
                print(f"心理咨询知识库构建完成: {info}")
                return True
            else:
                print("知识库构建失败")
                return False
                
        except Exception as e:
            print(f"构建知识库时出错: {e}")
            return False
    
    def generate_response(self, query: str, max_tokens: int = 1000) -> Dict[str, Any]:
        """智能生成回答（含对话持续监控Agent）"""
        try:
            print(f"处理查询: {query}")
            
            # 1. 使用AGENT分析用户输入（传递vector_store以支持先检索再引导）
            analysis = self.psychology_agent.analyze_user_input(query, self.conversation_history, self.vector_store)
            print(f"AGENT分析结果: {analysis}")
            
            # 2. 对话持续监控Agent：检查是否需要强制触发RAG
            force_rag = False
            if not analysis['need_rag']:
                self.no_rag_counter += 1
                print(f"📊 无RAG计数器: {self.no_rag_counter}/{MAX_NO_RAG_ROUNDS}")
                
                if self.no_rag_counter >= MAX_NO_RAG_ROUNDS:
                    force_rag = True
                    print(f"🚨 触发强制RAG检索！已连续{self.no_rag_counter}轮未使用检索")
                    # 重新进行完整的分析（包括主题分类和query改写）
                    analysis = self.psychology_agent.analyze_user_input(
                        query, self.conversation_history, self.vector_store, force_retrieval=True
                    )
                    print(f"🔄 强制检索分析结果: {analysis}")
            
            # 3. 根据分析结果决定是否使用RAG
            if analysis['need_rag'] or force_rag:
                # 使用RAG检索
                search_queries = analysis['search_queries']  # 获取所有生成的查询词
                search_query = analysis['search_query']  # 保持向后兼容
                topics = analysis.get('topics', [])
                topic = analysis['topic']  # 保持向后兼容
                
                # 对所有生成的查询词都进行检索，然后合并结果
                print(f"🔍 对 {len(search_queries)} 个查询词进行检索: {search_queries}")
                relevant_docs = self._search_with_multiple_queries(search_queries, topics)
                
                # 如果是强制检索，合并上次缓存的检索结果
                if force_rag and self.last_retrieval_docs:
                    print(f"🔗 合并上次检索结果（{len(self.last_retrieval_docs)}个文档）与新检索结果（{len(relevant_docs)}个文档）")
                    merged_docs = self._merge_and_sort_docs(self.last_retrieval_docs, relevant_docs)
                    relevant_docs = merged_docs
                    print(f"✅ 合并后共 {len(relevant_docs)} 个文档")
                
                # 重置计数器并缓存本次检索结果
                self.no_rag_counter = 0
                self.last_retrieval_docs = relevant_docs.copy() if relevant_docs else []
                
                if relevant_docs:
                    # 构建上下文
                    context = self._build_context(relevant_docs)
                    
                    # 构建提示词
                    prompt = self._build_psychology_prompt(query, context, topic, analysis)
                    
                    # 生成回答
                    answer = self._generate_response(prompt, query)
                    
                    # 准备源文档信息
                    sources = []
                    for doc in relevant_docs:
                        sources.append({
                            'source': doc['metadata']['source'],
                            'topic': doc['metadata'].get('topic', ''),
                            'qa_id': doc['metadata'].get('qa_id', ''),
                            'similarity': doc['similarity']
                        })
                    
                    result = {
                        'success': True,
                        'response': answer,
                        'sources': sources,
                        'context_length': len(context),
                        'used_rag': True,
                        'topic': topic,
                        'search_query': search_query
                    }
                else:
                    # 没有找到相关文档，直接回答
                    prompt = self._build_direct_psychology_prompt(query, topic)
                    answer = self._generate_response(prompt, query)
                    
                    result = {
                        'success': True,
                        'response': answer,
                        'sources': [],
                        'used_rag': False,
                        'topic': topic,
                        'reason': "没有找到相关的心理咨询案例"
                    }
            else:
                # 直接回答，不使用RAG
                prompt = self._build_direct_psychology_prompt(query)
                answer = self._generate_response(prompt, query)
                
                result = {
                    'success': True,
                    'response': answer,
                    'sources': [],
                    'used_rag': False,
                    'reason': "简单对话，无需检索知识库"
                }
            
            # 3. 更新对话历史
            self._update_conversation_history(query, result['response'])
            
            return result
            
        except Exception as e:
            print(f"生成回答时出错: {e}")
            return {
                'success': False,
                'response': f"处理查询时出错: {str(e)}",
                'sources': [],
                'reason': str(e)
            }
    
    def _search_with_multiple_queries(self, queries: List[str], topics: List[str] = None) -> List[Dict[str, Any]]:
        """对多个查询词进行检索，合并所有结果并返回TOP K
        
        Args:
            queries: 查询词列表（包括原始问题和改写的查询词）
            topics: 主题列表
        
        Returns:
            合并去重并按相似度排序的TOP K文档列表
        """
        if not queries:
            return []
        
        all_docs = []
        doc_dict = {}  # 用于去重，key为文档内容的前100个字符
        
        # 对每个查询词进行检索（显式使用config中的阈值）
        for i, query in enumerate(queries, 1):
            print(f"  查询词 {i}/{len(queries)}: {query}")
            docs = self.vector_store.search(query, topics=topics, threshold=SIMILARITY_THRESHOLD)
            print(f"    检索到 {len(docs)} 个文档（阈值={SIMILARITY_THRESHOLD}）")
            
            # 去重并收集文档
            for doc in docs:
                doc_key = doc['content'][:100]  # 使用前100个字符作为唯一标识
                if doc_key not in doc_dict:
                    doc_dict[doc_key] = doc
                    all_docs.append(doc)
                else:
                    # 如果文档已存在，保留相似度更高的那个
                    if doc['similarity'] > doc_dict[doc_key]['similarity']:
                        # 更新文档
                        doc_dict[doc_key] = doc
                        # 在列表中找到并替换
                        for idx, existing_doc in enumerate(all_docs):
                            if existing_doc['content'][:100] == doc_key:
                                all_docs[idx] = doc
                                break
        
        # 按相似度排序
        all_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # 返回TOP K结果
        top_k_docs = all_docs[:TOP_K_RESULTS]
        print(f"✅ 多查询词检索合并后共 {len(all_docs)} 个文档（去重后），返回TOP {len(top_k_docs)} 个")
        
        return top_k_docs
    
    def _merge_and_sort_docs(self, cached_docs: List[Dict[str, Any]], new_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """合并并排序上次缓存的文档和新检索的文档
        
        Args:
            cached_docs: 上次检索缓存的文档
            new_docs: 本次新检索的文档
        
        Returns:
            合并去重并按相似度排序的文档列表
        """
        # 使用字典去重（基于文档内容的哈希）
        doc_dict = {}
        
        # 先添加新文档（优先级更高）
        for doc in new_docs:
            doc_key = doc['content'][:100]  # 使用前100个字符作为唯一标识
            if doc_key not in doc_dict:
                doc_dict[doc_key] = doc
        
        # 再添加缓存文档
        for doc in cached_docs:
            doc_key = doc['content'][:100]
            if doc_key not in doc_dict:
                doc_dict[doc_key] = doc
        
        # 转换为列表并按相似度排序
        merged_docs = list(doc_dict.values())
        merged_docs.sort(key=lambda x: x.get('similarity', 0), reverse=True)
        
        # 限制在TOP_K范围内
        return merged_docs[:TOP_K_RESULTS]
    
    def _build_context(self, relevant_docs: List[Dict[str, Any]]) -> str:
        """构建心理咨询上下文"""
        context_parts = []
        
        for i, doc in enumerate(relevant_docs, 1):
            metadata = doc['metadata']
            topic = metadata.get('topic', '未知')
            qa_id = metadata.get('qa_id', '未知')
            
            context_parts.append(f"参考案例 {i} (主题: {topic}, ID: {qa_id}, 相似度: {doc['similarity']:.3f}):")
            context_parts.append(doc['content'])
            context_parts.append("---")
        
        return "\n".join(context_parts)
    
    def _generate_response(self, system_prompt: str, user_query: str, include_history: bool = True) -> str:
        """使用DeepSeek LLM生成回答"""
        try:
            headers = {
                "Authorization": f"Bearer {self.deepseek_api_key}",
                "Content-Type": "application/json"
            }
            
            # 构建消息列表，包含对话历史
            messages = []
            
            # 添加系统提示词
            messages.append({
                "role": "system",
                "content": system_prompt
            })
            
            # 如果需要包含历史记录且历史记录存在
            if include_history and self.conversation_history:
                # 只包含最近的6轮对话（12条消息），避免token过多
                recent_history = self.conversation_history[-12:]
                messages.extend(recent_history)
            
            # 添加当前用户查询
            messages.append({
                "role": "user", 
                "content": user_query
            })
            
            data = {
                "model": DEEPSEEK_MODEL,
                "messages": messages,
                "max_tokens": 1000,  # 减少token数，鼓励简短回答
                "temperature": 0.6,   # 降低随机性，更稳定
                "top_p": 0.9
            }
            
            response = requests.post(self.llm_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                print(f"LLM API响应格式错误: {result}")
                return "抱歉，我无法生成有效的回答。"
                
        except Exception as e:
            print(f"生成回答时出错: {e}")
            return f"处理查询时出错: {str(e)}"
    
    
    def _build_psychology_prompt(self, query: str, context: str, topic: str = None, analysis: Dict = None) -> str:
        """构建心理咨询提示词"""
        prompt = f"""你是一位精通理情行为疗法（Rational Emotive Behavior Therapy,REBT）的心理咨询师，能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。
        
请仔细阅读以下参考的心理咨询对话案例，学习其中助手的回应风格和技巧：
{context}

现在，请运用REBT理论和参考案例中的风格，针对用户的具体问题进行回应。

回答要求:
- 保持简短（1-5句话）
- 以鼓励和引导为主，体现共情和理解
- 请注意参考案例中助手的特点，但要针对用户的具体情况，不要混淆案例中的情况
- 可以用问题引导用户进一步思考和自我反省
- 语气温暖自然，具有专业性

请直接回答用户的问题，不要重复或引用案例内容。
"""
        return prompt
    
    def _build_direct_psychology_prompt(self, query: str, topic: str = None) -> str:
        """构建直接心理咨询提示词（不使用RAG）"""
        prompt = f"""你是一位精通理情行为疗法（Rational Emotive Behavior Therapy，简称REBT）的心理咨询师，能够合理地采用理情行为疗法给来访者提供专业地指导和支持，缓解来访者的负面情绪和行为反应，帮助他们实现个人成长和心理健康。

理情行为治疗主要包括以下几个阶段：
（1）**检查非理性信念和自我挫败式思维**：理情行为疗法把认知干预视为治疗的"生命"，因此，几乎从治疗一开始，在问题探索阶段，咨询师就以积极的、说服教导式的态度帮助来访者探查隐藏在情绪困扰后面的原因，包括来访者理解事件的思维逻辑，产生情绪的前因后果，借此来明确问题的所在。咨询师坚定地激励来访者去反省自己在遭遇刺激事件后，在感到焦虑、抑郁或愤怒前对自己"说"了些什么。
（2）**与非理性信念辩论**：咨询师运用多种技术（主要是认知技术）帮助来访者向非理性信念和思维质疑发难，证明它们的不现实、不合理之处，认识它们的危害进而产生放弃这些不合理信念的愿望和行为。
（3）**得出合理信念，学会理性思维**：在识别并驳倒非理性信念的基础上，咨询师进一步诱导、帮助来访者找出对于刺激情境和事件的适宜的、理性的反应，找出理性的信念和实事求是的、指向问题解决的思维陈述，以此来替代非理性信念和自我挫败式思维。为了巩固理性信念，咨询师要向来访者反复教导，证明为什么理性信念是合情合理的，它与非理性信念有什么不同，为什么非理性信念导致情绪失调，而理性信念导致较积极、健康的结果。
（4）**迁移应用治疗收获**：积极鼓励来访者把在治疗中所学到的客观现实的态度，科学合理的思维方式内化成个人的生活态度，并在以后的生活中坚持不懈地按理情行为疗法的教导来解决新的问题。

回答要求:
- 保持简短（1-5句话）
- 运用REBT理论，帮助来访者识别和质疑非理性信念，但不要明确提到REBT
- 以温暖、理解、非评判的语气回应
- 以鼓励和引导为主，而非直接给建议
- 用问题引导用户思考和自我反省
- 如果是简单问候，以友善自然的方式回应
- 体现共情和理解，具有专业性

请用中文回答，语气要亲切自然。
"""
        return prompt
    
    def _update_conversation_history(self, user_message: str, assistant_response: str):
        """更新对话历史"""
        self.conversation_history.append({
            'role': 'user',
            'content': user_message
        })
        self.conversation_history.append({
            'role': 'assistant', 
            'content': assistant_response
        })
        
        # 保持对话历史在合理长度内（最近10轮对话）
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]
    
    def clear_conversation_history(self):
        """清空对话历史"""
        self.conversation_history = []
    
    def get_knowledge_base_info(self) -> Dict[str, Any]:
        """获取知识库信息"""
        return self.vector_store.get_collection_info()
    
    def test_query(self, query: str) -> None:
        """测试查询功能"""
        print(f"\n{'='*50}")
        print(f"测试查询: {query}")
        print(f"{'='*50}")
        
        result = self.generate_response(query)
        
        if result['success']:
            print(f"\n回答:")
            print(result['response'])
            
            print(f"\n参考来源:")
            for source in result['sources']:
                print(f"- {source['source']} (相似度: {source['similarity']:.3f})")
                if source.get('header'):
                    print(f"  标题: {source['header']}")
        else:
            print(f"查询失败: {result.get('reason', '未知错误')}")


def main():
    """主函数"""
    rag = RAGSystem()
    
    # 检查知识库状态
    info = rag.get_knowledge_base_info()
    print(f"当前知识库状态: {info}")
    
    # 如果知识库为空，构建知识库
    if info.get('document_count', 0) == 0:
        print("知识库为空，开始构建...")
        success = rag.build_knowledge_base()
        if not success:
            print("知识库构建失败，退出程序")
            return
    
    # 交互式查询
    print("\n欢迎使用MCP知识库RAG系统！")
    print("输入 'quit' 或 'exit' 退出程序")
    print("输入 'info' 查看知识库信息")
    print("输入 'rebuild' 重新构建知识库")
    
    while True:
        try:
            query = input("\n请输入您的问题: ").strip()
            
            if not query:
                continue
                
            if query.lower() in ['quit', 'exit', 'q']:
                print("再见！")
                break
                
            if query.lower() == 'info':
                info = rag.get_knowledge_base_info()
                print(f"知识库信息: {info}")
                continue
                
            if query.lower() == 'rebuild':
                print("重新构建知识库...")
                success = rag.build_knowledge_base(clear_existing=True)
                if success:
                    print("知识库重建完成")
                else:
                    print("知识库重建失败")
                continue
            
            # 处理查询
            rag.test_query(query)
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"处理输入时出错: {e}")


if __name__ == "__main__":
    main()

