#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
心理咨询伴侣智能AGENT
能够自主决定是否进行RAG检索、主题分类和查询生成
"""

import requests
from typing import List, Dict, Any, Optional, Tuple
from src.config import *

class PsychologyAgent:
    def __init__(self):
        self.api_key = DEEPSEEK_API_KEY
        self.llm_url = f"{DEEPSEEK_BASE_URL}/chat/completions"
        
        print("心理咨询AGENT初始化完成")
    
    def should_use_rag(self, user_message: str, conversation_history: List[Dict] = None) -> bool:
        """判断是否需要使用RAG检索"""
        
        # 构建对话历史上下文
        context = ""
        if conversation_history:
            for msg in conversation_history:  # 使用完整的对话历史
                if msg.get('role') == 'user':
                    context += f"用户之前说: {msg.get('content', '')}\n"
                elif msg.get('role') == 'assistant':
                    context += f"助手之前回复: {msg.get('content', '')}\n"
        
        # 构建判断提示词
        prompt = f"""你是一个心理咨询助手的智能决策系统。请判断用户的问题是否需要从心理学知识库中检索相关信息来回答。

对话历史上下文:
{context if context.strip() else '无'}

当前用户问题: {user_message}

判断标准:
1. 如果是简单的问候、感谢或闲聊，返回 NO
2. 如果涉及具体的心理问题、情感困扰、人际关系问题等需要专业建议的，返回 YES  
3. 如果询问心理学知识、理论、方法等，返回 YES
4. 如果是需要情感支持和专业指导的问题，返回 YES

请结合对话历史理解用户问题的完整含义和真实意图。只回答 YES 或 NO，不要解释。"""

        try:
            response = self._call_llm(prompt, max_tokens=10)
            decision = response.strip().upper()
            return decision == 'YES'
        except:
            # 默认使用RAG，确保不遗漏重要问题
            return True
    
    def classify_topic(self, user_message: str, conversation_history: List[Dict] = None) -> List[str]:
        """使用大模型对用户问题进行智能主题分类，支持多标签分类"""
        
        # 构建对话历史上下文（使用完整的对话历史）
        context = ""
        if conversation_history:
            for msg in conversation_history:  # 使用完整的对话历史
                if msg.get('role') == 'user':
                    context += f"用户之前说: {msg.get('content', '')}\n"
                elif msg.get('role') == 'assistant':
                    context += f"助手之前回复: {msg.get('content', '')}\n"
        
        prompt = f"""你是一位专业的心理咨询主题分类专家。请分析用户的问题，识别其属于哪些心理学主题（可以是1-3个主题）。

主题详解（只能从以下12个主题中选择）：
1. 情绪 - 焦虑、抑郁、愤怒、恐惧等各种情绪问题，情绪管理、情感压抑、情绪波动
2. 人际 - 朋友关系、同学室友相处、社交恐惧、人际冲突、被排斥孤立、沟通困难
3. 婚恋 - 恋爱关系、伴侣沟通、情感矛盾、分手挽回、感情选择、亲密关系困扰
4. 家庭 - 父母关系、家庭暴力、原生家庭影响、家庭冲突、亲子关系、家庭责任
5. 性心理 - 性取向、性欲、性行为、性困惑、婚外情、性别认同
6. 成长 - 青少年发展、学业压力、习惯养成、自我突破、人生规划、考试压力
7. 治疗 - 心理疾病、躯体化障碍、心理治疗方法、专业心理干预、咨询技术
8. 社会 - 社会现象、心理健康科普、社会议题、公共心理问题
9. 职场 - 职业选择、工作压力、失业困境、职业发展、工作倦怠、职场人际
10. 自我 - 自我认同、自我价值、人生迷茫、个性表达、自信心、兴趣探索
11. 行为 - 强迫行为、习惯问题、行为模式、反复确认、行为改变
12. 心理学知识 - 心理学理论、人格特质、心理学概念、心理学应用、知识咨询

对话历史上下文:
{context}

当前用户问题: {user_message}

分类要求：
1. 结合对话历史理解用户问题的完整含义和真实意图
2. 从上述12个主题中选择最相关的1-3个主题
3. 支持多标签分类，复杂问题可涉及多个主题
4. 优先选择最核心、最直接相关的主题

示例：
- "我和女朋友总是吵架" → 婚恋, 人际
- "感觉自己做什么都不行" → 自我, 情绪
- "总是反复确认门锁" → 行为, 情绪
- 对话提到工作问题，用户说"我很焦虑" → 职场, 情绪

请只回答主题名称，多个主题用逗号分隔。如果实在无法分类，请回答"通用"。"""

        try:
            response = self._call_llm(prompt, max_tokens=50)
            # 解析返回的主题
            topics = [t.strip() for t in response.split(',') if t.strip()]
            
            # 处理"通用"的情况：如果只返回"通用"，返回空列表（表示不限定主题）
            if topics == ["通用"]:
                print("📊 主题分类结果: 通用（将在所有主题中检索）")
                return []
            
            # 过滤掉"通用"，保留其他主题
            topics = [t for t in topics if t != "通用"]
            
            print(f"📊 主题分类结果: {topics if topics else '通用（无特定主题）'}")
            # 返回分类结果（不再验证是否在预定义列表中，允许LLM自由分类）
            return topics
        except Exception as e:
            print(f"❌ LLM主题分类失败: {e}")
            return []
    
    def generate_search_queries_with_pre_retrieval(self, user_message: str, topics: List[str] = None, conversation_history: List[Dict] = None, vector_store=None) -> List[str]:
        """先检索再引导的Query改写机制"""
        
        if not vector_store:
            # 如果没有向量存储，降级到原始方法
            return self._fallback_query_generation(user_message, topics, conversation_history)
        
        # 第一步：轻检索 - 用原始Query获取知识库锚点
        print(f"🔍 第一步：轻检索原始Query: {user_message}")
        anchor_docs = self._light_retrieval(user_message, vector_store, top_k=3)
        
        if not anchor_docs:
            print("⚠️ 轻检索无结果，使用降级方案")
            return self._fallback_query_generation(user_message, topics, conversation_history)
        
        # 第二步：提取锚点信息
        anchor_info = self._extract_anchor_info(anchor_docs)
        print(f"📌 提取的锚点信息: {anchor_info}")
        
        # 第三步：智能判断是否需要进一步改写（返回是否改写 + 原因序号）
        need_rewrite, rewrite_reasons = self._should_rewrite_query(user_message, anchor_info, conversation_history)
        print(f"🤔 是否需要改写: {need_rewrite}")
        if rewrite_reasons:
            print(f"📋 改写原因: {rewrite_reasons}")
        
        if not need_rewrite:
            # 不需要改写，直接返回原始查询
            return [user_message]
        
        # 第四步：基于锚点引导的Query改写（传递改写原因）
        rewritten_queries = self._guided_query_rewrite(user_message, anchor_info, rewrite_reasons, topics, conversation_history)
        
        # 组合最终查询：原始问题 + 改写查询
        final_queries = [user_message] + rewritten_queries
        print(f"✅ 最终检索查询: {final_queries}")
        
        return final_queries
    
    def _light_retrieval(self, query: str, vector_store, top_k: int = 3) -> List[Dict]:
        """轻检索：快速获取知识库锚点"""
        try:
            # 使用较低的相似度阈值进行快速检索
            results = vector_store.search(query, top_k=top_k, threshold=0.05, topics=None)
            return results[:top_k] if results else []
        except Exception as e:
            print(f"轻检索失败: {e}")
            return []
    
    def _extract_anchor_info(self, anchor_docs: List[Dict]) -> Dict[str, Any]:
        """从锚点文档中提取关键信息"""
        anchor_info = {
            'keywords': [],
            'concepts': [],
            'expressions': [],
            'topics': []
        }
        
        for doc in anchor_docs:
            content = doc['content']
            metadata = doc['metadata']
            
            # 提取主题
            if 'topic' in metadata:
                topic = metadata['topic']
                if topic not in anchor_info['topics']:
                    anchor_info['topics'].append(topic)
            
            # 提取关键表达（简单实现：取前50个字符作为表达样例）
            if len(content) > 20:
                expression = content[:50].replace('\n', ' ').strip()
                if expression not in anchor_info['expressions']:
                    anchor_info['expressions'].append(expression)
        
        return anchor_info
    
    def _should_rewrite_query(self, original_query: str, anchor_info: Dict, conversation_history: List[Dict] = None) -> Tuple[bool, List[str]]:
        """判断是否需要进一步改写Query，并返回具体原因
        
        Args:
            original_query: 用户原始查询
            anchor_info: 锚点信息
            conversation_history: 对话历史
        
        Returns:
            Tuple[bool, List[str]]: (是否需要改写, 改写原因列表)
        """
        
        # 构建对话历史上下文（使用完整的对话历史）
        context = ""
        if conversation_history:
            for msg in conversation_history:  # 使用完整的对话历史
                if msg.get('role') == 'user':
                    context += f"用户之前说: {msg.get('content', '')}\n"
                elif msg.get('role') == 'assistant':
                    context += f"助手之前回复: {msg.get('content', '')}\n"
        
        # 构建判断提示词
        anchor_topics = ', '.join(anchor_info['topics'][:3]) if anchor_info['topics'] else '无'
        anchor_expressions = '\n'.join([f"- {expr}" for expr in anchor_info['expressions'][:3]])
        
        prompt = f"""你是一个智能检索优化专家。请判断是否需要对用户查询进行进一步改写，并给出具体原因。

对话历史上下文:
{context if context.strip() else '无'}

原始用户查询: {original_query}

知识库中相关内容的特征:
主题: {anchor_topics}
表达样例:
{anchor_expressions}

判断标准（可多选）:
1. 用户查询过于口语化，需要转换为更专业的心理学表达
2. 用户查询过于宽泛，需要细化为具体的心理学概念
3. 用户查询与知识库表达差异较大，需要对齐知识库的表达方式
4. 用户查询缺少关键信息，需要补充相关的心理学术语
5. 用户查询包含指代词（如"这样"、"那个"、"不知道"等），或结合对话历史发现查询真实含义与检索结果语境不匹配

请结合对话历史理解用户问题的完整含义。如果用户查询已经清晰明确且与知识库表达方式接近，不需要改写，返回: NO

如果需要改写，返回格式样例如下：
例如返回: YES,1,3 表示需要改写，原因是标准1和标准3
YES,2,3,4 表示需要改写，原因是标准2、标准3和标准4
NO 表示不需要改写，因此也无改写原因

请严格按照上述格式回答，不要有其他解释。"""

        try:
            response = self._call_llm(prompt, max_tokens=20)
            parts = [p.strip() for p in response.strip().upper().split(',')]
            
            if not parts or parts[0] == 'NO':
                return False, []
            
            if parts[0] == 'YES':
                # 解析原因序号
                reasons = []
                reason_map = {
                    '1': '过于口语化，需要专业表达',
                    '2': '过于宽泛，需要具体概念',
                    '3': '表达差异大，需要对齐知识库',
                    '4': '缺少关键信息，需要补充术语',
                    '5': '指代不明或语境不匹配，需明确化'
                }
                
                for part in parts[1:]:
                    if part in reason_map:
                        reasons.append(f"{part}. {reason_map[part]}")
                
                return True, reasons if reasons else ['需要改写查询以提高检索准确度']
            
            # 默认不改写
            return False, []
            
        except Exception as e:
            print(f"⚠️ 判断是否改写时出错: {e}")
            # 默认进行改写，确保检索效果
            return True, ['默认改写以提高检索效果']
    
    def _guided_query_rewrite(self, user_message: str, anchor_info: Dict, rewrite_reasons: List[str], topics: List[str] = None, conversation_history: List[Dict] = None) -> List[str]:
        """基于锚点信息和改写原因的引导式Query改写
        
        Args:
            user_message: 用户原始问题
            anchor_info: 知识库锚点信息
            rewrite_reasons: 改写原因列表（来自智能检索优化专家）
            topics: 主题列表
            conversation_history: 对话历史
        
        Returns:
            改写后的查询词列表
        """
        
        # 构建上下文（使用完整的对话历史）
        context = ""
        if conversation_history:
            for msg in conversation_history:  # 使用完整的对话历史
                if msg.get('role') == 'user':
                    context += f"用户之前说: {msg.get('content', '')}\n"
                elif msg.get('role') == 'assistant':
                    context += f"助手之前回复: {msg.get('content', '')}\n"
        
        # 构建锚点信息
        anchor_topics = ', '.join(anchor_info['topics']) if anchor_info['topics'] else '通用'
        anchor_expressions = '\n'.join([f"- {expr}" for expr in anchor_info['expressions'][:3]])
        
        # 构建改写原因说明
        reasons_text = '\n'.join([f"- {reason}" for reason in rewrite_reasons]) if rewrite_reasons else "- 需要改写以提高检索准确度"
        
        prompt = f"""你是心理咨询知识库检索专家。根据改写原因和知识库特征，将用户问题改写为3-5个精准查询词。

对话历史上下文:
{context if context.strip() else '无'}

当前用户问题: {user_message}
改写原因: {reasons_text}
知识库主题: {anchor_topics}
知识库表达: {anchor_expressions}

改写要求: 
1. 必须结合对话历史理解用户真实意图，处理指代词和语境
2. 参考知识库表达方式
3. 从不同角度描述（症状/原因/解决方案/时间维度），每个查询词≤15字

示例1（口语化改写）:
问题: "上班好累，不想干了"
原因: 过于口语化
查询词:
工作倦怠
职业压力
职场疲劳
工作动力缺失

示例2（指代不明改写）:
对话: 助手问"迷茫从什么时候开始？" 用户答"我也不知道，一直以来都是这样的"
原因: 指代不明，需明确化
查询词:
长期迷茫感
缺乏生活方向感
一直感到迷茫
存在价值迷失

请只返回查询词，每行一个，不要编号。"""

        try:
            response = self._call_llm(prompt, max_tokens=100)
            # 解析生成的查询词
            rewritten_queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                line = line.lstrip('- •123456789.').strip()
                if line and len(line) > 2:
                    rewritten_queries.append(line)
            
            # 返回大模型实际生成的所有查询词（3-5个范围内）
            return rewritten_queries[:5]  # 最多5个改写查询，与提示词中的"3-5个"一致
            
        except Exception as e:
            print(f"引导式改写失败: {e}")
            return []
    
    def _fallback_query_generation(self, user_message: str, topics: List[str] = None, conversation_history: List[Dict] = None) -> List[str]:
        """降级方案：原始的查询生成方法"""
        
        # 构建上下文（使用完整的对话历史）
        context = ""
        if conversation_history:
            for msg in conversation_history:  # 使用完整的对话历史
                if msg.get('role') == 'user':
                    context += f"用户之前说: {msg.get('content', '')}\n"
                elif msg.get('role') == 'assistant':
                    context += f"助手之前回复: {msg.get('content', '')}\n"
        
        topics_str = ', '.join(topics) if topics else '未分类'
        prompt = f"""你是心理咨询知识库检索专家。根据用户问题和对话历史，生成3-5个不同角度的检索查询词。

用户问题: {user_message}
主题分类: {topics_str}
对话历史: {context if context.strip() else '无'}

要求: 使用心理学专业术语，从不同角度描述（症状/原因/解决方案），每个查询词≤15字。

示例:
问题: "最近失业了，感觉被生活抛弃了，不知道怎么办"
查询词:
失业压力
工作焦虑
职业迷茫
生活困境
重新规划

请只返回查询词，每行一个，不要编号。"""

        try:
            response = self._call_llm(prompt, max_tokens=100)
            generated_queries = []
            for line in response.strip().split('\n'):
                line = line.strip()
                line = line.lstrip('- •123456789.').strip()
                if line and len(line) > 2:
                    generated_queries.append(line)
            
            # 返回原始问题 + 大模型实际生成的查询词（最多5个）
            all_queries = [user_message] + generated_queries[:5]
            return all_queries
            
        except Exception as e:
            print(f"降级查询生成失败: {e}")
            return [user_message]
    
    
    def _call_llm(self, prompt: str, max_tokens: int = 1000) -> str:
        """调用LLM API"""
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            data = {
                "model": DEEPSEEK_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "max_tokens": max_tokens,
                "temperature": 0.3,  # 降低随机性，提高一致性
                "top_p": 0.8
            }
            
            response = requests.post(self.llm_url, headers=headers, json=data)
            response.raise_for_status()
            
            result = response.json()
            if 'choices' in result and len(result['choices']) > 0:
                return result['choices'][0]['message']['content']
            else:
                return ""
                
        except Exception as e:
            print(f"调用LLM时出错: {e}")
            return ""
    
    def analyze_user_input(self, user_message: str, conversation_history: List[Dict] = None, vector_store=None, force_retrieval: bool = False) -> Dict[str, Any]:
        """综合分析用户输入，返回决策结果
        
        Args:
            user_message: 用户消息
            conversation_history: 对话历史
            vector_store: 向量存储对象
            force_retrieval: 是否强制检索（由对话持续监控Agent触发）
        
        Returns:
            分析结果字典
        """
        
        # 1. 判断是否需要RAG（如果force_retrieval=True，直接设为True）
        if force_retrieval:
            need_rag = True
            print("🚨 强制检索模式：跳过need_rag判断")
        else:
            need_rag = self.should_use_rag(user_message, conversation_history)
        
        # 2. 主题分类（支持多标签）
        topics = []
        search_queries = []
        
        if need_rag:
            topics = self.classify_topic(user_message, conversation_history)
            # 3. 使用"先检索再引导"的查询生成机制
            search_queries = self.generate_search_queries_with_pre_retrieval(
                user_message, topics, conversation_history, vector_store
            )
        
        return {
            'need_rag': need_rag,
            'topics': topics,  # 改为复数形式，支持多个主题
            'topic': topics[0] if topics else None,  # 保持向后兼容
            'search_queries': search_queries if need_rag else [],  # 多个检索查询
            'search_query': search_queries[0] if need_rag and search_queries else None,  # 保持向后兼容
            'original_message': user_message,
            'forced': force_retrieval  # 标记是否为强制检索
        }

