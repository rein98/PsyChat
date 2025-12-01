# 心理咨询伴侣 - 系统架构

## 项目类型
**RAG-Augmented Conversational Agent** (RAG增强型对话智能体)

## 核心Pipeline流程图

```mermaid
graph TD
    A[用户输入] --> B{Psychology Agent<br/>智能决策层}
    
    B --> C{是否需要RAG?}
    C -->|YES| D[主题分类]
    C -->|NO| E[直接生成回答]
    
    D --> F[轻检索获取锚点]
    F --> G{是否需要改写?}
    
    G -->|YES| H[引导式查询改写]
    G -->|NO| I[使用原始查询]
    
    H --> J[向量检索<br/>ChromaDB]
    I --> J
    
    J --> K{找到相关文档?}
    K -->|YES| L[构建上下文]
    K -->|NO| E
    
    L --> M[LLM生成回答<br/>DeepSeek]
    E --> M
    
    M --> N[更新对话历史]
    N --> O{对话监控Agent<br/>连续3轮无RAG?}
    
    O -->|YES| P[强制触发RAG检索]
    O -->|NO| Q[返回结果]
    
    P --> D
    Q --> R[输出回答]
    
    style B fill:#667eea,color:#fff
    style J fill:#764ba2,color:#fff
    style O fill:#ff9a9e,color:#fff
    style M fill:#fecfef,color:#333
```

## 系统架构层次

```mermaid
graph LR
    subgraph "用户交互层"
        A1[Web界面<br/>FastAPI]
        A2[CLI界面]
    end
    
    subgraph "核心业务层"
        B1[RAG System<br/>主控制器]
        B2[Psychology Agent<br/>智能决策]
    end
    
    subgraph "检索增强层"
        C1[Vector Store<br/>ChromaDB]
        C2[Data Processor<br/>文档处理]
    end
    
    subgraph "AI服务层"
        D1[DeepSeek LLM<br/>对话生成]
        D2[Alibaba Embedding<br/>向量化]
    end
    
    A1 --> B1
    A2 --> B1
    B1 --> B2
    B2 --> C1
    B1 --> C1
    C1 --> D2
    B1 --> D1
    C2 --> C1
    
    style B1 fill:#667eea,color:#fff
    style B2 fill:#764ba2,color:#fff
    style C1 fill:#ff9a9e,color:#fff
    style D1 fill:#fecfef,color:#333
```

## 核心组件说明

### 1. Psychology Agent（智能决策层）
- **功能**：智能判断是否需要RAG、主题分类、查询改写
- **特点**：基于LLM的决策能力，动态调整检索策略

### 2. RAG System（主控制器）
- **功能**：整合Agent、向量库、LLM，管理对话流程
- **特点**：对话历史管理、强制检索监控

### 3. Vector Store（检索层）
- **功能**：向量存储、语义检索、主题过滤
- **技术**：ChromaDB + Alibaba Embedding

### 4. Conversation Monitor（对话监控）
- **功能**：追踪连续无RAG轮数，自动触发强制检索
- **阈值**：连续3轮无RAG触发

## 技术栈
- **LLM**: DeepSeek Chat
- **Embedding**: Alibaba Text-Embedding-v4
- **Vector DB**: ChromaDB
- **Web框架**: FastAPI
- **AI理论**: 理情行为疗法 (REBT)

