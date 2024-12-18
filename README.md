# LangChain+LangFlow RAG工作流實作

## 1. 簡介

實作RAG WorkFlow，實現與大語言模型結合的聊天機器人

## 2. 目錄

- [LangChain+LangFlow RAG工作流實作](#langchainlangflow-rag工作流實作)
  - [1. 簡介](#1-簡介)
  - [2. 目錄](#2-目錄)
  - [3. 操作步驟](#3-操作步驟)
    - [3.1. 使用docker安裝langflow](#31-使用docker安裝langflow)
    - [3.2. 安裝LMStudio](#32-安裝lmstudio)
    - [3.3. lanflow RAG工作流設計](#33-lanflow-rag工作流設計)
      - [3.3.1. LangFlow RAG 向量資料庫與索引工作流建立](#331-langflow-rag-向量資料庫與索引工作流建立)
      - [3.3.2. LangFlow RAG 主工作流建立](#332-langflow-rag-主工作流建立)
      - [3.3.3. LangFlow RAG 測試](#333-langflow-rag-測試)
    - [3.4. lanflow RAG完成畫面](#34-lanflow-rag完成畫面)


## 3. 操作步驟


https://medium.com/@NeroHin/%E4%BD%BF%E7%94%A8-langflow-ollama-%E5%BF%AB%E9%80%9F%E6%A7%8B%E5%BB%BA-llama-3-8b-%E6%9C%AC%E5%9C%B0%E6%87%89%E7%94%A8-chatbot-chatpdf-%E8%88%87-macos-local-translator-06e5283f75ef


### 3.1. 使用docker安裝langflow

 > [!note] 小提示 
 > `LANGFLOW_CONFIG_DIR`、`langflow-postgres`路徑請記得調整為自己的windows目錄
 > 
 > 


```yaml
version: "3.8"  
services:  
langflow:  
image: langflowai/langflow:latest  
ports:  
- "7860:7860"  
depends_on:  
- postgres  
environment:  
- LANGFLOW_DATABASE_URL=postgresql://langflow:langflow@postgres:5432/langflow  
- LANGFLOW_CONFIG_DIR=/var/lib/langflow  
volumes:  
- langflow-data:/var/lib/langflow  
postgres:  
image: postgres:16  
environment:  
POSTGRES_USER: langflow  
POSTGRES_PASSWORD: langflow  
POSTGRES_DB: langflow  
ports:  
- "5432:5432"  
volumes:  
- langflow-postgres:/var/lib/postgresql/data  
volumes:  
langflow-postgres:  
langflow-data:
```

安裝成功畫面如下
![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218173707.png]]

### 3.2. 安裝LMStudio

https://lmstudio.ai/

點選紅框處安裝即可

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218173817.png]]

開啟`LMStudio`，點擊搜尋圖示，搜尋LLama3.2，可觀察後面有`GGUF`的模型，表示有經過模型轉檔和量化處理，比較適合在個人主機上面運行，最後點擊`download`等待下載

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218174055.png]]

下載後，請將`模型載入`，上方可察看目前載入哪一個模型，並可以`開始與模型對話`

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218174315.png]]


選取要啟動的模型，旁邊`兩個選項`請記得`勾選`，最後點擊`start`

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218174457.png]]

出現如紅框處，表示模型在運行當中

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218174612.png]]


### 3.3. lanflow RAG工作流設計

進入lanflow GUI介面，點擊New Flow，選擇Vector Store RAG

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218174736.png]]


#### 3.3.1. LangFlow RAG 向量資料庫與索引工作流建立
請按照紅框處設定

Path：對應路徑是在你當前docker langflow的綁定目錄下
Chunk Size: 為了避免模型超過上下文token上限，因此這邊設定為上限值為`4096`
LM Studio Base URL: http://localhost:1234/v1
ChromaDB: 名字和路徑可任意給，但記得這邊讀取文檔的工作流設定的名稱要和上面的一樣

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218174906.png]]

#### 3.3.2. LangFlow RAG 主工作流建立

![[Pasted image 20241218175424.png]]

`LMStudio` 區塊請添加以下提示詞

```
角色定位
你是一個專業的 RAG (檢索增強生成)智能助手,擅長結合知識庫內容提供準確、相關的回答。
==========================================================================
基本原則
1.無論使用者提出什麼問題，你都必須以流暢且完整的繁體中文回答，不要使用任何英文詞彙，除非特別要求，並確保所有回應都以繁體中文呈現
2.保持專業、客觀的語氣
3.回答要簡潔明瞭
4.確保資訊的準確性和時效性
5.請條列式逐行陳述，讓版面簡潔明瞭
==========================================================================
工作流程

步驟一：理解與檢索
仔細理解用戶問題的核心需求
從知識庫中檢索相關資訊
評估檢索結果的相關性

步驟二：回答生成
根據檢索到的資訊生成回答
清晰標註資訊來源
按照結構化格式組織內容

步驟三：品質確保
確保回答與問題高度相關
驗證資訊的準確性
必要時提供補充說明
==========================================================================
限制條件
1.僅使用知識庫中的資訊回答
2.如遇到知識庫中沒有的資訊,明確告知用戶
3.不進行推測或假設
4.保持資訊的客觀性
==========================================================================
特殊指令
1.當遇到模糊問題時,主動詢問細節
2.遇到技術性問題時,提供具體範例
3.需要時可使用表格或列表提升可讀性
```

#### 3.3.3. LangFlow RAG 測試

點擊`playground`，即可開始提問

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218175626.png]]

### 3.4. lanflow RAG完成畫面

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218175748.png]]

![[https://github.com/Mark850409/20241218_LangFlowRAG/refs/heads/master/images/Pasted image 20241218175811.png]]