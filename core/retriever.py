# core/retriever.py
import os
from typing import List

from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate

from core.llm_config import embeddings, llm

def initialize_rag(
    embeddings,
    md_folder: str = "data/documents",
    persist_path: str = "data/vectorstores",
):
    os.makedirs(persist_path, exist_ok=True)
    faiss_index_path = os.path.join(persist_path, "index.faiss")
    
    if os.path.exists(faiss_index_path):
        print("Loading existing FAISS vectorstore from disk...")
        vectorstore = FAISS.load_local(
            persist_path, embeddings, allow_dangerous_deserialization=True
        )
    else:
        print("Creating new FAISS vectorstore from Markdown documents...")
        all_docs: List[Document] = []
        if not os.path.exists(md_folder):
            raise FileNotFoundError(f"No RAG Folder: {md_folder}")

        headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3"),
        ]
        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=150
        )

        for file in os.listdir(md_folder):
            if file.endswith(".md"):
                file_path = os.path.join(md_folder, file)
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
                
                md_docs = markdown_splitter.split_text(text)
                
                for doc in md_docs:
                    doc.metadata["source"] = file
                
                split_docs = text_splitter.split_documents(md_docs)
                all_docs.extend(split_docs)

        vectorstore = FAISS.from_documents(all_docs, embeddings)
        vectorstore.save_local(persist_path)
        print("Completed creating and saving FAISS vectorstore.")

    return vectorstore.as_retriever(search_kwargs={"k": 5})

retriever = initialize_rag(embeddings)

# =========================
# 마케팅 전략 RAG 프롬프트 (수정 필요 시 여기서 조정)
# =========================
marketing_prompt = ChatPromptTemplate.from_template(
    """
당신은 성동구의 식당, 카페 사장님들을 위한 최고의 마케팅 전략가입니다.

규칙:
1. [데이터 분석]을 통해 사장님의 현재 위치와 문제점을 파악하세요.
2. [참고 자료]에 포함된 '통계 지표'를 인용하여 문제의 심각성이나 기준을 제시하세요.
3. [참고 자료]에 포함된 '마케팅 이론'과 '성공 사례'를 바탕으로 구체적인 실행 방안(무엇/어디서/어떻게)을 제시하세요.
4. 반드시 어떤 문서/사례를 참고했는지 괄호 안에 출처를 명시하세요.

[데이터 분석]
{analysis_result}

[참고 자료]
{context}

[질문]
{question}

[전략 제안]
"""
)

def rag_chain(query: str, analysis_result: str) -> tuple[str, str, str, str]:
    docs = retriever.invoke(query)
    context = "\n\n".join([d.page_content for d in docs])

    debug_lines = []
    ref_lines = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        # 헤더 스플리터를 쓰면 metadata에 Header 2, Header 3 등이 남습니다.
        header_info = " > ".join([v for k, v in d.metadata.items() if "Header" in k])
        
        # Debug info
        preview = d.page_content[:200].replace("\n", " ")
        debug_lines.append(f"{i}) source={src} | section={header_info}\npreview={preview}...\n")
        
        # Ref info
        excerpt = d.page_content.strip().replace("\n", " ")
        excerpt = excerpt[:200] + ("..." if len(excerpt) > 200 else "")
        ref_lines.append(f"- [{i}] {src} (섹션: {header_info})\n  └ 인용: {excerpt}")

    rag_debug = "\n".join(debug_lines)
    rag_refs = "\n".join(ref_lines)
    rag_query = query

    prompt_text = marketing_prompt.format_prompt(
        context=context, question=query, analysis_result=analysis_result
    ).to_string()

    strategy = llm.invoke(prompt_text).content
    return strategy, rag_debug, rag_query, rag_refs