"""
Упрощенный RAG (Retrieval Augmented Generation) Agent using LangGraph
Агент для работы с документами через простой поиск и обработку
"""

import os
from typing import TypedDict, List, Dict
from pathlib import Path
import re

from langgraph.graph import StateGraph, START, END


# ============================================================================
# ОПРЕДЕЛЕНИЕ СОСТОЯНИЯ ГРАФА
# Состояние содержит всю информацию, передаваемую между узлами графа
# ============================================================================
class RAGState(TypedDict):
    """
    Состояние RAG-агента
    - query: Запрос пользователя
    - documents: Загруженные документы (словари с path и content)
    - chunks: Разбитые на части документы
    - search_results: Найденные релевантные фрагменты
    - answer: Сгенерированный ответ
    """
    query: str
    documents: List[Dict[str, str]]
    chunks: List[Dict[str, str]]
    search_results: List[Dict[str, str]]
    answer: str


# ============================================================================
# УЗЕЛ 1: ЗАГРУЗКА ДОКУМЕНТОВ
# Загружает все текстовые файлы из указанной папки
# ============================================================================
def load_documents_node(state: RAGState) -> RAGState:
    """
    Узел загрузки документов из папки data

    Что делает:
    - Сканирует папку 'data' в текущей директории
    - Загружает все текстовые файлы (.txt)
    - Сохраняет содержимое в состояние графа

    Возвращает обновленное состояние с загруженными документами
    """
    print("---Узел 1: Загрузка документов---")

    # Путь к папке с данными
    data_folder = Path(__file__).parent / "data"

    # Проверяем существование папки
    if not data_folder.exists():
        print(f"[!] Папка {data_folder} не существует. Создаем...")
        data_folder.mkdir(parents=True, exist_ok=True)
        return {**state, "documents": []}

    documents = []

    # Загружаем все .txt файлы
    for file_path in data_folder.glob("*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append({
                    "path": str(file_path),
                    "filename": file_path.name,
                    "content": content
                })
        except Exception as e:
            print(f"[-] Ошибка при чтении {file_path}: {e}")

    print(f"[+] Загружено документов: {len(documents)}")
    return {**state, "documents": documents}


# ============================================================================
# УЗЕЛ 2: РАЗБИВКА НА ЧАНКИ
# Разбивает большие документы на маленькие части для лучшего поиска
# ============================================================================
def chunk_documents_node(state: RAGState) -> RAGState:
    """
    Узел разбивки документов на чанки (фрагменты)

    Что делает:
    - Берет загруженные документы
    - Разбивает их на маленькие части (по 500 символов)
    - Сохраняет метаданные о источнике каждого фрагмента

    Зачем это нужно:
    - Большие документы сложно обрабатывать целиком
    - Маленькие части позволяют точнее находить релевантную информацию
    """
    print("---Узел 2: Разбивка документов на чанки---")

    documents = state.get("documents", [])

    if not documents:
        print("[!] Нет документов для разбивки")
        return {**state, "chunks": []}

    chunks = []
    chunk_size = 500  # Размер чанка в символах

    for doc in documents:
        content = doc["content"]
        filename = doc["filename"]

        # Разбиваем по предложениям для более естественных границ
        sentences = re.split(r'(?<=[.!?])\s+', content)

        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += sentence + " "
            else:
                if current_chunk:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "source": filename
                    })
                current_chunk = sentence + " "

        # Добавляем последний чанк
        if current_chunk:
            chunks.append({
                "content": current_chunk.strip(),
                "source": filename
            })

    print(f"[+] Создано чанков: {len(chunks)}")
    return {**state, "chunks": chunks}


# ============================================================================
# УЗЕЛ 3: ПРОСТОЙ ПОИСК ПО КЛЮЧЕВЫМ СЛОВАМ
# Ищет фрагменты, содержащие слова из запроса
# ============================================================================
def search_documents_node(state: RAGState) -> RAGState:
    """
    Узел поиска релевантных документов

    Что делает:
    - Берет запрос пользователя
    - Извлекает ключевые слова из запроса
    - Ищет чанки, содержащие эти ключевые слова
    - Ранжирует результаты по количеству совпадений

    Это упрощенная версия без векторного поиска.
    Для production используйте FAISS или другие векторные БД.
    """
    print("---Узел 3: Поиск релевантных фрагментов---")

    query = state.get("query", "")
    chunks = state.get("chunks", [])

    if not chunks:
        print("[!] Нет чанков для поиска")
        return {**state, "search_results": []}

    if not query:
        print("[!] Запрос пустой")
        return {**state, "search_results": []}

    # Извлекаем ключевые слова из запроса (удаляем стоп-слова)
    stop_words = {"в", "на", "и", "с", "по", "для", "от", "к", "о", "об", "что", "как"}
    keywords = [
        word.lower()
        for word in re.findall(r'\w+', query)
        if word.lower() not in stop_words and len(word) >= 2
    ]

    print(f"[?] Ключевые слова для поиска: {keywords}")

    # Ищем и ранжируем чанки
    results = []
    for chunk in chunks:
        content_lower = chunk["content"].lower()
        score = sum(1 for keyword in keywords if keyword in content_lower)

        if score > 0:
            results.append({
                **chunk,
                "score": score
            })

    # Сортируем по релевантности
    results.sort(key=lambda x: x["score"], reverse=True)

    # Берем топ-3 результата
    top_results = results[:3]

    print(f"[+] Найдено релевантных фрагментов: {len(top_results)}")

    for i, result in enumerate(top_results, 1):
        preview = result["content"][:80].replace("\n", " ")
        print(f"  {i}. [{result['source']}] Score: {result['score']} - {preview}...")

    return {**state, "search_results": top_results}


# ============================================================================
# УЗЕЛ 4: ГЕНЕРАЦИЯ ОТВЕТА
# Формирует ответ на основе найденных документов
# ============================================================================
def generate_answer_node(state: RAGState) -> RAGState:
    """
    Узел генерации ответа

    Что делает:
    - Берет найденные релевантные фрагменты
    - Объединяет их в контекст
    - Формирует структурированный ответ

    Примечание:
    - Это упрощенная версия без LLM
    - Просто возвращает найденную информацию
    - Для полноценной генерации подключите OpenAI API или другую LLM
    """
    print("---Узел 4: Генерация ответа---")

    query = state.get("query", "")
    search_results = state.get("search_results", [])

    if not search_results:
        answer = "[-] К сожалению, не найдено релевантной информации по вашему запросу."
        print(answer)
        return {**state, "answer": answer}

    # Формируем ответ из найденных фрагментов
    answer_parts = [f"[*] ОТВЕТ НА ЗАПРОС: \"{query}\"\n"]
    answer_parts.append("="*60)
    answer_parts.append("\n[*] НАЙДЕННАЯ ИНФОРМАЦИЯ:\n")

    for i, result in enumerate(search_results, 1):
        answer_parts.append(f"\n[{i}] Источник: {result['source']}")
        answer_parts.append(f"Релевантность: {'*' * min(result['score'], 5)}")
        answer_parts.append(f"\n{result['content']}\n")
        answer_parts.append("-"*60)

    answer_parts.append("\n[i] РЕЗЮМЕ:")
    answer_parts.append(f"Найдено {len(search_results)} релевантных фрагмента(ов)")
    answer_parts.append("\n[i] Для более точных ответов подключите LLM (OpenAI, Anthropic, и т.д.)")

    answer = "\n".join(answer_parts)
    print("[+] Ответ сформирован")

    return {**state, "answer": answer}


# ============================================================================
# ПОСТРОЕНИЕ ГРАФА RAG
# Объединяет все узлы в единый граф обработки
# ============================================================================
def create_rag_graph():
    """
    Создает и компилирует RAG граф

    Структура графа:
    START → Загрузка → Разбивка → Поиск → Генерация → END

    Каждый узел выполняет свою задачу и передает обновленное состояние дальше
    """
    print("[*] Создаем RAG граф...")

    # Создаем граф с определенным типом состояния
    builder = StateGraph(RAGState)

    # Добавляем узлы в граф
    builder.add_node("load_documents", load_documents_node)
    builder.add_node("chunk_documents", chunk_documents_node)
    builder.add_node("search_documents", search_documents_node)
    builder.add_node("generate_answer", generate_answer_node)

    # Определяем порядок выполнения узлов
    builder.add_edge(START, "load_documents")
    builder.add_edge("load_documents", "chunk_documents")
    builder.add_edge("chunk_documents", "search_documents")
    builder.add_edge("search_documents", "generate_answer")
    builder.add_edge("generate_answer", END)

    # Компилируем граф
    graph = builder.compile()
    print("[+] RAG граф создан")

    return graph


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# Точка входа в программу
# ============================================================================
if __name__ == "__main__":
    """
    Основная функция для запуска RAG агента

    Как использовать:
    1. Положите текстовые файлы (.txt) в папку 'data'
    2. Запустите скрипт: python rag.py
    3. Граф обработает документы и ответит на запрос

    Примеры запросов:
    - "Что такое искусственный интеллект?"
    - "Расскажи про машинное обучение"
    - "Что такое RAG?"
    """

    # Создаем граф
    rag_graph = create_rag_graph()

    # Пример запроса - можете изменить на свой
    user_query = "искусственный интеллект машинное обучение"

    # Начальное состояние
    initial_state = {
        "query": user_query,
        "documents": [],
        "chunks": [],
        "search_results": [],
        "answer": ""
    }

    print("\n" + "="*80)
    print("[>] ЗАПУСК RAG АГЕНТА")
    print("="*80 + "\n")

    # Запускаем граф
    final_state = rag_graph.invoke(initial_state)

    print("\n" + "="*80)
    print("[*] РЕЗУЛЬТАТ")
    print("="*80)
    print(final_state["answer"])
    print("="*80 + "\n")
