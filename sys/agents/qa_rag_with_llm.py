"""
Q/A система с RAG и LLM (OpenAI/Anthropic) using LangGraph
Интерактивная система с интеллектуальной генерацией ответов
"""

import os
from typing import TypedDict, List, Dict, Optional
from pathlib import Path
import re
from datetime import datetime
from dotenv import load_dotenv

from langgraph.graph import StateGraph, START, END

# Загружаем переменные окружения из .env файла
load_dotenv()


# ============================================================================
# КОНФИГУРАЦИЯ API
# ============================================================================
class Config:
    """
    Конфигурация для API ключей и настроек

    ВАЖНО: Есть 3 способа указать API ключ:

    1. Через .env файл (РЕКОМЕНДУЕТСЯ):
       - Создайте файл .env в этой же папке
       - Добавьте: OPENAI_API_KEY=ваш-ключ

    2. Через переменные окружения:
       - Windows: set OPENAI_API_KEY=ваш-ключ
       - Linux/Mac: export OPENAI_API_KEY=ваш-ключ

    3. Прямо в коде (НЕ РЕКОМЕНДУЕТСЯ для продакшена):
       - Раскомментируйте строку ниже и вставьте ключ
    """

    # ВАРИАНТ 3: Вставьте ваш API ключ здесь (не коммитьте в Git!)
    # OPENAI_API_KEY = "sk-your-api-key-here"

    # Пытаемся получить ключ из разных источников
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or None
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or None

    # Настройки модели
    MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
    TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
    MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))

    # Режим работы
    USE_LLM = True  # True - использовать LLM, False - простой поиск

    @classmethod
    def has_api_key(cls) -> bool:
        """Проверяет, установлен ли API ключ"""
        return cls.OPENAI_API_KEY is not None or cls.ANTHROPIC_API_KEY is not None


# ============================================================================
# ОПРЕДЕЛЕНИЕ СОСТОЯНИЯ ГРАФА
# ============================================================================
class QARAGState(TypedDict):
    """Состояние Q/A RAG системы"""
    query: str
    documents: List[Dict[str, str]]
    chunks: List[Dict[str, str]]
    search_results: List[Dict[str, str]]
    answer: str
    history: List[Dict[str, str]]


# ============================================================================
# ХРАНИЛИЩЕ ДОКУМЕНТОВ
# ============================================================================
class DocumentStore:
    """Хранилище документов и чанков"""
    def __init__(self):
        self.documents = []
        self.chunks = []
        self.loaded = False

    def load_documents(self, data_folder: Path):
        """Загружает документы из папки"""
        if self.loaded:
            print("[i] Документы уже загружены, используем кэш")
            return

        print("[*] Загружаем документы...")

        if not data_folder.exists():
            print(f"[!] Папка {data_folder} не существует. Создаем...")
            data_folder.mkdir(parents=True, exist_ok=True)
            return

        for file_path in data_folder.glob("*.txt"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    self.documents.append({
                        "path": str(file_path),
                        "filename": file_path.name,
                        "content": content
                    })
            except Exception as e:
                print(f"[-] Ошибка при чтении {file_path}: {e}")

        print(f"[+] Загружено документов: {len(self.documents)}")
        self._create_chunks()
        self.loaded = True

    def _create_chunks(self):
        """Создает чанки из документов"""
        print("[*] Создаем чанки...")
        chunk_size = 500

        for doc in self.documents:
            content = doc["content"]
            filename = doc["filename"]
            sentences = re.split(r'(?<=[.!?])\s+', content)

            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) <= chunk_size:
                    current_chunk += sentence + " "
                else:
                    if current_chunk:
                        self.chunks.append({
                            "content": current_chunk.strip(),
                            "source": filename
                        })
                    current_chunk = sentence + " "

            if current_chunk:
                self.chunks.append({
                    "content": current_chunk.strip(),
                    "source": filename
                })

        print(f"[+] Создано чанков: {len(self.chunks)}")


doc_store = DocumentStore()


# ============================================================================
# УЗЛЫ ГРАФА
# ============================================================================
def load_documents_node(state: QARAGState) -> QARAGState:
    """Узел загрузки документов"""
    data_folder = Path(__file__).parent / "data"
    doc_store.load_documents(data_folder)
    return {
        **state,
        "documents": doc_store.documents,
        "chunks": doc_store.chunks
    }


def search_documents_node(state: QARAGState) -> QARAGState:
    """Узел поиска релевантных фрагментов"""
    query = state.get("query", "")
    chunks = state.get("chunks", [])

    if not chunks or not query:
        return {**state, "search_results": []}

    # Извлекаем ключевые слова
    stop_words = {
        "в", "на", "и", "с", "по", "для", "от", "к", "о", "об", "что", "как",
        "это", "все", "то", "так", "быть", "мочь", "весь", "свой", "один"
    }

    keywords = [
        word.lower()
        for word in re.findall(r'\w+', query)
        if word.lower() not in stop_words and len(word) >= 2
    ]

    print(f"[?] Ключевые слова: {keywords}")

    # Поиск
    results = []
    for chunk in chunks:
        content_lower = chunk["content"].lower()
        score = sum(2 if keyword in content_lower else 0 for keyword in keywords)
        score += sum(
            1 for keyword in keywords
            if any(keyword[:4] in word for word in content_lower.split() if len(word) > 4)
        )

        if score > 0:
            results.append({**chunk, "score": score})

    results.sort(key=lambda x: x["score"], reverse=True)
    top_results = results[:5]

    print(f"[+] Найдено релевантных фрагментов: {len(top_results)}")
    return {**state, "search_results": top_results}


def generate_answer_with_llm(query: str, context: str) -> str:
    """
    Генерирует ответ используя LLM (OpenAI)

    Args:
        query: Вопрос пользователя
        context: Контекст из найденных документов

    Returns:
        Сгенерированный ответ
    """
    try:
        from langchain_openai import ChatOpenAI

        # Создаем модель
        llm = ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            model=Config.MODEL_NAME,
            temperature=Config.TEMPERATURE,
            max_tokens=Config.MAX_TOKENS
        )

        # Формируем промпт
        prompt = f"""Ты - полезный ассистент, который отвечает на вопросы на основе предоставленного контекста.

КОНТЕКСТ:
{context}

ВОПРОС: {query}

ИНСТРУКЦИИ:
- Отвечай ТОЛЬКО на основе предоставленного контекста
- Если в контексте нет информации для ответа, скажи об этом
- Отвечай кратко и по существу (2-3 предложения)
- Используй русский язык
- Не добавляй информацию, которой нет в контексте

ОТВЕТ:"""

        # Генерируем ответ
        response = llm.invoke(prompt)
        return response.content

    except ImportError:
        return "[!] Ошибка: Установите langchain-openai: pip install langchain-openai"
    except Exception as e:
        return f"[-] Ошибка при генерации ответа: {str(e)}"


def generate_answer_node(state: QARAGState) -> QARAGState:
    """Узел генерации ответа"""
    query = state.get("query", "")
    search_results = state.get("search_results", [])

    if not search_results:
        answer = f"""[ВОПРОС]: {query}

[ОТВЕТ]: К сожалению, я не нашел информации по этому вопросу в загруженных документах.

[ПОДСКАЗКА]: Попробуйте переформулировать вопрос или проверьте наличие информации в документах."""
        return {**state, "answer": answer.strip()}

    # Формируем контекст
    context = "\n\n".join([
        f"[Источник: {r['source']}]\n{r['content']}"
        for r in search_results[:3]
    ])

    # Генерируем ответ
    if Config.USE_LLM and Config.has_api_key():
        print("[*] Генерируем ответ с помощью LLM...")
        llm_answer = generate_answer_with_llm(query, context)

        sources = list(set([r['source'] for r in search_results]))
        answer = f"""[ВОПРОС]: {query}

[ОТВЕТ]: {llm_answer}

[ИСТОЧНИКИ]: {', '.join(sources)}
[РЕЛЕВАНТНОСТЬ]: {'*' * min(search_results[0]['score'], 5)}/5
"""
    else:
        # Простой ответ без LLM
        if not Config.has_api_key():
            print("[!] API ключ не найден. Используем простой режим.")
            print("[i] Добавьте OPENAI_API_KEY в .env файл для использования LLM")

        best_match = search_results[0]
        sources = list(set([r['source'] for r in search_results]))

        answer = f"""[ВОПРОС]: {query}

[ОТВЕТ]: {best_match['content']}

[ИСТОЧНИКИ]: {', '.join(sources)}
[РЕЛЕВАНТНОСТЬ]: {'*' * min(best_match['score'], 5)}/5

[i] Для более точных ответов добавьте OPENAI_API_KEY в .env файл
"""

    return {**state, "answer": answer}


# ============================================================================
# СОЗДАНИЕ ГРАФА
# ============================================================================
def create_qa_graph():
    """Создает граф для Q/A системы"""
    builder = StateGraph(QARAGState)

    builder.add_node("load_documents", load_documents_node)
    builder.add_node("search_documents", search_documents_node)
    builder.add_node("generate_answer", generate_answer_node)

    builder.add_edge(START, "load_documents")
    builder.add_edge("load_documents", "search_documents")
    builder.add_edge("search_documents", "generate_answer")
    builder.add_edge("generate_answer", END)

    return builder.compile()


# ============================================================================
# ИНТЕРАКТИВНАЯ Q/A СИСТЕМА
# ============================================================================
class QASystem:
    """Интерактивная система вопросов-ответов"""
    def __init__(self):
        self.graph = create_qa_graph()
        self.history = []

        print("\n" + "="*70)
        print("[*] Q/A СИСТЕМА С RAG")
        print("="*70)

        # Проверяем наличие API ключа
        if Config.has_api_key():
            print(f"[+] LLM режим АКТИВЕН (модель: {Config.MODEL_NAME})")
        else:
            print("[!] LLM режим ВЫКЛЮЧЕН (API ключ не найден)")
            print("[i] Для активации создайте .env файл с OPENAI_API_KEY")

        print("\nКоманды:")
        print("  - Введите вопрос для получения ответа")
        print("  - 'история' - показать историю")
        print("  - 'выход' - завершить работу")
        print("="*70 + "\n")

    def ask(self, question: str) -> str:
        """Задает вопрос системе"""
        if not question.strip():
            return "[!] Пожалуйста, задайте вопрос"

        initial_state = {
            "query": question,
            "documents": [],
            "chunks": [],
            "search_results": [],
            "answer": "",
            "history": self.history
        }

        print(f"\n[?] Обрабатываю: {question}")
        print("-"*70)

        final_state = self.graph.invoke(initial_state)
        answer = final_state["answer"]

        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        return answer

    def run_interactive(self):
        """Запускает интерактивный режим"""
        while True:
            try:
                question = input("\n[Q/A] Ваш вопрос: ").strip()

                if not question:
                    continue

                if question.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("\n[*] До свидания!\n")
                    break

                elif question.lower() in ['история', 'history']:
                    if not self.history:
                        print("\n[i] История пуста\n")
                    else:
                        print("\n" + "="*70)
                        for i, item in enumerate(self.history, 1):
                            print(f"{i}. [{item['timestamp']}] {item['question']}")
                        print("="*70)
                    continue

                answer = self.ask(question)
                print("\n" + answer + "\n")
                print("="*70)

            except KeyboardInterrupt:
                print("\n\n[*] До свидания!\n")
                break
            except Exception as e:
                print(f"\n[-] Ошибка: {e}\n")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================
if __name__ == "__main__":
    """
    Запуск Q/A системы с RAG

    НАСТРОЙКА API КЛЮЧА:

    1. Создайте файл .env в этой папке
    2. Добавьте строку: OPENAI_API_KEY=ваш-ключ-здесь
    3. Запустите: python qa_rag_with_llm.py

    Где получить API ключ:
    - OpenAI: https://platform.openai.com/api-keys
    - Anthropic: https://console.anthropic.com/
    """

    qa_system = QASystem()
    qa_system.run_interactive()
