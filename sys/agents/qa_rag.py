"""
Q/A система с RAG (Retrieval Augmented Generation) using LangGraph
Интерактивная система вопросов-ответов на основе ваших документов
"""

import os
from typing import TypedDict, List, Dict
from pathlib import Path
import re
from datetime import datetime

from langgraph.graph import StateGraph, START, END


# ============================================================================
# ОПРЕДЕЛЕНИЕ СОСТОЯНИЯ ГРАФА
# ============================================================================
class QARAGState(TypedDict):
    """
    Состояние Q/A RAG системы
    - query: Вопрос пользователя
    - documents: Загруженные документы
    - chunks: Разбитые на части документы
    - search_results: Найденные релевантные фрагменты
    - answer: Сгенерированный ответ
    - history: История вопросов и ответов
    """
    query: str
    documents: List[Dict[str, str]]
    chunks: List[Dict[str, str]]
    search_results: List[Dict[str, str]]
    answer: str
    history: List[Dict[str, str]]


# ============================================================================
# ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ ДЛЯ ХРАНЕНИЯ СОСТОЯНИЯ
# Чтобы не перезагружать документы при каждом вопросе
# ============================================================================
class DocumentStore:
    """
    Хранилище документов и чанков
    Загружается один раз и переиспользуется
    """
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

            # Разбиваем по предложениям
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


# Создаем глобальное хранилище
doc_store = DocumentStore()


# ============================================================================
# УЗЕЛ 1: ЗАГРУЗКА ДОКУМЕНТОВ (использует кэш)
# ============================================================================
def load_documents_node(state: QARAGState) -> QARAGState:
    """
    Узел загрузки документов
    Использует глобальное хранилище для кэширования
    """
    data_folder = Path(__file__).parent / "data"
    doc_store.load_documents(data_folder)

    return {
        **state,
        "documents": doc_store.documents,
        "chunks": doc_store.chunks
    }


# ============================================================================
# УЗЕЛ 2: ПОИСК РЕЛЕВАНТНЫХ ФРАГМЕНТОВ
# ============================================================================
def search_documents_node(state: QARAGState) -> QARAGState:
    """
    Узел поиска релевантных фрагментов

    Улучшенный алгоритм поиска:
    - Извлекает ключевые слова
    - Ищет точные и частичные совпадения
    - Ранжирует по релевантности
    """
    query = state.get("query", "")
    chunks = state.get("chunks", [])

    if not chunks:
        print("[!] Нет чанков для поиска")
        return {**state, "search_results": []}

    if not query:
        print("[!] Вопрос пустой")
        return {**state, "search_results": []}

    # Извлекаем ключевые слова (удаляем стоп-слова)
    stop_words = {
        "в", "на", "и", "с", "по", "для", "от", "к", "о", "об", "что", "как",
        "это", "все", "то", "так", "быть", "мочь", "весь", "свой", "один",
        "два", "три", "можно", "нужно", "есть", "был", "была", "были"
    }

    keywords = [
        word.lower()
        for word in re.findall(r'\w+', query)
        if word.lower() not in stop_words and len(word) >= 2
    ]

    print(f"[?] Ключевые слова: {keywords}")

    # Поиск и ранжирование
    results = []
    for chunk in chunks:
        content_lower = chunk["content"].lower()

        # Подсчет точных совпадений
        exact_matches = sum(1 for keyword in keywords if keyword in content_lower)

        # Подсчет частичных совпадений (для словоформ)
        partial_matches = sum(
            1 for keyword in keywords
            if any(keyword[:4] in word for word in content_lower.split() if len(word) > 4)
        )

        total_score = exact_matches * 2 + partial_matches

        if total_score > 0:
            results.append({
                **chunk,
                "score": total_score
            })

    # Сортируем по релевантности
    results.sort(key=lambda x: x["score"], reverse=True)

    # Берем топ-5 результатов
    top_results = results[:5]

    print(f"[+] Найдено релевантных фрагментов: {len(top_results)}")

    return {**state, "search_results": top_results}


# ============================================================================
# УЗЕЛ 3: ГЕНЕРАЦИЯ ОТВЕТА
# ============================================================================
def generate_answer_node(state: QARAGState) -> QARAGState:
    """
    Узел генерации ответа

    Формирует структурированный ответ на основе:
    - Найденных фрагментов
    - Вопроса пользователя
    - Контекста из документов
    """
    query = state.get("query", "")
    search_results = state.get("search_results", [])

    if not search_results:
        answer = f"""
[ВОПРОС]: {query}

[ОТВЕТ]: К сожалению, я не нашел информации по этому вопросу в загруженных документах.

[ПОДСКАЗКА]: Попробуйте:
  - Переформулировать вопрос
  - Использовать другие ключевые слова
  - Проверить, есть ли нужная информация в документах
"""
        return {**state, "answer": answer.strip()}

    # Формируем развернутый ответ
    answer_parts = []
    answer_parts.append(f"[ВОПРОС]: {query}\n")
    answer_parts.append("="*70)

    # Основной ответ на основе самого релевантного фрагмента
    best_match = search_results[0]
    answer_parts.append(f"\n[ОТВЕТ]:")
    answer_parts.append(f"{best_match['content']}\n")

    # Дополнительная информация из других источников
    if len(search_results) > 1:
        answer_parts.append("\n[ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ]:\n")
        for i, result in enumerate(search_results[1:4], 1):
            answer_parts.append(f"{i}. (Источник: {result['source']})")
            answer_parts.append(f"   {result['content'][:200]}...\n")

    # Источники
    sources = list(set([r['source'] for r in search_results]))
    answer_parts.append(f"\n[ИСТОЧНИКИ]: {', '.join(sources)}")
    answer_parts.append(f"[РЕЛЕВАНТНОСТЬ]: {'*' * min(search_results[0]['score'], 5)}/5")

    answer = "\n".join(answer_parts)
    return {**state, "answer": answer}


# ============================================================================
# СОЗДАНИЕ ГРАФА
# ============================================================================
def create_qa_graph():
    """
    Создает граф для Q/A системы

    Поток: START → Загрузка → Поиск → Генерация → END
    """
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
    """
    Интерактивная система вопросов-ответов
    """
    def __init__(self):
        self.graph = create_qa_graph()
        self.history = []
        print("\n" + "="*70)
        print("[*] Q/A СИСТЕМА С RAG ИНИЦИАЛИЗИРОВАНА")
        print("="*70)
        print("\nКоманды:")
        print("  - Введите вопрос для получения ответа")
        print("  - 'история' - показать историю вопросов")
        print("  - 'очистить' - очистить историю")
        print("  - 'выход' или 'quit' - завершить работу")
        print("="*70 + "\n")

    def ask(self, question: str) -> str:
        """
        Задает вопрос системе

        Args:
            question: Вопрос пользователя

        Returns:
            Ответ системы
        """
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

        print(f"\n[?] Обрабатываю вопрос: {question}")
        print("-"*70)

        final_state = self.graph.invoke(initial_state)
        answer = final_state["answer"]

        # Сохраняем в историю
        self.history.append({
            "question": question,
            "answer": answer,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })

        return answer

    def show_history(self):
        """Показывает историю вопросов и ответов"""
        if not self.history:
            print("\n[i] История пуста\n")
            return

        print("\n" + "="*70)
        print("[*] ИСТОРИЯ ВОПРОСОВ И ОТВЕТОВ")
        print("="*70 + "\n")

        for i, item in enumerate(self.history, 1):
            print(f"{i}. [{item['timestamp']}]")
            print(f"   Вопрос: {item['question']}")
            print(f"   Ответ: {item['answer'][:100]}...")
            print("-"*70)

    def clear_history(self):
        """Очищает историю"""
        self.history = []
        print("\n[+] История очищена\n")

    def run_interactive(self):
        """
        Запускает интерактивный режим
        """
        while True:
            try:
                question = input("\n[Q/A] Ваш вопрос: ").strip()

                if not question:
                    continue

                # Обработка команд
                if question.lower() in ['выход', 'exit', 'quit', 'q']:
                    print("\n[*] Спасибо за использование Q/A системы! До свидания!\n")
                    break

                elif question.lower() in ['история', 'history', 'h']:
                    self.show_history()
                    continue

                elif question.lower() in ['очистить', 'clear', 'c']:
                    self.clear_history()
                    continue

                elif question.lower() in ['помощь', 'help']:
                    print("\n[*] Доступные команды:")
                    print("  - Любой вопрос - получить ответ из документов")
                    print("  - 'история' - показать все вопросы и ответы")
                    print("  - 'очистить' - очистить историю")
                    print("  - 'выход' - завершить работу\n")
                    continue

                # Обрабатываем вопрос
                answer = self.ask(question)
                print("\n" + answer + "\n")
                print("="*70)

            except KeyboardInterrupt:
                print("\n\n[*] Работа прервана пользователем. До свидания!\n")
                break
            except Exception as e:
                print(f"\n[-] Ошибка: {e}\n")


# ============================================================================
# ГЛАВНАЯ ФУНКЦИЯ
# ============================================================================
if __name__ == "__main__":
    """
    Запуск интерактивной Q/A системы

    Использование:
    1. Поместите .txt файлы в папку 'data'
    2. Запустите: python qa_rag.py
    3. Задавайте вопросы на русском языке

    Примеры вопросов:
    - Что такое искусственный интеллект?
    - Расскажи про машинное обучение
    - Как работает RAG?
    - Что такое нейронные сети?
    """

    # Создаем и запускаем систему
    qa_system = QASystem()
    qa_system.run_interactive()
