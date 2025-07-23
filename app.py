import os
import sys
import importlib
import shutil
import requests
import logging 

import yaml
import gradio as gr
from langchain_core.document_loaders import BaseLoader
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.prompts import PromptTemplate
import langchain_core.output_parsers as parsers
from langchain_google_genai import ChatGoogleGenerativeAI

# Изменение уровня логирования для unstructured
logging.getLogger("unstructured").setLevel(logging.ERROR)

# Базовый конфиг для логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)

class PortfolioAssistant:
    """
    AI-ассистент на базе RAG для ответов на вопросы по GitHub-портфолио.

    Этот класс представляет собой полноценную систему, которая загружает
    документы из репозитория, индексирует их в векторной базе данных
    и использует языковую модель для генерации ответов на основе
    найденной информации. Управляется через конфигурационный файл.
    """

    def __init__(self, config_file="config.yaml"):
        """
        Инициализирует экземпляр ассистента по портфолио.

        Этот метод выполняет полную настройку ассистента:
        1.  Загружает и валидирует конфигурационный файл `config.yaml`.
        2.  Динамически импортирует классы загрузчиков документов (loaders).
        3.  При необходимости скачивает файлы из GitHub-репозитория.
        4.  Загружает документы из локальной директории.
        5.  Инициализирует модель эмбеддингов и векторную базу ChromaDB.
        6.  Загружает LLM и создает цепочки (chains) для обработки запросов.
        """

        logging.info("Загрузка конфигурации...")
        if not os.path.exists(config_file):
            msg = "Файл конфигурации отсутствует, инициализация невозможна"
            self._raise_error(msg)
        else:
            with open(config_file, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)

        # Получение загрузчиков и их параметров
        logging.info("Импорт загрузчиков и их параметров...")
        msg = ""
        if "loaders" not in self.config: 
            msg = "Отсутствуют загрузчики в файле конфигурации" 
        elif not isinstance(self.config["loaders"], dict):
            msg = ("Некорректный формат списка загрузчиков, "
                   "ожидается словарь '.ext': 'full.path.to.LoaderClass'")
        elif not self.config["loaders"]:
            msg = "Список загрузчиков пуст"
        if msg:
            self._raise_error(msg)
        self.loaders, self.loader_params = {}, {}
        # Динамический импорт библиотек и загрузка классов
        for k, v in self.config["loaders"].items():
            self.loaders[k] = self._import_loader(v)
        if not isinstance(self.config["loader_params"], dict):
            logging.warning("Отсутствуют параметры для загрузчиков")
        else:
            for k, v in self.config["loader_params"].items():
                self.loader_params[k] = v

        # загрузка хранилища документов
        self.data_dir = self.config.get("data_dir", "data")
        self.chroma_dir = self.config.get("chroma_dir", "chroma")
        # Проверка наличия папки с документами для загрузки
        if not os.path.exists(self.data_dir):
            logging.warning("Отсутствует директория с документами, "
                            "данные будут загружены из репозитория GitHub")
            self._download_repo_files()
        general_docs, project_docs = self._load_documents()

        # Создание общего контекста
        if not general_docs:
            msg = "Отсутствуют общие документы"
            self._raise_error(msg)
        self.general_context = "\n\n".join(
            [doc.page_content for doc in general_docs]
        )
        # Создание списка поектов
        if not project_docs:
            msg = "Отсутствуют документы по проектам"
            self._raise_error(msg)
        self.project_map = {
            doc.metadata["source"]: doc.page_content
            for doc in project_docs
        }

        # Создание модели эмбеддингов
        if "embedding_model" not in self.config:
            msg = "Отсутствует название модели для эмбеддингов"
            self._raise_error(msg)
        self.embedding = HuggingFaceEmbeddings(
            model_name=self.config["embedding_model"])

        # Загрузка базы ChromaDB
        if os.path.exists(self.chroma_dir):
            self.vector_store = Chroma(
                persist_directory=self.chroma_dir,
                embedding_function=self.embedding
            )
            if self.vector_store._collection.count() > 0:
                is_new_store = False
            else:
                is_new_store = True
        else:
            is_new_store = True
        if is_new_store:
            logging.warning("База ChromaDB отсутствует или пустая. "
                            "Будет создана новая база из документов.")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.get("chunk_size", 1000),
                chunk_overlap=self.config.get("chunk_overlap", 200)
            )
            chunks = text_splitter.split_documents(project_docs)
            logging.info(f"Документы разделены на {len(chunks)} чанков. "
                         "Индексация и создание базы ChromaDB...")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding,
                persist_directory=self.chroma_dir
            )
            logging.info("Индексация и создание базы ChromaDB завершено.")
        logging.info("База эмбеддингов загружена, содержит "
                     f"{self.vector_store._collection.count()} записей.")
        # Получение ретривера
        self.retriever = self.vector_store.as_retriever(
            search_type=self.config.get("search_type", "similarity"),
            search_kwargs=self.config.get("search_kwargs", {"k": 15})
        )
        
        # Создание модели LLM
        if not os.environ.get("GOOGLE_API_KEY"):
            msg = "Отсутствует переменная окружения 'GOOGLE_API_KEY'"
            self._raise_error(msg)
        if "llm_model" not in self.config:
            msg = "Отсутствует название LLM модели"
            self._raise_error(msg)
        try:
            self.llm = ChatGoogleGenerativeAI(
                model=self.config["llm_model"]
            )
            logging.info(f"Модель '{self.llm.model}' загружена.")
        except Exception as e:
            msg = ("Возникла ошибка при загрузке модели "
                   f"'{self.llm.model}': {e}")
            self._raise_error(msg)

        # Загрузка конфига для лимитов по контексту
        self.context_len = self.config.get("context_len", 100000)
        if not isinstance(self.context_len, int):
            msg = "Значение 'context_len' должно быть целочисленным!"
            self._raise_error(msg)

        self.context_ratio = self.config.get("context_ratio", 2.5)
        if not isinstance(self.context_ratio, (int, float)):
            msg = "Значение 'context_ratio' должно быть численным!"
            self._raise_error(msg)

        # Проверка соответствия формата конфигурации промптов
        msg = ""
        prompts = ["history_prompt", "research_prompt", "answer_prompt"]
        configs = {
            "template": str, 
            "input_variables": list, 
            "output_parser": str
        }
        for prompt in prompts:
            if prompt not in self.config:
                msg += f"Отсутствует конфиг для промпта '{prompt}'. "
            elif not isinstance(self.config[prompt], dict):
                msg += (
                    f"Конфиг промпта '{prompt}' должен быть словарем, "
                    f"и содержать следующие типы данных {configs}"
                )
            else:
                current_prompt = self.config[prompt]
                for cfg_name, cfg_type in configs.items():
                    if cfg_name not in current_prompt:
                        msg += (
                            f"Отсутствует конфиг '{cfg_name}' для '{prompt}'. "
                        )
                    elif not isinstance(current_prompt[cfg_name], cfg_type):
                        msg += (
                            f"Значение '{cfg_name}' не является '{cfg_type}'. "
                        )
                    elif not current_prompt[cfg_name]:
                        msg += (
                            f"Значение '{current_prompt[cfg_name]}' пустое. ")
                    elif cfg_name == "output_parser":
                        parser_name = current_prompt[cfg_name]
                        try:
                            parser_class = getattr(parsers, parser_name)
                            if not issubclass(parser_class,
                                            parsers.BaseOutputParser):
                                msg += (
                                    f"Парсер '{parser_name} должен быть "
                                    f"дочерним классом 'BaseOutputParser'. ")
                        except Exception as e:
                            msg += (
                                f"\nОшибка при валидации парсера "
                                f"'{parser_name}': {e}\n")
        if msg:
            self._raise_error(msg)

        # Создание цепочек для промптов
        self.chains, self.inputs = {}, {}
        for prompt in prompts:
            current_prompt = self.config[prompt]
            prompt_template = PromptTemplate(
                template=current_prompt["template"],
                input_variables=current_prompt["input_variables"]
            )
            parser_class = getattr(parsers, current_prompt["output_parser"])
            ch_name = prompt.replace("prompt", "chain")
            self.chains[ch_name] = prompt_template | self.llm | parser_class()
            self.inputs[ch_name] = current_prompt["input_variables"]
        
        logging.info("Экземпляр класса PortfolioAssistant создан!")


    def _raise_error(self, msg: str):
        logging.error(msg)
        raise ValueError

    def _import_loader(self, loader_path: str) -> BaseLoader:
        """
        Импортирует библиотеки загрузчиков langchain по полному пути, 
        возвращает класс загрузчика производного от BaseLoader.
        """
        try:
            module_name, class_name = loader_path.rsplit(".", 1)
            module = importlib.import_module(module_name)
            loader_class = getattr(module, class_name)
            if not issubclass(loader_class, BaseLoader):
                msg = f"Класс {loader_class} не является потомком BaseLoader"
                logging.error(msg)
                raise TypeError(msg)
            return loader_class
        except Exception as e:
            logging.error(
                f"Ошибка при импорте загрузчика '{loader_path}': {e}")
            raise

    def _delete_dir(self, dir):
        if os.path.exists(dir):
            logging.warning(
                f"Директория '{dir}' существует, она будет удалена")
            shutil.rmtree(dir)        

    def _download_repo_files(self):
        """
        Получает список файлов из репозитория с помощью GitHub API, 
        загружает файлы в директорию 'data_dir', 
        определенную в файле конфигурации (по умолчанию data). 

        Внимание! При загрузке файлов репозитория удаляются всё содержимое 
        директорий 'data_dir' и 'chroma_dir'.
        """
        logging.info("Загрузка файлов из репозитория")
        owner = self.config.get("owner", "rzarubayev")
        repo = self.config.get("repo", "machine-learning-portfolio")
        branch = self.config.get("branch", "main")

        file_types = tuple(self.loaders.keys())

        api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"

        logging.info(f"Запрос структуры репозитория {api_url}")
        try:
            # Получение списка файлов
            response = requests.get(api_url)
            response.raise_for_status()
            data = response.json()
            if data.get("truncated"):
                logging.warning("Внимание: список файлов был усечен, "
                                "так как репозиторий слишком большой")
            file_list = data.get("tree", [])
        except requests.exceptions.RequestException as e:
            logging.error(f"Ошибка при запросе к GitHub API: {e}")
            return
        files = [
            f for f in file_list
            if f["type"] == "blob" and f["path"].endswith(file_types)
        ]
        if not files:
            logging.error(
                f"В репозитории не найдены файлы с раширениями {file_types}")
            return
        
        # Удаление директорий, если они существуют
        output_dir = self.data_dir
        self._delete_dir(output_dir)
        chroma_dir = self.chroma_dir
        self._delete_dir(chroma_dir)

        for f in files:
            # Получаем URL для доступа к самим файлам
            file_path = f["path"]
            raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{file_path}"
            # Создаем папки, если они отсутствуют
            local_file_path = os.path.join(output_dir, file_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            # Скачиваем файлы
            try:
                response = requests.get(raw_url)
                response.raise_for_status()
                with open(local_file_path, "w", encoding="utf8") as file:
                    file.write(response.text)
            except requests.exceptions.RequestException as e:
                logging.error(f"Ошибка при скачивании файла {file_path}: {e}")

    def _load_documents(self) -> tuple[list, list]:
        """
        Загружает все файлы посредством соответствующих загрузчиков langchain,
        используя словари для выбора нужного загрузчика и его параметров.
        """
    
        # Инициализируем переменные
        general_docs, project_docs, target_files = [], [], []
        supported_ext = tuple(self.loaders.keys())

        # Получим только известные файлы 
        for root, _, files in os.walk(self.data_dir):
            for f in files:
                if f.endswith(supported_ext):
                    target_files.append(os.path.join(root,f))

        if not target_files:
            logging.warning(f"В папке {self.data_dir} не найдены файлы "
                            f"с расширением {supported_ext}")
            return [], []

        # Обработка файлов из хранилища документов
        logging.info(
            f"Найдено {len(target_files)} файлов. Начинается обработка.")
        for file_path in target_files:
            try:
                # Получение загрузчика по расширению
                file_ext = os.path.splitext(file_path)[1]
                loader_cls = self.loaders[file_ext]

                if not loader_cls:
                    logging.warning("пропуск файла: не найден загрузчик "
                                    f"для расширения {file_ext}")
                    continue

                params = self.loader_params.get(loader_cls.__name__, {})
                loader = loader_cls(file_path, **params)
                docs = loader.load()

                # Объединение в один документ
                content = "\n\n".join([doc.page_content for doc in docs])

                doc = Document(page_content=content)

                # Добавление метаданных (источник, категория, проект)
                # в соответствии со структурой портфоли
                rel_path = os.path.relpath(file_path, self.data_dir)
                doc.metadata["source"] = rel_path

                parts = rel_path.split(os.sep)

                if len(parts) == 1:
                    # Файлы из корневой директории
                    general_docs.append(doc)
                else:
                    # Получаем категорию по названию директории
                    doc.metadata["category"] = parts[0]
                    if len(parts) == 2:
                        # Проект - имя файла, если нет субдиректорий
                        doc.metadata["project"] = os.path.splitext(parts[1])[0]
                    else:
                        # Если это субдиректория, проект - ее имя
                        doc.metadata["project"] = parts[1]
                    project_docs.append(doc)

            except Exception as e:
                logging.error(f"Ошибка при обработке файла {file_path}: {e}")
        logging.info("Загрузка документов завершена. "
                     f"Загружено {len(general_docs)} общих документов "
                     f"и {len(project_docs)} документов по проектам")
        return general_docs, project_docs

    def _limit_context(self, file_path_list: list, used_tokens = 0) -> str:
        """
        Объединяет документы с ограничением на длину контекста. 
        Использует context_len для ограничения количества токенов. 
        Для оценки длины токенов используется context_ratio, 
        которая определяет среднее количество символов на токен.

        Эсли документов нет, возвращает строку 'Ничего не найдено'.

        Если документы есть, возвращает объедиенный контекст, который 
        не выходит за пределы ограничений по токенам.
        """
        if len(file_path_list) == 0:
            return "Ничего не найдено"
        result = []
        total_tokens = used_tokens
        for file_path in file_path_list:
            if file_path in self.project_map:
                tokens = len(self.project_map[file_path]) / self.context_ratio
                if total_tokens + tokens < self.context_len:
                    result.append(self.project_map[file_path])
                    total_tokens += tokens
                else:
                    break
            else:
                logging.warning(f"Файл {file_path} не найден в документах")

        return "\n\n".join(result)

    def run_chain(self, chain: str, prompt_inputs: dict) -> str | list | dict:
        """
        Функция для запуска одной цепочки. 
        Принимает название цепочки и словарь с параметрами.
        """
        try:
            params = {key: prompt_inputs[key] for key in self.inputs[chain]}
            return self.chains[chain].invoke(params)
        except Exception as e:
            logging.error(
                f"Ошибка вызова функции 'run_chain' для цепочки '{chain}: "
                f"{e}\n\n Входные данные для промпта:\n {prompt_inputs}")


    def query_portfolio(self, message, history) -> str:
        """
        Функция для запуска всех цепочек AI-ассистента, 
        принимает новое сообщение и историю чата.

        Возвращает ответ AI-ассистента
        """
        logging.info("Поступило новое сообщение.")

        # Переменная для сборки всех параметров для промпта
        inputs = {}
        
        # Преобразование истории в текст
        inputs["chat_history"] = "\n".join([
            f"**{msg['role']}:** {msg['content']}"
            for msg in history])
        # Добавление запроса пользователя
        inputs["question"] = message

        # Получение уточненного вопроса из истории чата
        inputs["question"] = self.run_chain("history_chain", inputs)

        logging.info("Получен ответ от первой цепочки.")
        
        # Добавление основного контекста, фрагментов и списка файлов
        inputs["general_context"] = self.general_context

        retrieved_chunks = self.retriever.invoke(inputs["question"])
        inputs["retrieved_chunks"] = "\n\n".join(
            [doc.page_content for doc in retrieved_chunks]
        )

        inputs["all_files"] = "\n".join(self.project_map.keys())

        # Получение списка релевантных файлов
        files_to_read = self.run_chain("research_chain", inputs)
        logging.info(f"Получен список из {len(files_to_read)} файлов "
                     "от второй цепочки.")

        used_tokens = (
            len(inputs["question"]) +
            len(inputs["general_context"])) / self.context_ratio

        inputs["specific_content"] = self._limit_context(
            file_path_list=files_to_read,
            used_tokens=used_tokens
        )

        answer = self.run_chain("answer_chain", inputs)
        logging.info(f"Получен окончательный ответ")
        return answer

if __name__ == "__main__":

    assistant = PortfolioAssistant()

    with gr.Blocks(title="AI-ассистент по GitHub-портфолио Зарубаева Руслана",
                theme=gr.themes.Default()) as demo:
        gr.Markdown(
            """
            # 🤖 AI-ассистент по GitHub-портфолио Зарубаева Руслана
            Задайте мне вопрос о проектах, использованных технологиях или опыте Руслана.
            """
        )

        gr.ChatInterface(
            fn=assistant.query_portfolio,
            type="messages",
            chatbot=gr.Chatbot(type="messages"),
            examples=[
                "В каких проектах использовался LightGBM?",
                "Какие инструменты использовались для деплоя модели стоимости квартир?",
                "Опиши задачу из проекта по анализу токсичности комментариев."
            ],
            textbox=gr.Textbox(
                placeholder="Задайте Ваш вопрос о проектах...",
                container=True, scale=7
            )
        )

    demo.launch()