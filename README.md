# Chat with PDFs

Этот код предоставляет веб-приложение Streamlit, которое позволяет пользователям общаться с PDF-файлами. Приложение использует модели чатов от OpenAI для генерации ответов на основе запросов пользователей.


## Использование

Для запуска кода выполните следующую команду:

```
streamlit run '.\AI Reader.py'
```

После запуска приложения вы можете получить к нему доступ через веб-браузер по адресу `http://localhost:8501`.

## Функциональность

Веб-приложение предоставляет следующую функциональность:

- Аутентификация: пользователи могут создать учетную запись и войти в приложение.
- Главная страница: пользователи могут загружать PDF-файлы и общаться с ними с использованием моделей чатов от OpenAI.
- Сохраненные диалоги: пользователи могут просматривать и загружать ранее сохраненные диалоги.
- Сохранение диалогов: пользователи могут сохранять свою историю чата в виде диалогов.

## Структура

Код имеет следующую структуру:

- Функции создания базы данных и таблиц
- Функции аутентификации пользователей
- Функции сохранения и загрузки диалогов
- Функции создания модели чата и контекста
- Функции пользовательского интерфейса Streamlit
- Основная функция для запуска приложения

## Примечания

- Код использует переменные окружения для хранения конфиденциальной информации, такой как ключ API OpenAI. Убедитесь, что установили эти переменные окружения перед запуском кода.