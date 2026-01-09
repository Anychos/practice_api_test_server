# Инструкция по запуску

## Предусловие

Создать базу данных PostgreSQL

## Для Windows:

### 1. Клонировать репозиторий
git clone https://github.com/Anychos/practice_api_test_server.git  
cd practice_api_test_server  

### 2. Создать виртуальное окружение
python -m venv venv

### 3. Активировать виртуальное окружение
venv\Scripts\activate

### 4. Установить зависимости
pip install -r requirements.txt

### 5. Запустить сервер
python app.py

## Для Linux/Mac:

### 1. Клонировать репозиторий (если нужно)
git clone https://github.com/Anychos/practice_api_test_server.git  
cd practice_api_test_server 

### 2. Создать виртуальное окружение
python3 -m venv venv

### 3. Активировать виртуальное окружение
source venv/bin/activate

### 4. Установить зависимости
pip install -r requirements.txt

### 5. Запустить сервер
python app.py

## Настройка файла .env

    Переименуйте файл .env.example в .env в корне проекта

    Измените настройки в файле .env на свои

    Для PostgreSQL:

    DATABASE_URL=postgresql://your_username:your_password@your_host/your_database_name
    SECRET_KEY=your_secret_key

### Сервер запущен и доступен по адресу: http://localhost:8080

Документация Swagger http://localhost:8080/docs   
Документация ReDoc   http://localhost:8080/redoc  
Проверка состояния сервера http://localhost:8080/health  
