import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

try:
    # Подключаемся к PostgreSQL (к базе postgres)
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password=os.getenv("DB_PASSWORD", ""),
        host="localhost",
        port="5432"
    )
    
    # Устанавливаем режим автокоммита для создания БД
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    
    # Создаем курсор
    cur = conn.cursor()
    
    # Создаем базу данных medialux
    cur.execute("CREATE DATABASE medialux;")
    
    cur.close()
    conn.close()
    
    print("✅ База данных 'medialux' успешно создана!")
    
except psycopg2.errors.DuplicateDatabase:
    print("⚠️  База данных 'medialux' уже существует!")
except Exception as e:
    print(f"❌ Ошибка: {e}")
    print("\nВозможные причины:")
    print("1. PostgreSQL не запущен")
    print("2. Неверный пароль (попробуйте изменить в коде)")
    print("3. Не установлен psycopg2 (выполните: py -m pip install psycopg2-binary)")
    