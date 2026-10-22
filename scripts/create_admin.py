"""
Создаёт администратора (или повышает существующего пользователя до администратора).

Запуск:
    python -m scripts.create_admin admin
    python -m scripts.create_admin admin --password 'S3cret!'
"""
from __future__ import annotations

import argparse
import getpass

from passlib.context import CryptContext
from sqlalchemy import select

from db import Base, SessionLocal, engine
from models import User

# Та же схема хеширования, что и в main.py
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")


def main() -> None:
    parser = argparse.ArgumentParser(description="Создать/повысить администратора")
    parser.add_argument("username")
    parser.add_argument("--password", help="если не указан — будет запрошен интерактивно")
    args = parser.parse_args()

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        user = db.execute(select(User).where(User.username == args.username)).scalar_one_or_none()
        if user is None:
            password = args.password or getpass.getpass("Пароль: ")
            if len(password) < 6:
                raise SystemExit("Пароль должен быть не короче 6 символов")
            user = User(
                username=args.username,
                hashed_password=pwd_context.hash(password),
                is_active=True,
                is_superuser=True,
            )
            db.add(user)
            action = "создан"
        else:
            user.is_superuser = True
            user.is_active = True
            if args.password:
                user.hashed_password = pwd_context.hash(args.password)
            action = "повышен до администратора"
        db.commit()
        print(f"Пользователь '{args.username}' {action}.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
