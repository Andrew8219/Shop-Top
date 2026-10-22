"""
generate_fake_data.py — генерация тестовых товаров для ВКР (добавление в БД, не удаляет существующие).

Запуск:
    python -m scripts.generate_fake_data           # по умолчанию 1000 новых офферов
    python -m scripts.generate_fake_data 9000      # добавить 9000 офферов (к уже имеющимся)

Генерирует реалистичные товары по 12 категориям с разнообразными
характеристиками — достаточно для тестирования семантического поиска.
После загрузки пересоберите индекс: POST /admin/search-index/build
"""

import random
import re
import sys
from sqlalchemy.orm import Session
from sqlalchemy import func

from db import SessionLocal, engine, Base
from models import Product, ProductOffer, Company, Category, PropertyName, ProductProperty

Base.metadata.create_all(bind=engine)

random.seed(42)

# =============================================================================
#   СПРАВОЧНИКИ
# =============================================================================

LAPTOPS = {
    "category": "Ноутбуки",
    "companies": ["Apple", "Lenovo", "ASUS", "HP", "Dell", "Acer", "MSI", "Huawei", "Samsung", "LG"],
    "templates": [
        ("MacBook Pro {year}", "MacBook Pro"),
        ("ThinkPad X{num}", "ThinkPad"),
        ("VivoBook {num}", "ASUS VivoBook"),
        ("Pavilion {num}", "HP Pavilion"),
        ("Inspiron {num}", "Dell Inspiron"),
        ("Aspire {num}", "Acer Aspire"),
        ("Stealth {num}", "MSI Stealth"),
        ("MateBook D{num}", "Huawei MateBook"),
        ("Galaxy Book{num}", "Samsung Galaxy Book"),
        ("Gram {num}", "LG Gram"),
    ],
    "properties": lambda: {
        "Процессор": random.choice([
            "Intel Core i3-1215U", "Intel Core i5-1235U", "Intel Core i7-1255U",
            "Intel Core i9-12900H", "AMD Ryzen 5 5600U", "AMD Ryzen 7 6800H",
            "Apple M1", "Apple M2", "Apple M2 Pro", "AMD Ryzen 9 6900HX",
        ]),
        "Оперативная память": random.choice(["8 ГБ", "16 ГБ", "32 ГБ", "64 ГБ"]),
        "Накопитель": random.choice(["256 ГБ SSD", "512 ГБ SSD", "1 ТБ SSD", "2 ТБ SSD"]),
        "Диагональ экрана": random.choice(["13.3 дюйм", "13.6 дюйм", "14 дюйм", "15.6 дюйм", "16 дюйм", "17.3 дюйм"]),
        "Разрешение": random.choice(["1920x1080", "2560x1600", "3840x2160", "2560x1440"]),
        "Видеокарта": random.choice([
            "Intel Iris Xe", "AMD Radeon 680M", "NVIDIA GeForce RTX 3050",
            "NVIDIA GeForce RTX 3060", "NVIDIA GeForce RTX 4070", "Apple M2 GPU",
        ]),
        "Аккумулятор": random.choice(["45 мАч", "54 мАч", "72 мАч", "99 мАч"]),
        "Вес": random.choice(["1.2 кг", "1.4 кг", "1.6 кг", "1.8 кг", "2.1 кг", "2.4 кг"]),
        "ОС": random.choice(["Windows 11 Home", "Windows 11 Pro", "macOS Ventura", "Без ОС"]),
        "Тип": "Ноутбук",
    },
}

SMARTPHONES = {
    "category": "Смартфоны",
    "companies": ["Apple", "Samsung", "Xiaomi", "OPPO", "Realme", "Vivo", "OnePlus", "Google", "Sony", "Motorola"],
    "templates": [
        ("iPhone {num} Pro Max", "iPhone"),
        ("Galaxy S{num} Ultra", "Samsung Galaxy S"),
        ("Redmi Note {num} Pro", "Redmi Note"),
        ("OPPO Find X{num}", "OPPO Find"),
        ("Realme {num} Pro", "Realme"),
        ("Vivo V{num}", "Vivo V"),
        ("OnePlus {num}T", "OnePlus"),
        ("Pixel {num}a", "Google Pixel"),
        ("Xperia {num} V", "Sony Xperia"),
        ("Edge {num}", "Motorola Edge"),
    ],
    "properties": lambda: {
        "Диагональ экрана": random.choice(["6.1 дюйм", "6.4 дюйм", "6.6 дюйм", "6.7 дюйм", "6.8 дюйм"]),
        "Разрешение экрана": random.choice(["1080x2400", "1440x3200", "1080x2340"]),
        "Процессор": random.choice([
            "Apple A16 Bionic", "Snapdragon 8 Gen 2", "Snapdragon 778G",
            "Dimensity 9200", "Exynos 2200", "Google Tensor G2",
        ]),
        "Оперативная память": random.choice(["6 ГБ", "8 ГБ", "12 ГБ", "16 ГБ"]),
        "Встроенная память": random.choice(["128 ГБ", "256 ГБ", "512 ГБ"]),
        "Основная камера": random.choice(["12 МП", "48 МП", "50 МП", "64 МП", "108 МП", "200 МП"]),
        "Фронтальная камера": random.choice(["8 МП", "12 МП", "16 МП", "32 МП"]),
        "Аккумулятор": random.choice(["3500 мАч", "4500 мАч", "5000 мАч", "5500 мАч"]),
        "5G": random.choice(["Да", "Нет"]),
        "NFC": random.choice(["Да", "Нет"]),
        "Цвет": random.choice(["Чёрный", "Белый", "Синий", "Зелёный", "Золотой", "Красный", "Серый"]),
    },
}

TVS = {
    "category": "Телевизоры",
    "companies": ["LG", "Samsung", "Sony", "Philips", "Hisense", "TCL", "Xiaomi", "Haier", "Thomson", "BBK"],
    "templates": [
        ("OLED{num}C3", "LG OLED"),
        ("QLED Q{num}B", "Samsung QLED"),
        ("Bravia XR {num}K", "Sony Bravia"),
        ("Ambilight {num}", "Philips Ambilight"),
        ("ULED U{num}H", "Hisense ULED"),
        ("C{num}35", "TCL"),
        ("A{num}Pro", "Xiaomi A Pro"),
        ("55 Smart", "Haier Smart"),
        ("Thomson {num}", "Thomson"),
        ("BBK {num}LEX", "BBK"),
    ],
    "properties": lambda: {
        "Диагональ": random.choice(["43 дюйм", "50 дюйм", "55 дюйм", "65 дюйм", "75 дюйм", "85 дюйм"]),
        "Разрешение": random.choice(["Full HD 1920x1080", "4K UHD 3840x2160", "8K 7680x4320"]),
        "Тип матрицы": random.choice(["OLED", "QLED", "IPS", "VA", "Mini-LED"]),
        "Частота обновления": random.choice(["60 Гц", "100 Гц", "120 Гц", "144 Гц"]),
        "Smart TV": random.choice(["webOS 23", "Tizen 7", "Android TV 11", "Google TV", "Без Smart TV"]),
        "HDR": random.choice(["HDR10", "HDR10+", "Dolby Vision", "HLG"]),
        "Звук": random.choice(["20 Вт", "40 Вт", "60 Вт", "80 Вт"]),
        "HDMI": random.choice(["2", "3", "4"]),
        "Год выпуска": random.choice(["2022", "2023", "2024"]),
    },
}

MONITORS = {
    "category": "Мониторы",
    "companies": ["LG", "Samsung", "Dell", "ASUS", "Acer", "BenQ", "HP", "Lenovo", "MSI", "Philips"],
    "templates": [
        ("UltraGear {num}GP", "LG UltraGear"),
        ("Odyssey G{num}", "Samsung Odyssey"),
        ("U{num}20Q", "Dell UltraSharp"),
        ("ProArt PA{num}", "ASUS ProArt"),
        ("Nitro VG{num}", "Acer Nitro"),
        ("PD{num}U", "BenQ PD"),
        ("X{num}4", "HP X"),
        ("ThinkVision T{num}", "Lenovo ThinkVision"),
        ("MAG {num}Q", "MSI MAG"),
        ("E-Line {num}", "Philips E"),
    ],
    "properties": lambda: {
        "Диагональ": random.choice(["24 дюйм", "27 дюйм", "32 дюйм", "34 дюйм", "49 дюйм"]),
        "Разрешение": random.choice(["1920x1080", "2560x1440", "3840x2160", "3440x1440"]),
        "Матрица": random.choice(["IPS", "VA", "OLED", "TN"]),
        "Частота": random.choice(["60 Гц", "75 Гц", "144 Гц", "165 Гц", "240 Гц"]),
        "HDR": random.choice(["HDR10", "DisplayHDR 400", "DisplayHDR 600", "Нет"]),
        "Крепление VESA": random.choice(["100x100", "75x75", "Нет"]),
        "HDMI / DP": random.choice(["1xHDMI 2xDP", "2xHDMI 1xDP", "1xHDMI 1xUSB-C"]),
        "Изогнутый экран": random.choice(["Да", "Нет"]),
    },
}

DESKTOP_PCS = {
    "category": "Компьютеры",
    "companies": ["HP", "Dell", "Lenovo", "ASUS", "Acer", "Apple", "MSI", "CyberPower", "iRU", "Cactus"],
    "templates": [
        ("Pavilion Desktop {num}", "HP Pavilion"),
        ("OptiPlex {num}", "Dell OptiPlex"),
        ("ThinkCentre M{num}", "Lenovo ThinkCentre"),
        ("ROG Strix G{num}", "ASUS ROG"),
        ("Aspire TC-{num}", "Acer Aspire"),
        ("Mac Studio M{num}", "Mac Studio"),
        ("MAG Infinite {num}", "MSI Infinite"),
        ("Gamer Xtreme {num}", "CyberPower"),
        ("Power {num}", "iRU Power"),
        ("Green {num}", "Cactus Green"),
    ],
    "properties": lambda: {
        "Процессор": random.choice([
            "Intel Core i5-13400", "Intel Core i7-13700", "AMD Ryzen 5 7600",
            "AMD Ryzen 7 7700X", "Apple M2 Max", "Intel Core i9-13900",
        ]),
        "ОЗУ": random.choice(["16 ГБ", "32 ГБ", "64 ГБ"]),
        "SSD": random.choice(["512 ГБ", "1 ТБ", "2 ТБ"]),
        "Видеокарта": random.choice([
            "Intel UHD 770", "NVIDIA RTX 4060", "NVIDIA RTX 4070",
            "AMD RX 7600", "Интегрированная", "NVIDIA RTX 4080",
        ]),
        "Форм-фактор": random.choice(["Mini Tower", "Mid Tower", "Full Tower", "SFF", "Моноблок"]),
        "ОС": random.choice(["Windows 11 Home", "Windows 11 Pro", "Без ОС", "macOS"]),
        "Wi-Fi": random.choice(["Wi-Fi 6", "Wi-Fi 6E", "Нет"]),
    },
}

HEADPHONES = {
    "category": "Наушники",
    "companies": ["Sony", "Bose", "Apple", "Samsung", "Sennheiser", "Jabra", "Beats", "JBL", "Plantronics", "Bang & Olufsen"],
    "templates": [
        ("WH-{num}XM5", "Sony WH"),
        ("QuietComfort {num}", "Bose QC"),
        ("AirPods Pro {num}", "Apple AirPods"),
        ("Galaxy Buds {num}", "Samsung Galaxy Buds"),
        ("HD {num}", "Sennheiser HD"),
        ("Evolve {num}", "Jabra Evolve"),
        ("Studio Pro {num}", "Beats Studio"),
        ("Live {num}", "JBL Live"),
        ("Voyager {num}", "Plantronics Voyager"),
        ("Beoplay H{num}", "B&O Beoplay"),
    ],
    "properties": lambda: {
        "Тип": random.choice(["Полноразмерные", "Вкладыши", "Накладные", "Внутриканальные"]),
        "Подключение": random.choice(["Bluetooth 5.0", "Bluetooth 5.2", "Bluetooth 5.3", "Проводные 3.5 мм"]),
        "Шумоподавление": random.choice(["Активное (ANC)", "Пассивное", "Нет"]),
        "Время работы": random.choice(["8 часов", "15 часов", "20 часов", "30 часов", "40 часов"]),
        "Кейс": random.choice(["Да", "Нет"]),
        "Микрофон": random.choice(["Да", "Нет"]),
        "Водозащита": random.choice(["IPX4", "IPX5", "IPX7", "Нет"]),
        "Цвет": random.choice(["Чёрный", "Белый", "Серебристый", "Бежевый", "Синий"]),
        "Вес": random.choice(["150 г", "200 г", "250 г", "300 г", "5 г", "7 г"]),
    },
}

REFRIGERATORS = {
    "category": "Холодильники",
    "companies": ["Samsung", "LG", "Bosch", "Haier", "Indesit", "Hotpoint", "Atlant", "Beko", "Gorenje", "Liebherr"],
    "templates": [
        ("Side-by-Side {num}", "Samsung Side"),
        ("French Door {num}", "LG French Door"),
        ("KGN{num}WI", "Bosch KGN"),
        ("HTF-{num}", "Haier HTF"),
        ("HD {num}", "Indesit HD"),
        ("HF {num}", "Hotpoint HF"),
        ("ХМ-{num}", "Atlant ХМ"),
        ("RCNA{num}", "Beko RCNA"),
        ("NRK{num}", "Gorenje NRK"),
        ("CBNes {num}", "Liebherr"),
    ],
    "properties": lambda: {
        "Тип": random.choice(["Двухкамерный", "Однокамерный", "Side-by-Side", "French Door", "Многодверный"]),
        "Общий объём": random.choice(["200 л", "270 л", "330 л", "400 л", "450 л", "550 л", "650 л"]),
        "Объём морозильной камеры": random.choice(["50 л", "80 л", "100 л", "120 л", "150 л"]),
        "No Frost": random.choice(["Да", "Нет"]),
        "Класс энергопотребления": random.choice(["A", "A+", "A++", "A+++"]),
        "Высота": random.choice(["150 см", "170 см", "185 см", "190 см", "200 см"]),
        "Ширина": random.choice(["54 см", "60 см", "70 см", "90 см"]),
        "Цвет": random.choice(["Серебристый", "Белый", "Чёрный", "Нержавеющая сталь", "Серый"]),
        "Инверторный компрессор": random.choice(["Да", "Нет"]),
        "Уровень шума": random.choice(["35 дБ", "38 дБ", "40 дБ", "42 дБ"]),
    },
}

WASHING_MACHINES = {
    "category": "Стиральные машины",
    "companies": ["Bosch", "Samsung", "LG", "Haier", "Indesit", "Hotpoint", "Atlant", "Beko", "Candy", "Siemens"],
    "templates": [
        ("Serie {num}", "Bosch Serie"),
        ("WW{num}T", "Samsung WW"),
        ("F{num}V", "LG F"),
        ("HW{num}", "Haier HW"),
        ("BWSE {num}", "Indesit BWSE"),
        ("RSF {num}", "Hotpoint RSF"),
        ("60У{num}", "Atlant"),
        ("BWFX{num}", "Beko BWFX"),
        ("CS{num}", "Candy CS"),
        ("WM{num}", "Siemens WM"),
    ],
    "properties": lambda: {
        "Тип загрузки": random.choice(["Фронтальная", "Вертикальная"]),
        "Максимальная загрузка": random.choice(["5 кг", "6 кг", "7 кг", "8 кг", "9 кг", "10 кг", "12 кг"]),
        "Скорость отжима": random.choice(["800 об/мин", "1000 об/мин", "1200 об/мин", "1400 об/мин", "1600 об/мин"]),
        "Класс стирки": random.choice(["A", "A+", "A++"]),
        "Класс энергопотребления": random.choice(["A", "A+", "A++", "A+++"]),
        "Пар": random.choice(["Да", "Нет"]),
        "Wi-Fi": random.choice(["Да", "Нет"]),
        "Дисплей": random.choice(["LED", "TFT", "Нет"]),
        "Защита от протечек": random.choice(["Полная", "Частичная", "Нет"]),
        "Уровень шума": random.choice(["48 дБ", "52 дБ", "55 дБ", "58 дБ"]),
    },
}

CAMERAS = {
    "category": "Фотоаппараты",
    "companies": ["Canon", "Nikon", "Sony", "Fujifilm", "Olympus", "Panasonic", "Leica", "Pentax", "Sigma", "Hasselblad"],
    "templates": [
        ("EOS R{num}", "Canon EOS R"),
        ("Z{num}", "Nikon Z"),
        ("Alpha A{num}", "Sony Alpha"),
        ("X-T{num}", "Fujifilm X-T"),
        ("OM-{num}", "Olympus OM"),
        ("Lumix S{num}", "Panasonic Lumix"),
        ("Q{num}", "Leica Q"),
        ("K-{num}", "Pentax K"),
        ("fp {num}", "Sigma fp"),
        ("X{num}D", "Hasselblad X"),
    ],
    "properties": lambda: {
        "Тип": random.choice(["Беззеркальная", "Зеркальная", "Компактная", "Среднеформатная"]),
        "Разрешение матрицы": random.choice(["12 МП", "20 МП", "24 МП", "33 МП", "45 МП", "61 МП", "100 МП"]),
        "Размер матрицы": random.choice(["1/2.3\"", "1\"", "APS-C", "Full Frame", "Medium Format"]),
        "Видео": random.choice(["4K 30fps", "4K 60fps", "4K 120fps", "8K 30fps", "6K RAW"]),
        "Стабилизация": random.choice(["Оптическая", "Матричная (IBIS)", "Нет"]),
        "Экран": random.choice(["Поворотный", "Наклонный", "Фиксированный"]),
        "Байонет": random.choice(["RF", "Z", "E", "X", "MFT", "L", "M", "K", "SA", "XCD"]),
        "Автофокус": random.choice(["Фазовый", "Контрастный", "Гибридный"]),
        "Аккумулятор (кол-во кадров)": random.choice(["250", "400", "500", "700", "900"]),
        "Водозащита": random.choice(["Да", "Нет"]),
    },
}

TABLETS = {
    "category": "Планшеты",
    "companies": ["Apple", "Samsung", "Xiaomi", "Huawei", "Lenovo", "Microsoft", "Amazon", "OPPO", "Realme", "TCL"],
    "templates": [
        ("iPad Pro {num}", "Apple iPad Pro"),
        ("Galaxy Tab S{num}", "Samsung Galaxy Tab"),
        ("Pad {num} Pro", "Xiaomi Pad"),
        ("MatePad Pro {num}", "Huawei MatePad"),
        ("Tab P{num} Pro", "Lenovo Tab"),
        ("Surface Pro {num}", "Microsoft Surface Pro"),
        ("Fire HD {num}", "Amazon Fire"),
        ("Pad Air {num}", "OPPO Pad"),
        ("Pad {num}", "Realme Pad"),
        ("NXTPAPER {num}", "TCL NXTPAPER"),
    ],
    "properties": lambda: {
        "Диагональ": random.choice(["8.7 дюйм", "10.4 дюйм", "10.9 дюйм", "11 дюйм", "12.4 дюйм", "12.9 дюйм", "13 дюйм"]),
        "Разрешение": random.choice(["1920x1200", "2000x1200", "2560x1600", "2732x2048", "2800x1752"]),
        "Процессор": random.choice([
            "Apple M2", "Snapdragon 8 Gen 2", "Dimensity 9000",
            "Kirin 9000s", "Snapdragon 870", "Intel Core i5", "MediaTek Helio G99",
        ]),
        "Оперативная память": random.choice(["4 ГБ", "6 ГБ", "8 ГБ", "12 ГБ", "16 ГБ"]),
        "Встроенная память": random.choice(["64 ГБ", "128 ГБ", "256 ГБ", "512 ГБ", "1 ТБ"]),
        "SIM-карта": random.choice(["Да (LTE)", "Да (5G)", "Нет (только Wi-Fi)"]),
        "Аккумулятор": random.choice(["6000 мАч", "7040 мАч", "8000 мАч", "10090 мАч", "11900 мАч"]),
        "ОС": random.choice(["iPadOS 17", "Android 13", "Android 12", "Windows 11"]),
        "Стилус в комплекте": random.choice(["Да", "Нет"]),
    },
}

SMART_WATCHES = {
    "category": "Умные часы",
    "companies": ["Apple", "Samsung", "Garmin", "Fitbit", "Huawei", "Amazfit", "Xiaomi", "OPPO", "Fossil", "Withings"],
    "templates": [
        ("Apple Watch Series {num}", "Apple Watch"),
        ("Galaxy Watch {num}", "Samsung Galaxy Watch"),
        ("Fenix {num}", "Garmin Fenix"),
        ("Versa {num}", "Fitbit Versa"),
        ("Watch GT {num}", "Huawei Watch GT"),
        ("GTR {num} Pro", "Amazfit GTR"),
        ("Mi Watch S{num}", "Xiaomi Mi Watch"),
        ("Watch X{num}", "OPPO Watch"),
        ("Gen {num}", "Fossil Gen"),
        ("ScanWatch {num}", "Withings ScanWatch"),
    ],
    "properties": lambda: {
        "Диаметр корпуса": random.choice(["40 мм", "41 мм", "44 мм", "45 мм", "47 мм", "49 мм"]),
        "Дисплей": random.choice(["AMOLED", "OLED", "LCD", "Transflective LCD"]),
        "GPS": random.choice(["Да", "Нет"]),
        "Пульсометр": random.choice(["Да", "Нет"]),
        "ЭКГ": random.choice(["Да", "Нет"]),
        "Время работы": random.choice(["1 день", "2 дня", "5 дней", "7 дней", "14 дней", "21 день"]),
        "Водозащита": random.choice(["5 ATM", "10 ATM", "IP68", "100 м"]),
        "NFC (оплата)": random.choice(["Да", "Нет"]),
        "Совместимость": random.choice(["iOS", "Android", "iOS и Android"]),
        "Материал корпуса": random.choice(["Алюминий", "Нержавеющая сталь", "Титан", "Пластик"]),
    },
}

VACUUM_CLEANERS = {
    "category": "Пылесосы",
    "companies": ["Dyson", "Samsung", "Xiaomi", "Philips", "Bosch", "Miele", "iRobot", "Roborock", "LG", "Karcher"],
    "templates": [
        ("V{num} Absolute", "Dyson V"),
        ("Bespoke Jet {num}", "Samsung Bespoke"),
        ("Mi Vacuum {num}", "Xiaomi Mi"),
        ("SpeedPro {num}", "Philips SpeedPro"),
        ("Unlimited {num}", "Bosch Unlimited"),
        ("Boost CX{num}", "Miele Boost"),
        ("Roomba j{num}", "iRobot Roomba"),
        ("S{num} MaxV", "Roborock S"),
        ("CordZero {num}", "LG CordZero"),
        ("WD {num}", "Karcher WD"),
    ],
    "properties": lambda: {
        "Тип": random.choice(["Вертикальный беспроводной", "Робот-пылесос", "Контейнерный", "Мешочный", "Моющий"]),
        "Мощность всасывания": random.choice(["60 Вт", "100 Вт", "150 Вт", "185 Вт", "230 Вт"]),
        "Ёмкость контейнера": random.choice(["0.3 л", "0.5 л", "0.7 л", "1.0 л", "2.5 л"]),
        "Время работы": random.choice(["20 мин", "40 мин", "60 мин", "90 мин", "120 мин"]),
        "Уровень шума": random.choice(["58 дБ", "65 дБ", "70 дБ", "75 дБ"]),
        "HEPA-фильтр": random.choice(["Да", "Нет"]),
        "Влажная уборка": random.choice(["Да", "Нет"]),
        "Wi-Fi / Приложение": random.choice(["Да", "Нет"]),
        "Вес": random.choice(["1.5 кг", "2.0 кг", "2.8 кг", "3.5 кг", "4.5 кг", "5.0 кг"]),
        "Цвет": random.choice(["Чёрный", "Золотой", "Синий", "Серый", "Белый", "Фиолетовый"]),
    },
}

ALL_CATEGORIES = [
    LAPTOPS, SMARTPHONES, TVS, MONITORS, DESKTOP_PCS, HEADPHONES, REFRIGERATORS,
    WASHING_MACHINES, CAMERAS, TABLETS, SMART_WATCHES, VACUUM_CLEANERS,
]

# =============================================================================
#   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =============================================================================

def get_or_create(session: Session, model, **kwargs):
    obj = session.query(model).filter_by(**kwargs).first()
    if not obj:
        obj = model(**kwargs)
        session.add(obj)
        session.flush()
    return obj


def get_next_id(session: Session, model, id_field):
    max_id = session.query(func.max(getattr(model, id_field))).scalar() or 0
    return max_id + 1


# =============================================================================
#   ГЕНЕРАТОР ТОВАРОВ
# =============================================================================

# Реалистичные ценовые диапазоны (₽) по категориям
PRICE_RANGES = {
    "Ноутбуки": (35_000, 250_000),
    "Смартфоны": (12_000, 160_000),
    "Телевизоры": (25_000, 300_000),
    "Мониторы": (12_000, 120_000),
    "Компьютеры": (40_000, 350_000),
    "Наушники": (2_500, 60_000),
    "Холодильники": (30_000, 250_000),
    "Стиральные машины": (25_000, 150_000),
    "Фотоаппараты": (45_000, 600_000),
    "Планшеты": (12_000, 180_000),
    "Умные часы": (5_000, 90_000),
    "Пылесосы": (8_000, 90_000),
}

_APPLE_RE = re.compile(r"\bApple\b|macOS|iPadOS")


def make_consistent_properties(cat_data: dict, company_name: str) -> dict:
    """
    Генерирует характеристики, согласованные с брендом:
    у Apple — Apple Silicon / macOS / iOS, у остальных — без «яблочных» значений.
    """
    cat_name = cat_data["category"]
    props = cat_data["properties"]()
    for _ in range(200):
        has_apple = any(
            _APPLE_RE.search(str(props.get(k, ""))) for k in ("Процессор", "Видеокарта", "ОС")
        )
        if has_apple == (company_name == "Apple") or company_name == "Apple":
            break
        props = cat_data["properties"]()

    if company_name == "Apple":
        if cat_name == "Ноутбуки":
            props["Процессор"] = random.choice(["Apple M1", "Apple M2", "Apple M2 Pro"])
            props["Видеокарта"] = "Встроенная (Apple Silicon)"
            props["ОС"] = "macOS"
        elif cat_name == "Компьютеры":
            props["Процессор"] = random.choice(["Apple M2 Max", "Apple M2 Ultra"])
            props["Видеокарта"] = "Встроенная (Apple Silicon)"
            props["ОС"] = "macOS"
        elif cat_name == "Смартфоны":
            props["Процессор"] = "Apple A16 Bionic"
        elif cat_name == "Планшеты":
            props["ОС"] = "iPadOS 17"
        elif cat_name == "Умные часы":
            props["Совместимость"] = "iOS"
    elif cat_name == "Планшеты" and props.get("ОС") == "iPadOS 17":
        props["ОС"] = "Android 13"

    # У ноутбуков ёмкость батареи указывается в Вт·ч
    if cat_name == "Ноутбуки" and "Аккумулятор" in props:
        props["Аккумулятор"] = str(props["Аккумулятор"]).replace("мАч", "Вт·ч")
    return props


def generate_products(session: Session, total: int = 1000) -> int:
    per_category = total // len(ALL_CATEGORIES)
    extra = total - per_category * len(ALL_CATEGORIES)

    imported = 0

    for cat_idx, cat_data in enumerate(ALL_CATEGORIES):
        count = per_category + (1 if cat_idx < extra else 0)
        cat_name = cat_data["category"]
        companies = cat_data["companies"]
        templates = cat_data["templates"]

        print(f"\n📦 Категория: {cat_name} ({count} товаров)")

        category = get_or_create(session, Category,
                                  name=cat_name,
                                  category_id=get_next_id(session, Category, "category_id"))

        session.flush()

        for i in range(count):
            # Шаблон модели и бренд идут парами (MacBook — Apple, ThinkPad — Lenovo, …)
            idx = random.randrange(len(templates))
            tmpl_name, short_base = templates[idx]
            company_name = companies[idx % len(companies)]

            # Генерируем уникальный номер/год для шаблона
            num = random.randint(10, 99)
            year = random.randint(2021, 2024)
            full_name = tmpl_name.format(num=num, year=year)
            short_name = (
                full_name
                if company_name.casefold() in full_name.casefold()
                else f"{company_name} {full_name}"
            )

            company = get_or_create(session, Company, name=company_name)
            session.flush()

            # Product
            product_id = get_next_id(session, Product, "product_id")
            product = Product(product_id=product_id)
            session.add(product)
            session.flush()

            # ProductOffer
            offer_id = get_next_id(session, ProductOffer, "offer_id")
            lo, hi = PRICE_RANGES.get(cat_name, (1_000, 100_000))
            price_rub = float(round(random.randint(lo, hi), -2) - 10)

            offer = ProductOffer(
                offer_id=offer_id,
                product_id=product_id,
                company_id=company.company_id,
                category_id=category.category_id,
                short_name=short_name,
                price_rub=price_rub,
            )
            session.add(offer)
            session.flush()

            # Properties
            props = make_consistent_properties(cat_data, company_name)
            for prop_name_str, prop_value in props.items():
                prop_name = get_or_create(session, PropertyName, name=prop_name_str)
                session.flush()
                pp = ProductProperty(
                    offer_id=offer_id,
                    property_name_id=prop_name.property_name_id,
                    value=str(prop_value),
                )
                session.add(pp)

            imported += 1
            if imported % 100 == 0:
                session.commit()
                print(f"   ✅ {imported} / {total} товаров добавлено")

    session.commit()
    return imported


# =============================================================================
#   MAIN
# =============================================================================

def main():
    total = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print("=" * 55)
    print(f"🚀 Генерация {total} тестовых товаров")
    print("=" * 55)

    db: Session = SessionLocal()
    try:
        before = db.query(func.count(ProductOffer.offer_id)).scalar() or 0
        added = generate_products(db, total=total)
        after = db.query(func.count(ProductOffer.offer_id)).scalar() or 0

        print("\n" + "=" * 55)
        print(f"✅ ГОТОВО")
        print(f"   Добавлено: {added} товаров")
        print(f"   Всего в БД: {after} (было {before})")
        print("=" * 55)
        print("Следующий шаг: запустите индексацию поиска")
        print("  POST /admin/search-index/build")
    except Exception as e:
        db.rollback()
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()