import asyncio
import logging
import sys
import numpy as np

import gc

import os

from aiogram import Bot, Dispatcher, html
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import CommandStart
from aiogram import F
from aiogram.filters import Command
from aiogram.types import Message, ContentType, File
from aiogram.types import ReplyKeyboardRemove, \
    ReplyKeyboardMarkup, KeyboardButton, \
    InlineKeyboardMarkup, InlineKeyboardButton

from pathlib import Path

from random import randint

import soundfile as sf
from saiga import Saiga

from model import BinarModel, ClassModel, SpeechModel

from deep_translator import GoogleTranslator

from db import DB

# Переводчик
translator = GoogleTranslator(source='auto', target='en')

# FILE ID BAACAgIAAxkBAAIVs2Y8xbDMffMsJ7RYAmfue-_J-pr9AAJbSAACWPnoSW1At0bPOxN8NQQ
# GIF ID CgACAgIAAxkBAAIcPmY-ERTq3INWgvyTXmiN3nC8Oq33AAKpTQACQCb5SdXgFCJrzpLXNQQ

PASSWORD = 'HHSSEE'

bot = None

BOT_TOKEN = ''

dp = Dispatcher()

diagnosis_button = KeyboardButton(text='Получить диагноз')
kb = [
    [diagnosis_button],
]
keyboard = ReplyKeyboardMarkup(resize_keyboard=True, keyboard=kb)

# MODELS

b_model = BinarModel()
c_model = ClassModel()
d_model = Saiga()

s_model = SpeechModel()

# LABELS FOR CLASSIFIER

id2label = {0: 'Пограничное растройство личности', 1: 'Биполярное расстройство', 2: 'Депрессия', 3: 'Тревожность',
            4: 'Шизофрения', 5: 'Психическое заболевание'}

# WRITE LOG

# def write_log(text):
#     file = open('log.txt', 'a', encoding='utf-8')
#     file.write(text)
#     file.write('\n')
#     file.close()


# DATABASE

db = DB()


# FILES
async def handle_file(file, file_name, path):
    Path(f'{path}').mkdir(parents=True, exist_ok=True)
    await bot.download(file=file, destination=f'{path}/{file_name}')


def prepare_file(file_name, path):
    data, samplerate = sf.read(f'{path}/{file_name}')
    wav_file_name = file_name.split('.')[0] + '.wav'
    print(wav_file_name)
    sf.write(f'{path}/{wav_file_name}', data, samplerate)
    return wav_file_name


def delete_file(file_name, path):
    Path(f'{path}/{file_name}').unlink(missing_ok=True)


# LOGIN
logged_users = []


@dp.message(Command('login'))
async def login(message: Message):
    if PASSWORD in message.text:
        logged_users.append(message.from_user.id)
        await message.answer('OK', reply_markup=keyboard)
    else:
        await message.answer('ПАРОЛЬ НЕВЕРНЫЙ', reply_markup=keyboard)


# SET PASSWORD
@dp.message(Command('set'))
async def set_password(message: Message):
    if message.from_user.id != 5050514557:
        return None
    global PASSWORD
    PASSWORD = message.text[5:]


# Стартовые команды, начинают всё заново
@dp.message(Command('start', 'stop', 'cancel', 'clear'))
async def start_message_handler(message: Message):
    db.delete_messages(message.from_user.id)
    await message.answer(
        'Привет! Я ваш виртуальный психолог и готов оказать вам поддержку и помощь. Расскажите мне о своих мыслях и '
        'чувствах, и мы вместе найдем пути к вашему благополучию.',
        reply_markup=keyboard)


# Получаем диагноз
@dp.message(F.text.lower() == 'получить диагноз')
async def get_diagnosis(message: Message):
    if message.from_user.id not in logged_users:
        await message.answer('ВЫ НЕ ВОШЛИ В СИСТЕМУ\nЧТОБЫ ЭТО СДЕЛАТЬ ИСПОЛЬЗУЙТЕ КОМАНДУ /login')
        return None
    upload_message = await bot.send_animation(message.from_user.id,
                                              animation='CgACAgIAAxkBAAIcPmY-ERTq3INWgvyTXmiN3nC8Oq33AAKpTQACQCb5SdXgFCJrzpLXNQQ')
    text = db.get_messages(message.from_user.id)
    if text == '':
        await bot.delete_message(message_id=upload_message.message_id, chat_id=message.chat.id)
        await message.answer('Извините, но предоставленной Вами информации недостаточно для определения диагноза')
    b_text = '<br>'.join(map(lambda x: x[0], text))
    c_model_pred = np.sum([c_model.predict(translator.translate(sub_text[0])) for sub_text in text], axis=0)

    db.delete_messages(message.from_user.id)
    b_model_pred = b_model.predict(translator.translate(b_text))
    illness = ''
    if b_model_pred > 0.5:
        illness = id2label[np.argmax(c_model_pred, axis=-1)]
    else:
        illness = 'здоровый человек'
    response = d_model.support_message(illness.lower())
    await bot.delete_message(message_id=upload_message.message_id, chat_id=message.chat.id)
    await message.answer(illness.upper(), reply_markup=keyboard)
    if b_model_pred > 0.5:
        await message.answer(response, reply_markup=keyboard)
    else:
        await message.answer(response, reply_markup=keyboard)

    gc.collect()


# Диалог с пользователем, обрабатываем сообщения
@dp.message(F.text)
async def message_handler(message: Message):
    if message.from_user.id not in logged_users:
        await message.answer('ВЫ НЕ ВОШЛИ В СИСТЕМУ\nЧТОБЫ ЭТО СДЕЛАТЬ ИСПОЛЬЗУЙТЕ КОМАНДУ /login')
        return None
    text = message.text.lower()
    if len(text) < 10:
        await message.answer('Недостаточно текста, попробуйте еще раз', reply_markup=keyboard)
        return None
    print(message.from_user.username)
    print(text)
    db.add_message(message.from_user.id, text)
    upload_message = await bot.send_animation(message.from_user.id,
                                              animation='CgACAgIAAxkBAAIcPmY-ERTq3INWgvyTXmiN3nC8Oq33AAKpTQACQCb5SdXgFCJrzpLXNQQ')
    response = d_model.process_message(message.from_user.id, text)
    await bot.delete_message(message_id=upload_message.message_id, chat_id=message.chat.id)
    await message.answer(response, reply_markup=keyboard)
    print(response)
    gc.collect()


@dp.message(F.voice)
async def voice_handler(message: Message):
    if message.from_user.id not in logged_users:
        await message.answer('ВЫ НЕ ВОШЛИ В СИСТЕМУ\nЧТОБЫ ЭТО СДЕЛАТЬ ИСПОЛЬЗУЙТЕ КОМАНДУ /login')
        return None
    voice = message.voice
    path = '.'
    file_name = f'{voice.file_id}.ogg'

    async def r():
        await handle_file(file=voice, file_name=file_name, path=path)
        result = s_model.recognize(f'{path}/{file_name}')
        delete_file(file_name=file_name, path=path)
        return result

    upload_message = await bot.send_animation(message.from_user.id,
                                              animation='CgACAgIAAxkBAAIcPmY-ERTq3INWgvyTXmiN3nC8Oq33AAKpTQACQCb5SdXgFCJrzpLXNQQ')
    recognize = await asyncio.create_task(r())
    #
    if recognize is None:
        await message.answer('Извините, мы не распознали Ваше голосовое сообщение!', reply_markup=keyboard)
    print(message.from_user.username)
    print(recognize)
    await message.answer('[' + recognize + ']', reply_markup=keyboard)
    db.add_message(message.from_user.id, recognize)
    response = d_model.process_message(message.from_user.id, recognize)
    await bot.delete_message(message_id=upload_message.message_id, chat_id=message.chat.id)
    await message.answer(response, reply_markup=keyboard)
    gc.collect()


async def main():
    global bot
    bot = Bot(token=BOT_TOKEN)
    await dp.start_polling(bot)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    asyncio.run(main())
