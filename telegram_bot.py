# @Readability_test_bot
import telebot
from books_n_kids import predict_text_class

bot = telebot.TeleBot('1753560110:AAE-8mGbP4g7aFopWvci6B9yny4MgsO8bGI')


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == '/start':
        bot.send_message(message.from_user.id, 'Привет, пришли текст')
    else:
        idx, prob = predict_text_class(message.text)
        if idx == 0:
            class_idx = '1-4 класс'
        elif idx == 1:
            class_idx = '5-9 класс'
        else:
            class_idx = '10-11 класс'
        bot.send_message(message.from_user.id, class_idx + " " + str(prob))


bot.polling()
