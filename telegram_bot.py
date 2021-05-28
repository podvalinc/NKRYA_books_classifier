# @Readability_test_bot
import telebot
from books_n_kids import predict_text_class
from predict.BERT_RUSAGE import predict_rusage_class
bot = telebot.TeleBot('')
print('Bot is ready')

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == '/start':
        bot.send_message(message.from_user.id, 'Привет, пришли текст')
    else:
        idx_cnn, prob_cnn = predict_text_class(message.text)
        idx_rusage, prob_rusage = predict_rusage_class(message.text)
        if idx_cnn == 0:
            class_name = '1-4 класс'
        elif idx_cnn == 1:
            class_name = '5-9 класс'
        else:
            class_name = '10-11 класс'
        bot.send_message(message.from_user.id, "CNN: " + class_name + " " + str(prob_cnn))
        bot.send_message(message.from_user.id, "BERT Rusage: " + idx_rusage + " " + str(prob_rusage))

bot.polling()
