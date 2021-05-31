# @Readability_test_bot
import telebot
from books_n_kids import predict_text_class
from predict.BERT_RUSAGE import predict_rusage_class
from predict.BERT_MINOBR import Kfold_predict

bot = telebot.TeleBot('1753560110:AAEOLVjuMiOreebh6nCiC88CkkXm1FEggf4')
print('Bot is ready')


@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == '/start':
        bot.send_message(message.from_user.id, 'Отправьте текст для оценки')
    else:
        idx_cnn, prob_cnn = predict_text_class(message.text)
        idx_rusage, prob_rusage = predict_rusage_class(message.text)
        idx_minobr, prob_minobr = Kfold_predict(message.text, 5)
        if idx_cnn == 0:
            class_name = '1-4 класс'
        elif idx_cnn == 1:
            class_name = '5-9 класс'
        else:
            class_name = '10-11 класс'

        bot.send_message(message.from_user.id, f'CNN: {class_name} {prob_cnn:.{3}f} \n' +
                         f'BERT Мин. Просв: {idx_minobr} {prob_minobr:.{3}f}\n' +
                         f'BERT Rusage: {idx_rusage} {prob_rusage:.{3}f}')


bot.polling()
