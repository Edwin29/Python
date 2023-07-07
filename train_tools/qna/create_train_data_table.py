import pymysql
import sys
#sys.path.append('D:\Python Source\CHATBOT\config')
from config.DatabaseConfig import *  # import DB connection information

db = None
try:
    db = pymysql.connect(
        host='127.0.0.1',
        user='root',
        passwd='Edwin124610@!',
        db='ChatBot',
        charset='utf8'
    )
    print("DB connect Successfully")

    # Define sql for Generate tabel
    sql = '''
      CREATE TABLE IF NOT EXISTS `chatbot_train_data` (
      `id` INT UNSIGNED NOT NULL AUTO_INCREMENT,
      `intent` VARCHAR(45) NULL,
      `ner` VARCHAR(1024) NULL,
      `query` TEXT NULL,
      `answer` TEXT NOT NULL,
      `answer_image` VARCHAR(2048) NULL,
      PRIMARY KEY (`id`))
    ENGINE = InnoDB DEFAULT CHARSET=utf8
    '''
    # Generate table
    with db.cursor() as cursor:
        cursor.execute(sql)


except Exception as e:
    print(e)
    

finally:
    if db is not None:
        db.close()
