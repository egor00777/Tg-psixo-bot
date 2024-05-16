import sqlite3 as sq

class DB:
    def __init__(self):
        self.connection = sq.connect('messages.db')
        self.cursor = self.connection.cursor()
        self.cursor.execute('''
            CREATE TABLE IF NOT EXISTS Users (
                id INTEGER NOT NULL,
                text TEXT NOT NULL
            )
        ''')

    def add_message(self, user_id, text):
        self.cursor.execute('INSERT INTO Users (id, text) VALUES (?, ?)', (user_id, text))
        self.connection.commit()

    def get_messages(self, user_id):
        self.cursor.execute('SELECT text FROM Users WHERE id == (?) LIMIT 50', (user_id,))
        result = self.cursor.fetchall()
        self.connection.commit()
        return result

    def delete_messages(self, user_id):
        self.cursor.execute('DELETE FROM Users WHERE id == (?)', (user_id,))

    def __del__(self):
        self.connection.commit()
        self.connection.close()
