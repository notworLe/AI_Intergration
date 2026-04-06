from face_recognition.database.db import DatabaseConfiguration

# Test DatabaseConfiguration class
def test_url():
    config = DatabaseConfiguration(
        user="notworle",
        password=123123,
        host="localhost",
        port=248,
        database="notworle_db",
        dialect="postgresql",
        driver="psycopg2"
    )

    assert config.url() == \
           "postgresql+psycopg2://notworle:123123@localhost:248/notworle_db"


