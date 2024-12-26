import pandas as pd
import tensorflow as tf
from keras.src.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras.src.layers import Embedding, Input, Concatenate , Dense , Flatten , Dropout , LeakyReLU
from keras.src.models import Model
import numpy as np
import time
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes



nume_coloane_u_data = ["user_id", "movie_id", "rating", "timestamp"]


data = pd.read_csv("D:/Proiecte Python/TelegramBot/u.data", sep="\t", names=nume_coloane_u_data)


columns = [
    "movie_id", "movie_title", "release_date", "video_release_date", "IMDb_URL",
    "unknown", "Action", "Adventure", "Animation", "Children's", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"
]
item = pd.read_csv("D:/Proiecte Python/TelegramBot/u.item", sep="|", encoding="latin-1", names=columns)
genres_matrix = item.iloc[:, 5:].to_numpy()
movie_ids = item["movie_id"].to_numpy()


movie_genres_dict = {int(movie_id): genres_matrix[i] for i, movie_id in enumerate(movie_ids)}

movie_titles = item["movie_title"]

async def training_function(update: Update, context: ContextTypes.DEFAULT_TYPE):

    global data
    next_user_id = data['user_id'].max() + 1
    user_input = update.message.text
    print(f"Input primit: {user_input}")

    lines = user_input.split("\n")

    for line in lines:
        if line.strip().lower() == "stop":
            break

        try:

            parts = line.rsplit(' ', 1)
            movie_name = parts[0].strip()
            rating = int(parts[1].strip())

            # Validare rating
            if rating < 1 or rating > 5:
                await update.message.reply_text(f"Rating invalid pentru '{movie_name}'. Introdu un număr între 1 și 5.")
                continue


            movie_row = item[item["movie_title"].str.contains(movie_name, case=False, na=False)]

            if movie_row.empty:
                await update.message.reply_text(f"Filmul '{movie_name}' nu a fost găsit!")
                continue

            movie_id = movie_row.iloc[0]['movie_id']  # Obține ID-ul filmului
            timestamp = int(time.time())  # Timestamp bazat pe timpul curent

            # Adaugă o nouă intrare în DataFrame-ul data
            new_entry = pd.DataFrame([[next_user_id, movie_id, rating, timestamp]], columns=nume_coloane_u_data)
            data = pd.concat([data, new_entry], ignore_index=True)

            await update.message.reply_text(f"Adăugat: {movie_name} (ID: {movie_id}), Rating: {rating}")

        except ValueError:
            await update.message.reply_text(f"Format invalid pentru linia: '{line}'. Asigură-te că ai introdus corect: 'nume_film rating'.")
        except Exception as e:
            await update.message.reply_text(f"A apărut o eroare pentru filmul '{movie_name}': {e}")

    await update.message.reply_text("....Analizez preferintele utilizatorilor asemanatoare cu ale dumneavoastra....")

    id_useri = data["user_id"].unique()
    id_filme = data["movie_id"].unique()


    user2idx = {user_id: idx for idx, user_id in enumerate(id_useri)}
    movie2idx = {movie_id: idx for idx, movie_id in enumerate(id_filme)}

    data["user_id"] = data["user_id"].map(user2idx)
    data["movie_id"] = data["movie_id"].map(movie2idx)
    default_genre_vector = np.zeros(19)
    data ["movie_genres"] = data["movie_id"].map(lambda x: movie_genres_dict.get(x, default_genre_vector))


    train_data, test_data = train_test_split(data, test_size=0.2)

    train_data['movie_genres'] = train_data['movie_genres'].apply(lambda x: np.array(x))
    test_data['movie_genres'] = test_data['movie_genres'].apply(lambda x: np.array(x))

    def create_datasets(train_data, test_data, batch_size=128):
        train_movie_genres = np.stack(train_data["movie_genres"].values).astype(np.int32)
        test_movie_genres = np.stack(test_data["movie_genres"].values).astype(np.int32)
        train_user_ids = train_data["user_id"].values.astype(np.int32)
        train_movie_ids = train_data["movie_id"].values.astype(np.int32)
        test_user_ids = test_data["user_id"].values.astype(np.int32)
        test_movie_ids = test_data["movie_id"].values.astype(np.int32)

        train_dataset = tf.data.Dataset.from_tensor_slices(({
            "user_id": train_user_ids,
            "movie_id": train_movie_ids,
            "movie_genres" : train_movie_genres
        }, train_data["rating"].values))
        test_dataset = tf.data.Dataset.from_tensor_slices(({
            "user_id": test_user_ids,
            "movie_id": test_movie_ids,
            "movie_genres": test_movie_genres
        }, test_data["rating"].values))

        train_dataset = train_dataset.shuffle(len(train_data)).batch(batch_size)
        test_dataset = test_dataset.batch(batch_size)

        return train_dataset, test_dataset

    train_dataset, test_dataset = create_datasets(train_data, test_data)


    users_number = len(id_useri)
    movies_number = len(id_filme)
    dim_embedding = 64

    user_input = Input(shape=(1,), name='user_id', dtype=tf.int32)
    movie_input = Input(shape=(1,), name='movie_id', dtype=tf.int32)
    genre_input = Input(shape=(19,), name='movie_genres', dtype=tf.int32)

    user_embedding = Embedding(input_dim=users_number, output_dim=dim_embedding, name="user_embedding")(user_input)
    movie_embedding = Embedding(input_dim=movies_number, output_dim=dim_embedding, name="movie_embedding")(movie_input)

    user_flatten = Flatten(name="user_flatten")(user_embedding)
    movie_flatten = Flatten(name="movie_flatten")(movie_embedding)
    concat_features = Concatenate(name="concatenate")([user_flatten, movie_flatten, genre_input])

    dense_layer = Dense(128, activation='relu', name="dense_1")(concat_features)
    dense_layer = LeakyReLU()(dense_layer)
    dense_layer = Dropout(0.2, name="dropout_1")(dense_layer)
    dense_layer = Dense(64, activation='relu', name="dense_2")(dense_layer)
    dense_layer = Dropout(0.2, name="dropout_2")(dense_layer)
    dense_layer = Dense(64, activation='relu', name="dense_3")(dense_layer)

    output = Dense(1, activation='linear', name="output")(dense_layer)

    model = Model(inputs=[user_input, movie_input, genre_input], outputs=output)
    model.compile(optimizer="adam", loss="mean_squared_error", metrics=['mae'])


    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=100,
        batch_size=128,
        callbacks=[early_stopping]
    )
    def recommend_movies(user_id, model, num_movies=1681,top_k=10):

        movie_ids_tensor = tf.constant(np.arange(num_movies))
        user_ids_tensor = tf.constant([user2idx[user_id]] * num_movies)
        genre_tensor = np.stack([movie_genres_dict.get(int(movie_id), np.zeros(19)) for movie_id in np.arange(1, num_movies + 1)])

        predictions = model.predict(
            {"user_id": user_ids_tensor,
             "movie_id": movie_ids_tensor,
             "movie_genres": genre_tensor
             }
        )

        top_k_movies = predictions.flatten().argsort()[-top_k:][::-1]

        recommended_movie_ids = [id_filme[idx] for idx in top_k_movies]
        recommended_movie_titles = [movie_titles[movie_id] for movie_id in recommended_movie_ids]

        return recommended_movie_titles


    recommended_movies = recommend_movies(next_user_id, model)
    response = "Altor utilizatori cu aceleași preferințe ca și tine le-a plăcut și:\n\n"
    response += "\n".join(recommended_movies)

    await update.message.reply_text(response)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Salut! Trimite-mi filmele și ratingurile tale pe fiecare linie")
    await update.message.reply_text("Formatul este : 'nume_film' 'rating' ; Spre exemplu : 'toy story 5'")

if __name__ == "__main__":

    TOKEN = "7662407354:AAGKyXvxGFShblrscA36h1j6a4Gi16oUnug"

    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))

    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, training_function))

    print("Botul rulează...")

    application.run_polling()
    print(data['user_id'].tail())


