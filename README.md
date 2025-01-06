# AI-Telegram-Bot
   **Go to the master branch for the source code.**
   **This is a machine learning project based on Tensorflow , Keras and SkLearn.**

# RO LANGUAGE :
# ChatBot de recomandare a filmelor

<img alt="pic1" src="https://github.com/user-attachments/assets/c9a81750-a63e-4326-b464-b61d7565b88b" />
<img alt="pic2" src="https://github.com/user-attachments/assets/2c1cd812-6e22-4086-9987-679a1ad2b79a" />

Un ChatBot AI creat pentru a recomanda filme utilizatorilor, bazat pe preferințele acestora. Proiectul utilizează diverse tehnologii de Machine Learning și este integrat cu Telegram pentru o interfață prietenoasă.

## Caracteristici
- **Interfață Telegram**: ChatBot-ul poate fi accesat de pe orice dispozitiv compatibil cu Telegram.
- **Recomandări personalizate**: Recomandă filme pe baza preferințelor utilizatorului și a datelor altor utilizatori similari.
- **Model AI antrenat**: Utilizează un model neural pentru a prezice ratingurile utilizatorului.

## Tehnologii utilizate
1. **Python**: Limbaj principal utilizat pentru implementare.
2. **Biblioteci Python**:
   - **Pandas**: Procesarea și gestionarea dataset-urilor.
   - **Numpy**: Crearea și manipularea vectorilor numerici.
   - **TensorFlow/Keras**: Antrenarea modelului de recomandare.
   - **Scikit-learn**: Separarea datelor de antrenament și test.
   - **Telegram API**: Integrarea ChatBot-ului cu Telegram.

## Cum funcționează
### 1. Interfață Telegram
Utilizatorul poate scrie comanda `/start` pentru a iniția conversația. Bot-ul oferă instrucțiuni despre cum să introducă datele (ex. `Nume_film Rating`).

### 2. Procesarea Preferințelor
- Bot-ul validează datele introduse (ratinguri între 1-5).
- Verifică dacă filmele există în dataset și pregătește datele pentru modelul AI.

### 3. Modelul AI
- **Arhitectură**: Rețele neurală cu straturi de embedding pentru utilizatori și filme.
- **Antrenare**: Datele sunt transformate în TensorFlow datasets și utilizate pentru antrenarea modelului.
- **Recomandări**: Modelul prezice ratinguri pentru filme, iar primele 10 filme cu cele mai mari scoruri sunt recomandate utilizatorului.

## Cum se utilizează
1. **Configurare**:
   - Asigură-te că ai instalat toate dependențele: `pip install -r requirements.txt`.
   - Creează un bot pe Telegram și obține un API token.
2. **Rulare**:
   - Rulează scriptul principal: `python main.py`.
   - Introdu comanda `/start` în chat-ul bot-ului pentru a începe.
3. **Introdu preferințele**:
   - Trimite o listă de filme și ratinguri, câte unul pe linie, de forma: `Nume_film Rating`.
   - Bot-ul va procesa datele și va oferi o listă de recomandări personalizate.

## Exemple de utilizare
- **Start:** `/start`
- **Introducere preferințe:**
  ```
  The Shawshank Redemption 5
  Inception 4
  Avatar 3
  ```
- **Răspuns:**
  ```
  Recomandări:
  1. The Godfather
  2. The Dark Knight
  3. Interstellar
  ```
---
Daca botul este hostat si activ , il poti accesa de aici [Telegram link](https://t.me/AI_OldMoviesBot).

# EN LANGUAGE : 
# Movie Recommendation ChatBot

An AI-powered ChatBot designed to recommend movies to users based on their preferences. The project leverages various Machine Learning technologies and integrates with Telegram for a user-friendly interface.

## Features
- **Telegram Interface**: The ChatBot can be accessed on any Telegram-compatible device.
- **Personalized Recommendations**: Recommends movies based on user preferences and data from similar users.
- **Trained AI Model**: Uses a neural network to predict user ratings for movies.

## Technologies Used
1. **Python**: The primary programming language used for implementation.
2. **Python Libraries**:
   - **Pandas**: For processing and managing datasets.
   - **Numpy**: For creating and manipulating numerical vectors.
   - **TensorFlow/Keras**: For training the recommendation model.
   - **Scikit-learn**: For splitting the data into training and testing sets.
   - **Telegram API**: For integrating the ChatBot with Telegram.

## How It Works
### 1. Telegram Interface
Users can type the `/start` command to initiate a conversation. The bot provides instructions on how to input data (e.g., `Movie_Name Rating`).

### 2. Preference Processing
- The bot validates the input data (ratings between 1-5).
- It checks if the movies exist in the dataset and prepares the data for the AI model.

### 3. AI Model
- **Architecture**: A neural network with embedding layers for users and movies.
- **Training**: The data is transformed into TensorFlow datasets and used to train the model.
- **Recommendations**: The model predicts ratings for movies, and the top 10 movies with the highest scores are recommended to the user.

## How to Use
1. **Setup**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`.
   - Create a bot on Telegram and obtain an API token.
2. **Run**:
   - Execute the main script: `python main.py`.
   - Enter the `/start` command in the bot's chat to begin.
3. **Input Preferences**:
   - Send a list of movies and ratings, one per line, in the format: `Movie_Name Rating`.
   - The bot will process the data and provide a personalized recommendation list.

## Example Usage
- **Start:** `/start`
- **Input Preferences:**

## Link to the Bot
If the bot is hosted and available, you can access it via this [Telegram link](https://t.me/AI_OldMoviesBot).


