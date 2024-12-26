# AI-Telegram-Bot
This is a machine learning project based on Tensorflow , Keras and SkLearn.

# Movie Recommendation ChatBot

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

<img width="700" height="400" alt="pic1" src="https://github.com/user-attachments/assets/c9a81750-a63e-4326-b464-b61d7565b88b" />
<img width="468" alt="pic2" src="https://github.com/user-attachments/assets/2c1cd812-6e22-4086-9987-679a1ad2b79a" />


