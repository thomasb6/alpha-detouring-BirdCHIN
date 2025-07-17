FROM python:3.9.13-slim

# Définir le répertoire de travail
ENV APP_HOME=/app
WORKDIR $APP_HOME

# Copier les fichiers dans le conteneur
COPY . ./

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Exposer le port utilisé par Dash
EXPOSE 8050

# Lancer l'application
CMD ["python", "main.py"]
