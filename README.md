# ibi-deeprl
# Utilisation de `Python`

Les programmes présents sont compatibles avec une version maximale de `3.7` pour `Python`.
Des premiers tests ont étés d'abord effectués avec une version `3.8` qui menaient à des problèmes de compatibilité.

Installer les dépendances :

```
pip install -r requirements.txt
```

# Pour l'utilisation du `Notebook`

S'assurer auparavent d'avoir `Jupyter` installé.
On utilisera `virtualenvwrapper` pour la création et la gestion des `virtualenv`.


Créer un nouvel environnement virtuel
```
mkvirualenv -p /usr/bin/python3.7 deeprl
```
Se mettre dans l'environnement
```
workon deeprl
```

Ajouter le kernel Jupyter
```
python -m ipykernel install --user --name=deeprl
```

Enfin lancer Jupyter
```
jupyter lab
```
