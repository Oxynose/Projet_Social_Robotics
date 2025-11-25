# Projet_Social_Robotics

## Installation

Pour l'installation de CarRacing-v2 sous environnement `conda`:
- Aller voir le `INSTALL.md`

## Test de l'agent PPO

Pour entraîner/tester l'agent ppo:
```bash
python main.py --ppo --train
python main.py --ppo --play
```

Pour entraîner/tester l'agent gail:
```bash
python main.py --gail --train        !!! Attention, l'entraînement de l'agent GAIL est très gourmant en ressource, 
python main.py --gail --play         il peut donc entraîner de fort ralentissements sur votre ordinateur 
                                     s'il n'est pas assez puissant
```

Sinon pour juste lancer RaicingCar :
```bash
python main.py
```
