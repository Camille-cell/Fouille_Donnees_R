# Rapport du mini Projet1 : 

## Classification bayésienne
**[Fouille de données avec R--E4 AP-4209]**  
Résumé : Nous développerons un projet de classification bayésienne en utilisant l'ensemble de données sur les émotions (Kaggle) en plusieurs étapes. Nous allons employer une série de pré-traitement plus complexes et éventuellement étendre l'approche bayésienne pour inclure des ajustements (tuning) ou des probabilité supplémentaires. 

## Enseignant
**[Badr TAJINI]**

## Date de réalisation
**02/2025**

---

## Objectifs
Développer un classificateur bayésien pour prédire les émotions à partir de données textuelles.

---

## Environnement de travail
- **Langage utilisé** : *R*  
- **Outils nécessaires** : *RStudio, github*  
- **Source des données** : *Lien : Emotion Dataset (kaggle.com)*  

## Description des données 
Cet ensemble de données de classification des émotions est conçu pour faciliter la recherche et l'expérimentation dans le domaine du traitement du langage naturel et de l'analyse des émotions.
Il contient une collection diversifiée d'échantillons de texte, chacun étiqueté avec l'émotion correspondante qu'il transmet. Les émotions sont : le bonheur, la peur et la colère. Il possède 5934 valeurs et peut être utilisé pour l'analyse des sentiments, la classification des émotions, l'applications sensibles aux émotions, l'analyse des commentaires des clients, la surveillance des sentiments sur les réseaux sociaux ou la formation aux chatbots et aux assistants virtuels. 
![images des premières lignes des données](images/appercu_data.png)
---

## Résumé des travaux réalisés
1. **Étape 1** : *Chargement et exploration des données :*  
chargement des librairies nécessaires au traitement des données :
```r
library(SnowballC)       # Pour effectuer le stemming (réduction des mots à leur racine)
library(kableExtra)      # Pour créer des tableaux stylés dans les rapports (markdown/HTML)
library(dplyr)           # Pour manipuler les données (filtrage, regroupement, transformations)
library(ggplot2)         # Pour créer des visualisations graphiques
library(tidytext)        # Pour analyser les données textuelles (tokenization, etc.)
library(stringr)         # Pour manipuler les chaînes de caractères
library(tidyr)           # Pour transformer et organiser les données (pivot, unnest, etc.)
library(stopwords)       # Pour gérer les listes de mots vides (stop words)
library(tm)              # Pour effectuer des tâches de text mining (corpus, preprocessing, etc.)
library(textstem)        # Pour effectuer le lemmatization (réduction des mots à leur forme de base)
library(e1071)           # Pour les algorithmes de machine learning (SVM, naïve Bayes, etc.)
library(caret)           # Pour diviser les données en train/test et évaluer les modèles
library(stats)           # Pour les statistiques classiques et les tests (ex: tests d'hypothèse)
```
chargement et affichage des données : 
```r
data<-read.csv("G:/Mon Drive/E4/Fouille_de_donnees_avec_R/project1/archive (1)/Emotion_classify_Data.csv")
# Affichage des premières lignes sous forme de tableau HTML
data %>%
  kbl(digits = 3) %>%
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center",
                latex_options = 'stripped') %>%
  scroll_box(height = "250px")
```
Nous effectuons ensuite une analyse exploratoire des données (EDA) pour comprendre la distribution des classes, la longueur des entrées de texte :
```r
ggplot(emotion_distribution, aes(x = reorder(Emotion, -count), y = count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "Emotion", y = "Count", title = "Distribution des catégories d'émotions") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))
```
![histogramme de la distribution des émotions](images/histogramme_distribution_emotions.png)

On observe que les classes sont distribuées presque équitablement ce qui permet d'avoir un modèle robuste et d'éviter les biais. 

On analyse ensuite la distribution du nombre de caractères par commentaire
![histogramme de la distribution de la longueur des commentaires](images/nombre_caracteres_commentaires.png)
On obtient la distribution suivante :
![distribution de la longueur des commentaires](images/distrib_nb_char.png)

Afin de voir si la longueur des commentaires est bien répartie par émotion, on réalise un boxplot :
![box plot de la répartition de la longueur des commentaires par émotion](images/box_plot_nb_char.png)

On observe une répartion équitable par émotion ce qui va nous permettre de réaliser une analyse non biaisée en théorie. 

2. **Étape 2** : *Prétraitement des données* 
Nous allons d'abord nettoyer les données textuelles en supprimant les caractères spéciaux, les chiffres et les mots vides. Cela permet de rendre les données plus exploitables et d'obtenir de meilleurs résultats
```r
#nettoyage du texte
clean_text <- function(text) {
  text %>%
    # Convertir le texte en minuscules
    tolower() %>%
    # Supprimer les caractères spéciaux et les chiffres
    str_replace_all("[^a-z\\s]", "") %>%
    # Supprimer les chiffres
    str_replace_all("[0-9]", "") %>%
    # Supprimer les espaces excessifs
    str_replace_all("\\s+", " ") %>%
    # Supprimer les espaces en début et fin de chaîne
    str_trim()
}
```
Une fois le nettoyage appliqué, on supprime les mots vides. 
Cela permet de supprimer les mots qui apparaissent fréquements mais qui sont inutiles pour l'analyse des émotions tels que les articles. Cela permet de diminuer la taille des données sans perdre d'information. On ajoute certain mots personnalisés tels que ("not", "can", "will")

On tokenise en mots individuels afin diviser un texte en mots individuels, ce qui permet de rendre le texte plus manipulable pour les tâches d'analyse.

On applique ensuite le stemming pour réduire les mots tokenisés à leur racine. Cela permet par exemple de ne pas tenir compte de la conjugaison ou des pluriel. Cela permet ainsi de réduire les variations des mots. 
 
3. **Étape 3** : *Entraînement du modèle bayésien*  
4. **Étape 4** : *Évaluation du modèle*  
5. **Étape 5** : *Amélioration et optimisation*  

---

## 📈 Résultats principaux
- *Inclure les résultats sous forme de tableau, graphique ou texte.*  
- *Expliquer brièvement ce qu'ils signifient.*

---

## 💡 Conclusion
- **Ce que j'ai appris** : *[Énumérer les notions ou compétences acquises.]*  
- **Problèmes rencontrés** : *[Lister les difficultés et les solutions apportées.]*  
- **Pistes d'amélioration** : *[Suggestions pour aller plus loin ou améliorer les travaux.]*  

---

## 📂 Organisation des fichiers
- `script.R` : *Script principal utilisé pour...*  
- `data.csv` : *Jeu de données utilisé dans les exercices.*  
- `rapport.pdf` : *Version PDF du rapport détaillé (si applicable).*  

---

## ✍️ Auteur
- **Nom** : [Votre nom]  
- **Contact** : [Votre email, lien GitHub, etc.]

