#effacer toutes les variables en mémoire
rm(list=ls())

# Charger le package SnowballC
# Charger les librairies nécessaires
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

data<-read.csv("archive (1)/Emotion_classify_Data.csv")
# Affichage des premières lignes sous forme de tableau HTML
data %>%
  kbl(digits = 3) %>%
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center",
                latex_options = 'stripped') %>%
  scroll_box(height = "250px")
# Compter la fréquence des différentes émotions
emotion_distribution<-data %>%
  group_by(Emotion) %>%
  summarise(count=n()) %>%
  arrange(desc(count))

# Visualiser la distribution des émotions
ggplot(emotion_distribution, aes(x = reorder(Emotion, -count), y = count)) +
  geom_bar(stat = "identity", fill = "skyblue") +
  labs(x = "Emotion", y = "Count", title = "Distribution des catégories d'émotions") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#analyser le nombre de caractère des commentaires
data$comment_length<-nchar(data$Comment)
summary(data$comment_length)
#histogramme
ggplot(data, aes(x = comment_length)) +
  geom_histogram(binwidth = 10, fill = "lightblue", color = "black") +
  labs(x = "Longueur du commentaire", y = "Fréquence", title = "Distribution de la longueur des commentaires") +
  theme_minimal()



# Visualiser la longueur des commentaires par émotion
ggplot(data, aes(x = Emotion, y = comment_length, fill = Emotion)) +
  geom_boxplot() +
  labs(x = "Emotion", y = "Longueur du commentaire", title = "Longueur des commentaires par émotion") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

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

#application du nettoyage sur les commentaires
data$cleaned_comment<-sapply(data$Comment, clean_text)
head(data$cleaned_comment)

#suppression des mots vides
# Charger les mots vides en anglais
stop_words_list <-  c(stopwords("en"), "not", "can", "will", "actual", "ago","also", "around","back")
print(stop_words_list)

# Tokeniser le texte en mots individuels et supprimer les mots vides
cleaned_data <- data %>%
  mutate(doc_id = row_number()) %>% # Ajouter un identifiant unique pour chaque commentaire
  unnest_tokens(word, cleaned_comment) %>% # Diviser les commentaires en mots
  filter(!word %in% stop_words_list) # Supprimer les mots vides


# Reconstruire les commentaires nettoyés
cleaned_comments <- cleaned_data %>%
  group_by(doc_id) %>%
  summarise(cleaned_comment = paste(word, collapse = " ")) %>%
  ungroup()
print(cleaned_comments)

data <- data %>%
  mutate(doc_id = row_number())
# Joindre les commentaires nettoyés au jeu de données original
data <- data %>%
  left_join(cleaned_comments, by = "doc_id")

# Afficher un échantillon des commentaires nettoyés
head(data$cleaned_comment.y)

# Préparation des données
preprocessed_data <- data %>%
  select(-Comment, -comment_length, -cleaned_comment.x)
print(preprocessed_data)
# Tokenisation
tokenized_data <- preprocessed_data %>%
  unnest_tokens(word, cleaned_comment.y)
print(tokenized_data)

# Calcul des fréquences des mots
word_counts <- tokenized_data %>%
  count(word)
print(word_counts)
# Filtrer les mots en fonction de leur fréquence
filtered_words <- word_counts %>%
  filter(n > 0.0005 * nrow(tokenized_data) & n < 0.9 * nrow(tokenized_data))
print(filtered_words)
# Filtrer les données tokenisées pour ne garder que les mots les plus fréquents
tokenized_data <- tokenized_data %>%
  filter(word %in% filtered_words$word)
# Calculer TF-IDF
tokenized_data <- tokenized_data %>%
  mutate(doc_id = as.character(doc_id))  # S'assurer que doc_id est une chaîne de caractères

tf_idf_data <- tokenized_data %>%
  count(doc_id, word) %>%  # Compter les occurrences des mots dans chaque document
  bind_tf_idf(term = word, document = doc_id, n = n)  # Calculer TF-IDF

# Trier par score TF-IDF décroissant
tf_idf_data <- tf_idf_data %>%
  arrange(desc(tf_idf))
print(tf_idf_data)
# Garder les 5000 mots avec les meilleurs scores TF-IDF
top_5000_words <- tf_idf_data %>%
  group_by(word) %>%
  summarise(max_tf_idf = max(tf_idf)) %>%
  arrange(desc(max_tf_idf)) %>%
  top_n(10000, max_tf_idf)

# Filtrer les données TF-IDF pour ne garder que les 1000 mots
tf_idf_data_filtered <- tf_idf_data %>%
  filter(word %in% top_5000_words$word)
print(tf_idf_data_filtered)
# Créer la matrice TF-IDF
tf_idf_matrix <- tf_idf_data_filtered %>%
  pivot_wider(
    names_from = word, 
    values_from = tf_idf, 
    values_fill = list(tf_idf = 0),  # Remplir les valeurs manquantes par 0
    names_repair = "unique"  # Corriger les noms dupliqués
  )

# Joindre les étiquettes d'Emotion
preprocessed_data <- preprocessed_data %>%
  mutate(doc_id = as.character(doc_id))

tf_idf_matrix <- tf_idf_matrix %>%
  left_join(preprocessed_data %>% select(doc_id, Emotion), by = "doc_id")

# Diviser les données en train (80%) et test (20%)
set.seed(123)  # Pour reproductibilité
split <- createDataPartition(tf_idf_matrix$Emotion, p = 0.7, list = FALSE)
train_data <- tf_idf_matrix[split, ]
test_data <- tf_idf_matrix[-split, ]

# Entraîner le modèle bayésien naïf
train_data_model <- train_data %>%
  select(-doc_id)  # Supprimer doc_id si présent
test_data_model <- test_data %>%
  select(-doc_id)  # Supprimer doc_id si présent
# Configurer la validation croisée à 10 plis
ctrl <- trainControl(method = "cv", number = 10, verboseIter = TRUE)
# Entraîner le modèle Naive Bayes avec validation croisée
naive_model_cv <- train(Emotion ~ ., data = train_data_model, 
                        method = "naive_bayes", 
                        trControl = ctrl, 
                        tuneLength = 5,
)
# Afficher les résultats de la validation croisée
print(naive_model_cv)

# Prédictions avec le modèle validé par croisée
test_predictions <- predict(naive_model_cv, newdata = test_data_model)
# Convertir test_predictions et test_data_model$Emotion en facteurs avec les mêmes niveaux
levels_emotion <- levels(factor(train_data_model$Emotion))  # Niveaux des émotions dans les données d'entraînement

# Convertir les prédictions et les vraies étiquettes en facteurs avec les mêmes niveaux
test_predictions <- factor(test_predictions, levels = levels_emotion)
test_data_model$Emotion <- factor(test_data_model$Emotion, levels = levels_emotion)
# Évaluer les performances
conf_matrix <- confusionMatrix(test_predictions, test_data_model$Emotion)

# Afficher la matrice de confusion
print(conf_matrix)

# Calcul des F1 scores pour chaque classe
precision <- conf_matrix$byClass[, "Pos Pred Value"]
recall <- conf_matrix$byClass[, "Sensitivity"]

# Calcul du F1 score pour chaque classe
f1_scores <- 2 * (precision * recall) / (precision + recall)

# Affichage des F1 scores pour chaque classe
f1_scores


# Calcul de la précision globale
accuracy <- conf_matrix$overall['Accuracy']
cat("Précision globale :", accuracy, "\n")

# Visualiser la matrice de confusion
library(ggplot2)
ggplot(as.data.frame(conf_matrix$table), aes(Reference, Prediction, fill = Freq)) +
  geom_tile() +
  scale_fill_gradient(low = "white", high = "blue") +
  labs(title = "Matrice de Confusion", x = "Vraies Étiquettes", y = "Prédictions") +
  theme_minimal()

