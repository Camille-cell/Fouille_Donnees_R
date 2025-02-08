# #effacer toutes les variables en mémoire
rm(list=ls()) 


#permet de réaliser des “beaux” tableaux au format html ou pdf.
library(kableExtra)
library(dplyr)
library(ggplot2)
library(tidytext)
library(stringr)
library(text)
library(quanteda)
library(tm)
library(tidyverse)
library(quanteda.textstats)
library(MASS)
library(e1071)
library(caret)
library(pROC)
library(ROSE)
library(text2vec)

#Chargement des données
data<-read.csv("C:/Users/mimia/Fouille_de_donnees_avec_R/projet_final/datasets/train_drcat_04.csv")

#Analyse exploratoire
head(data)

# Résumé statistique des variables
summary(data)



# Vérifier les valeurs manquantes
colSums(is.na(data))

# Séparer les données en jeu d'entraînement (70%) et jeu de test (30%)
set.seed(123)  # Pour la reproductibilité
train_index <- createDataPartition(data$label, p = 0.7, list = FALSE)

train_data <- data[train_index, ]
test_data <- data[-train_index, ]

#Distribution des classes
ggplot(data, aes(x = as.factor(label))) +
  geom_bar(fill = "steelblue") +
  labs(title = "Distribution des classes", x = "Classe", y = "Nombre d'observations") +
  theme_minimal()

#Réquilibrage des classes
# Séparation des classes
data_balanced_human <- train_data %>% filter(label == 0)
data_balanced_ai <- train_data %>% filter(label == 1)

# Sous-échantillonnage de la classe majoritaire
set.seed(123)  # Assurez la reproductibilité
data_balanced_human_sample <- data_balanced_human %>% sample_n(nrow(data_balanced_ai))

# Réunification des données équilibrées
data_balanced <- bind_rows(data_balanced_human_sample, data_balanced_ai)

# Vérification
table(data_balanced$label)

#Réduction de la taille du jeu de donnée pour plus de rapidité d'execution
set.seed(123)  # Assurer la reproductibilité

# Taille d'échantillon souhaitée
sample_size <- 500  

# Sélection aléatoire d'observations équilibrées
data_sample <- data_balanced %>% group_by(label) %>% sample_n(min(n(), sample_size / 2)) %>% ungroup()

# Vérification de la distribution des classes
table(data_sample$label)

#Etude de la longeur des textes
# Ajouter une colonne pour la longueur des textes
data_balanced <- data_balanced %>% mutate(text_length = nchar(text))

# Statistiques de la longueur des textes
summary(data_balanced$text_length)

# Histogramme des longueurs de texte
ggplot(data_balanced, aes(x = text_length)) +
  geom_histogram(fill = "orange", bins = 50) +
  labs(title = "Distribution de la longueur des textes", x = "Nombre de caractères", y = "Fréquence") +
  theme_minimal()

#Etude de la fréquence des mots
# Tokenisation et comptage des mots
word_counts <- data_balanced %>%
  unnest_tokens(word, text) %>%
  count(word, sort = TRUE)

# Affichage des mots les plus fréquents (sans stopwords)
word_counts %>%
  filter(!word %in% stopwords::stopwords("en")) %>%
  top_n(20, n) %>%
  ggplot(aes(x = reorder(word, n), y = n)) +
  geom_col(fill = "purple") +
  coord_flip() +
  labs(title = "Mots les plus fréquents", x = "Mot", y = "Fréquence") +
  theme_minimal()

#Comparaison entre label=0(humain) et label=1(LLM)
# Comparaison des longueurs de texte par classe
ggplot(data_balanced, aes(x = as.factor(label), y = text_length, fill = as.factor(label))) +
  geom_boxplot() +
  labs(title = "Comparaison des longueurs de texte entre classes", x = "Classe", y = "Longueur du texte") +
  theme_minimal()

#analyse de la complexité du texte
# Ajouter une colonne pour le nombre de mots
data_balanced <- data_balanced %>% mutate(word_count = str_count(text, "\\S+"))

# Comparaison des mots par classe
ggplot(data_balanced, aes(x = as.factor(label), y = word_count, fill = as.factor(label))) +
  geom_boxplot() +
  labs(title = "Nombre de mots par classe", x = "Classe", y = "Nombre de mots") +
  theme_minimal()


#Features extraction
#perplexité (pour savoir si le texte est prévisible selon un llm)
# Tokenisation des textes
tokens <- tokens(data_balanced$text, remove_punct = TRUE, remove_symbols = TRUE)


# Extraction des bigrammes
bigrams <- data_balanced %>%
  unnest_tokens(bigram, text, token = "ngrams", n = 2) %>%
  count(bigram, sort = TRUE)

# Extraction des trigrams
#trigrams <- data_balanced %>%
#  unnest_tokens(trigram, text, token = "ngrams", n = 3) %>%
#  count(trigram, sort = TRUE)

# Affichage des 10 bigrammes et trigrams les plus fréquents
print(head(bigrams, 10))
#print(head(trigrams, 10))


# Création d'un modèle de langue basé sur un corpus de texte
text_corpus <- corpus(data_balanced$text)

#Création d'un document-feature matrix (DFM)
dfm_matrix <- dfm(tokens(text_corpus), remove_punct = TRUE, remove_numbers = TRUE)

# Création d'un modèle de langue (bigrammes)
bigram_model <- dfm(tokens(text_corpus), ngrams = 2)





# Fonction pour calculer la perplexité sur les bigrammes
compute_perplexity <- function(texts) {
  vocabulary <- create_vocabulary(itoken(texts, progressbar = FALSE))
  vectorizer <- vocab_vectorizer(vocabulary)
  dtm <- create_dtm(itoken(texts), vectorizer)
  
  term_prob <- colSums(dtm) / sum(dtm)
  perplexity <- exp(-mean(log(term_prob[term_prob > 0])))
  return(perplexity)
}

data_balanced$perplexity <- sapply(data_balanced$text, compute_perplexity)







#Etude de la variabilité de la longueur des phrases
# Calcul du nombre de mots par phrase
data_balanced$sentence_count <- str_count(data_balanced$text, "[.!?]")  # Nombre de phrases
data_balanced$word_count <- str_count(data_balanced$text, "\\S+")  # Nombre total de mots
data_balanced$avg_sentence_length <- data_balanced$word_count / (data_balanced$sentence_count + 1)

# Affichage des statistiques
summary(data_balanced$avg_sentence_length)
colnames(data_balanced)
# Visualisation
ggplot(data_balanced, aes(x = avg_sentence_length, fill = as.factor(label))) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution de la longueur moyenne des phrases",
       x = "Nombre moyen de mots par phrase",
       y = "Densité") +
  theme_minimal()

#score de lisibilité
# Calcul des scores de lisibilité
readability_scores <-  textstat_readability(data_balanced$text, measure = "Flesch.Kincaid")
data_balanced$readability <- readability_scores$Flesch.Kincaid  # Extraction correcte
# Comparaison entre les classes
ggplot(data_balanced, aes(x = readability, fill = as.factor(label))) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution des scores de lisibilité",
       x = "Score Flesch-Kincaid",
       y = "Densité") +
  theme_minimal()




#complexité et diversité du vocabulaire

# Type-Token Ratio (TTR)
data_balanced$TTR <- sapply(strsplit(data_balanced$text, " "), function(words) length(unique(words)) / length(words))

# Visualisation
ggplot(data_balanced, aes(x = TTR, fill = as.factor(label))) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution de la diversité lexicale (TTR)",
       x = "Type-Token Ratio",
       y = "Densité") +
  theme_minimal()
# Charger les stopwords en anglais
library(stopwords)

# Séparation des textes par classe
text_human <- data_balanced %>% filter(label == 0) %>% pull(text)
text_ai <- data_balanced %>% filter(label == 1) %>% pull(text)

# Fonction pour tokeniser et compter les mots après suppression des stopwords
clean_and_count_words <- function(text_vector) {
  words <- unlist(strsplit(tolower(paste(text_vector, collapse = " ")), "\\W+"))  # Tokenisation
  words <- words[!(words %in% stopwords("en"))]  # Suppression des stopwords
  return(data.frame(table(words)))  # Fréquence des mots restants
}

# Extraction des mots fréquents après suppression des stopwords
word_counts_human <- clean_and_count_words(text_human)
word_counts_ai <- clean_and_count_words(text_ai)

# Sélectionner les mots les plus fréquents
top_words_human <- word_counts_human %>% arrange(desc(Freq)) %>% head(50) %>% pull(words)
top_words_ai <- word_counts_ai %>% arrange(desc(Freq)) %>% head(50) %>% pull(words)

# Fonction pour compter l'apparition des mots caractéristiques dans chaque texte
count_word_occurrences <- function(text, word_list) {
  words <- unlist(strsplit(tolower(text), "\\W+"))
  words <- words[!(words %in% stopwords("en"))]  # Supprimer stopwords ici aussi
  return(sum(words %in% word_list))
}

# Appliquer cette fonction pour chaque texte
data_balanced$human_word_count <- sapply(data_balanced$text, count_word_occurrences, word_list = top_words_human)
data_balanced$ai_word_count <- sapply(data_balanced$text, count_word_occurrences, word_list = top_words_ai)

# Visualisation de la distribution des nouvelles features
ggplot(data_balanced, aes(x = human_word_count, fill = as.factor(label))) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution des mots caractéristiques humains", x = "Nombre de mots humains", y = "Densité") +
  theme_minimal()

ggplot(data_balanced, aes(x = ai_word_count, fill = as.factor(label))) +
  geom_density(alpha = 0.5) +
  labs(title = "Distribution des mots caractéristiques IA", x = "Nombre de mots IA", y = "Densité") +
  theme_minimal()



#TF-IDF
# Création d'un corpus
text_corpus <- corpus(data_balanced$text)

# Tokenization et transformation en Document-Feature Matrix (DFM)
dfm_matrix <- dfm(tokens(text_corpus), dmf_remove = stopwords("en"), dmf_stem = TRUE, remove_punct = TRUE)

# Application de TF-IDF
dfm_tfidf <- dfm_weight(dfm_matrix, scheme = "logcount")

# Vérification de la matrice
dim(dfm_tfidf)
library(dplyr)
#Analyse discriminante linéaire (LDA)
# Sélectionner un sous-ensemble des données
set.seed(123)
data_balanced_sample <- data_balanced[sample(nrow(data_balanced), 1000), ]  # Échantillon de 1000 lignes
str(data_balanced_sample)
# Suppression des colonnes non pertinentes en Base R
data_balanced_sample_cleaned <- data_balanced_sample[, setdiff(names(data_balanced_sample), c("text", "source", "fold"))]
str(data_balanced_sample_cleaned)
# Convertir label en facteur
data_balanced_sample_cleaned$label <- as.factor(data_balanced_sample_cleaned$label)

# Identifier et supprimer les colonnes constantes
constant_columns <- sapply(data_balanced_sample_cleaned[, -1], function(x) length(unique(x)) == 1)
data_balanced_sample_cleaned <- data_balanced_sample_cleaned[, !constant_columns]
str(data_balanced_sample_cleaned)
# Vérifier les colonnes conservées
print(names(data_balanced_sample_cleaned))
print(names(data_balanced))

# Calcul de la matrice de corrélation
cor_matrix <- cor(data_balanced_sample_cleaned[, sapply(data_balanced_sample_cleaned, is.numeric)])
print(cor_matrix)
# Visualisation avec ggplot
library(ggcorrplot)
ggcorrplot(cor_matrix, lab = TRUE, title = "Corrélation des variables")
label_column <- data_balanced_sample_cleaned$label  # Sauvegarder la colonne cible
high_correlation <- findCorrelation(cor_matrix, cutoff = 0.9)  # Choix du seuil
print(high_correlation)
data_balanced_sample_cleaned <- data_balanced_sample_cleaned[, -high_correlation]
data_balanced_sample_cleaned$label <- label_column
colnames(data_balanced_sample_cleaned)
cor_matrix <- cor(data_balanced_sample_cleaned[, sapply(data_balanced_sample_cleaned, is.numeric)])
print(cor_matrix)


# Supprimer la colonne "word_count" en Base R
data_balanced_sample_cleaned <- data_balanced_sample_cleaned[, !names(data_balanced_sample_cleaned) %in% "word_count"]
colnames(data_balanced_sample_cleaned)

# Appliquer LDA
lda_model <- lda(label ~ ., data = data_balanced_sample_cleaned)

# Résumé du modèle
lda_model

# Visualiser la projection LDA
lda_projection <- predict(lda_model)$x
lda_df <- data.frame(lda_projection, label = data_balanced_sample_cleaned$label)

# Graphe des projections LDA
ggplot(lda_df, aes(x = LD1, color = label)) +
  geom_histogram(bins = 30, alpha = 0.7) +
  labs(title = "Distribution de la première composante LDA",
       x = "LD1 (Composante 1)", y = "Fréquence") +
  theme_minimal()







colnames(data_balanced_sample_cleaned)
# Sélectionner uniquement les colonnes pertinentes : TTR et label
data_balanced_selected <- data_balanced_sample_cleaned[, c("label", "TTR","ai_word_count","perplexity","sentence_count")]
str(data_balanced_selected)
# Convertir la variable label en facteur (si ce n'est pas déjà fait)
data_balanced_selected$label <- as.factor(data_balanced_selected$label)
####
# Configuration de la validation croisée stratifiée k-fold
set.seed(123)
train_control <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Correction explicite pour caret (labels AI vs Human)
data_balanced_selected$label <- as.factor(ifelse(data_balanced_selected$label == 1, "AI", "Human"))

# Entraînement du modèle Naïve Bayes avec validation croisée
nb_model_cv <- train(
  label ~ ., 
  data = data_balanced_selected,
  method = "naive_bayes",
  trControl = train_control,
  metric = "ROC"
)

# Résumé du modèle après validation croisée
print(nb_model_cv)

test_data$TTR <- sapply(strsplit(test_data$text, " "), function(words) length(unique(words)) / length(words))
test_data$ai_word_count <- sapply(test_data$text, count_word_occurrences, word_list = top_words_ai)
test_data$perplexity <- sapply(test_data$text, compute_perplexity)
test_data$sentence_count <- str_count(test_data$text, "[.!?]")  # Nombre de phrases
# Appliquer le modèle sur le jeu de test
test_data$label <- as.factor(ifelse(test_data$label == 1, "AI", "Human"))

test_predictions <- predict(nb_model_cv, newdata = test_data)
test_probabilities <- predict(nb_model_cv, newdata = test_data, type = "prob")

# Matrice de confusion
conf_matrix_test <- confusionMatrix(test_predictions, test_data$label)
print(conf_matrix_test)

# Courbe ROC
roc_curve_test <- roc(test_data$label, test_probabilities[, "AI"])
plot(roc_curve_test, main = "Courbe ROC (Test Set)", col = "darkgreen")

# Calcul de l'AUC
auc_test <- auc(roc_curve_test)
cat("\nAUC (Test Set):", auc_test)
brier_score_test <- mean((test_probabilities[, 2] - (test_data$label == 1))^2)
cat("\nBrier Score (Test Set):", brier_score_test)

