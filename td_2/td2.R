
# #effacer toutes les variables en mémoire
rm(list=ls()) 

#permet de réaliser des “beaux” tableaux au format html ou pdf.
library(kableExtra)
library(dplyr)


MANOVA <- function(data) {  
  # Convertir les colonnes numériques si nécessaire
  data[, 1:(ncol(data) - 1)] <- lapply(data[, 1:(ncol(data) - 1)], as.numeric)
  
  # Informations sur les données
  N <- nrow(data)  # Nombre d'individus
  
  P <- ncol(data) - 1  # Nombre de variables prédictives (sans la colonne Class)
  
  X <- data[, 1:P]
  
  Y <- data[,P]
  
  K <- length(unique(data$Y))  # Nombre de groupes
  
  
  # Séparation des variables prédictives (X) et catégorielles (Y)
  
  print(Y)
  
  # Moyenne générale des variables
  G <- colMeans(X)
  print(G)
  # Calcul des inerties partielles intra-classe
  XK <- split(seq_len(N), Y)  # Liste des indices par groupe
  print(XK)
  GK <- lapply(split(X, Y), colMeans)  # Moyennes par groupe
  print(GK)
  # Matrice des sommes des carrés intra-classe
  SS_Intra <- matrix(0, ncol = P, nrow = P)  # Initialisation
  for (k in seq_along(GK)) {
    indices <- XK[[k]]  # Indices du groupe k
    group_data <- X[indices, ]  # Données du groupe k
    
    # Contribution du groupe k à la matrice intra-classe
    for (i in seq_len(nrow(group_data))) {
      diff <- as.numeric(group_data[i, ] - GK[[k]])  # Différence par rapport à la moyenne du groupe
      diff <- matrix(diff, ncol = 1)  # Convertir en matrice colonne
      SS_Intra <- SS_Intra + diff %*% t(diff)  # Produit matriciel
    }
  }

  # Matrice des sommes des carrés totales
  SS_tot <- sum(rowSums((X - matrix(rep(G, times = N), nrow = N, byrow = TRUE))^2))
  

  # Matrice des sommes des carrés inter-classes
  SS_Inter <- SS_tot - sum(diag(SS_Intra))  # Diagonale pour obtenir SS_Intra scalaire
  
  # Vérification des inerties
  print(c(SS_tot, SS_Inter, sum(diag(SS_Intra))))
  
  # Inférence statistique
  lambda <- abs(sum(diag(SS_Intra))) / abs(SS_tot)
  corr <- (N - 1 - (P + K) / 2) * log(lambda)
  valeur_critique <- qchisq(0.05, df = K * (P - 1))
  
  print(c(lambda, corr, valeur_critique))
  
  # Calcul de la probabilité associée au test de Wilks Lambda
  v <- N - P - 1
  u <- P * (K - 1)
  F_stat <- ((1 - lambda^(1 / u)) / lambda^(1 / u)) * (v / P)
  p_value <- pf(F_stat, df1 = P * (K - 1), df2 = v, lower.tail = FALSE)
  
  
  # Création de la liste contenant toutes les informations
  result <- list(
    SS_tot = SS_tot,
    SS_Intra = SS_Intra,
    SS_Inter = SS_Inter,
    Gk = GK,
    G = G,
    NK = table(data$Y),
    P = P,
    N = N,
    Lambda = lambda,
    p_value = p_value
  )
  
  return(result)
}

#Application pour notre fichier csv :

#charger datasets
data<-read.csv("MANOVA_DATASET(in).csv",header=TRUE,sep=",",fill=TRUE,quote="",check.names = FALSE )

#mise en forme du tableau
data$Class <- substr(data$`"Site`, 1, 2)

# Suppression de la colonne "Site"
data <- data[, -1]

# Renommer les colonnes pour une meilleure lisibilité
colnames(data) <- c("Al", "Fe", "Mg", "Ca", "Na", "Class")

# Transformation de la variable "Class" en facteur
data$Class <- as.factor(data$Class)

# Affichage des premières lignes sous forme de tableau HTML
data %>%
  kbl(digits = 3) %>%
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center",
                latex_options = 'stripped') %>%
  scroll_box(height = "250px")
results <- MANOVA(data)
print(results)
# vaidation
validation <- manova(cbind(Al, Fe, Mg, Ca, Na) ~ Class, data = data)
summary_manova <- summary(validation, test = "Wilks")
print(summary_manova)


#tester avec le jeu de données iris
# Charger le jeu de données iris
data(iris)

# Convertir les espèces en facteur pour les utiliser dans la MANOVA
iris$Species <- as.factor(iris$Species)

# Renommer les colonnes pour la lisibilité
colnames(iris) <- c("Sepal.Length", "Sepal.Width", "Petal.Length", "Petal.Width", "Species")

# Appliquer la fonction MANOVA sur le jeu de données iris
results_iris <- MANOVA(iris)

# Affichage des résultats
print(results_iris)

# Application de la MANOVA avec la fonction manova de base de R
manova_res_iris <- manova(cbind(Sepal.Length, Sepal.Width, Petal.Length, Petal.Width) ~ Species, data = iris)

# Résumé des résultats de manova
summary_manova_iris <- summary(manova_res_iris, test = "Wilks")
print(summary_manova_iris)

# Extraire et afficher la valeur de Wilks' Lambda calculée par manova()
lambda_wilks_iris <- summary_manova_iris$stats[1, 2]  # Récupérer la valeur de Wilks' Lambda
cat("Wilks' Lambda calculé par manova sur iris:", lambda_wilks_iris, "\n")

#appliquer notre fonction MANOVA
results <- MANOVA(data)
print(results)

