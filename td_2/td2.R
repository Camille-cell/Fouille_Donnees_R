
# #effacer toutes les variables en mémoire
rm(list=ls()) 

#permet de réaliser des “beaux” tableaux au format html ou pdf.
library(kableExtra)
library(dplyr)

#chargement des données
data<-read.csv("MANOVA_DATASET(in).csv",header=TRUE,sep=",",fill=TRUE,quote="",check.names = FALSE )

#mise en forme du tableau
data$Class <- substr(data$`"Site`, 1, 2)

# Suppression de la colonne "Site"
data <- data[, -1]

# Renommer les colonnes afin d'être conforme à l'exercice
colnames(data) <- c("Al", "Fe", "Mg", "Ca", "Na", "Class")

# Transformation de la variable "Class" en facteur
data$Class <- as.factor(data$Class)

# Affichage sous forme de tableau HTML
data %>%
  kbl(digits = 3) %>%
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center",
                latex_options = 'stripped') %>%
  scroll_box(height = "250px")

#Création de la fonction MANOVA qui retourne une liste : list(SS_tot, SS_Intra, SS_Inter, GK,G,N,P,N, Lambda, p_value )
MANOVA <- function(data) { 
  data[, 1:(ncol(data) - 1)] <- lapply(data[, 1:(ncol(data) - 1)], as.numeric)
  
  # Informations sur les données
  N <- nrow(data)  # Nombre de ligne
  
  P <- ncol(data) - 1  # Nombre de variables prédictives
  
  X <- data[, 1:P]
  
  Y <- data[,P+1]
  print(Y)
  
  #K <- length(unique(data$Y))  # Nombre de groupes
  K<-nlevels(Y)
  print(K)
  # Moyenne générale des variables
  G <- colMeans(X)
  print(G)
  
  
  # Calcul des inerties partielles intra-classe
  XK <- split(X, Y)  # Liste des indices par groupe
  print(XK)
  print(names(XK))
  NK <- as.vector(table(data$Class))
  print(NK)
  
  # Calcul de GK : Moyennes des variables par groupe
  GK <- lapply(split(X, Y), colMeans)
  print(GK)
  
  # Transformer X en matrice
  X_matrix <- as.matrix(X)
  
  # Créer une matrice G_matrix dont chaque ligne est le vecteur G
  G_matrix <- matrix(rep(G, times=N), nrow=N, byrow=TRUE)
  
  # Calculer la différence
  diff_matrix <- X_matrix - G_matrix
  print(X_matrix)
  print(G_matrix)
  print(diff_matrix)
  # Calculer SS_tot (produit matriciel)
  SS_tot <- t(diff_matrix) %*% diff_matrix
  
  # Afficher SS_tot
  print("Matrice des sommes des carrés totales (SS_tot) :")
  print(SS_tot)
  
  # Initialiser une liste vide pour stocker les inerties intra-partielles
  SS_partiel_Intra <- list()
  
  # Itérer sur chaque groupe
  for (groupe in names(XK)) {
    # Extraire les individus du groupe
    individus_groupe <- XK[[groupe]]
    
    # Calculer la moyenne du groupe (G_k)
    G_k <- GK[[groupe]]
    
    # Transformer les données du groupe en matrice
    X_groupe <- as.matrix(individus_groupe)
    
    # Créer une matrice où chaque ligne correspond à G_k
    G_k_matrix <- matrix(rep(G_k, times = nrow(X_groupe)), nrow = nrow(X_groupe), byrow = TRUE)
    
    # Calculer la différence
    diff_matrix <- X_groupe - G_k_matrix
    
    # Calculer l'inertie intra-partielle pour le groupe
    SS_groupe <- t(diff_matrix) %*% diff_matrix
    
    # Stocker le résultat dans la liste
    SS_partiel_Intra[[groupe]] <- SS_groupe
  }
  
  # Afficher les inerties intra-partielles
  print("Inerties intra-partielles (SS_partiel_Intra) :")
  print(SS_partiel_Intra)
  
  # Initialiser SS_Intra avec une matrice nulle
  SS_Intra <- matrix(0, nrow = P, ncol = P)
  
  # Additionner les matrices de SS_partiel_Intra
  for (groupe in names(SS_partiel_Intra)) {
    SS_Intra <- SS_Intra + SS_partiel_Intra[[groupe]]
  }
  
  # Afficher l'inertie intra (SS_Intra)
  print("Inertie Intra (SS_Intra) :")
  print(SS_Intra)
  
  # Calcul de l'inertie inter-classe (SS_Inter)
  SS_Inter <- SS_tot - SS_Intra
  
  # Afficher l'inertie inter-classe
  print("Inertie Inter-classe (SS_Inter) :")
  print(SS_Inter)
  
  lambda <- det(SS_Intra) / det(SS_tot)
  corr <- (N - 1 - (P + K) / 2) * log(lambda)
  valeur_critique <-qchisq(0.05,K*(P-1))
  print(K)
  print(P)
  print(c(lambda, corr, valeur_critique))
  
  # Calcul de la probabilité associée au test de Wilks Lambda
  v <- N - P - 1
  u <- P * (K - 1)
  F_stat <- ((1 - lambda^(1 / u)) / lambda^(1 / u)) * (v / P)
  p_value <- pf(F_stat, df1 = P * (K - 1), df2 = v, lower.tail = FALSE)
  
  
  # liste contenant tous les résultats
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

