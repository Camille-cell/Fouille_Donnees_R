rm(list=ls())
library(ggplot2)
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
  
 
  # liste contenant tous les résultats
  result <- list(
    SS_tot = SS_tot,
    SS_Intra = SS_Intra,
    SS_Inter = SS_Inter
    
  )
  return(result)
}




AFD <- function(X, Y, SS_tot, SS_intra, SS_inter, nb_axes = 2) {
  # Vérification que SS_intra est inversible
  if (det(SS_intra) == 0) {
    stop("Erreur : La matrice SS_intra est singulière et ne peut pas être inversée.")
  }
  
  # Calcul du rapport SS_inter / SS_intra
  Ratio <- solve(SS_intra) %*% SS_inter
  
  # Diagonalisation
  eigens <- eigen(Ratio)
  
  # Extraction des valeurs propres (réelles uniquement)
  valeurs_propres <- Re(eigens$values)
  
  # Suppression des valeurs propres nulles ou très faibles
  valeurs_propres[valeurs_propres < 1e-7] <- 0
  
  # Calcul des vecteurs propres normalisés
  U <- Re(eigens$vectors)
  W <- diag(ncol(U))  # Matrice d'identité
  U_transpose_W_U <- t(U) %*% W %*% U
  normalization_factors <- sqrt(diag(U_transpose_W_U))
  normalization_factors[normalization_factors == 0] <- 1  # Éviter division par zéro
  Un <- sweep(U, 2, normalization_factors, "/")  # Normalisation colonne par colonne
  
  # Sélection des nb_axes premiers axes
  lambda_selected <- valeurs_propres[1:nb_axes]
  U_selected <- Un[, 1:nb_axes]
  
  # Calcul des scores des individus sur les axes factoriels
  Z <- scale(X, center = TRUE, scale = FALSE)  # Centrage des données
  Scores <- Z %*% U_selected  # Projection des individus
  
  # Création du data frame des scores
  Scores_df <- data.frame(Scores, Qualite = Y)
  
  # Test de Wilks
  corr <- lambda_selected / (1 + lambda_selected)
  wilks_lambda <- prod(1 - corr^2)
  
  if (wilks_lambda <= 0) {
    statistic <- NA
    p_value <- NA
  } else {
    n <- nrow(X)
    p <- ncol(X)
    k <- length(unique(Y))
    statistic <- -((n - 1 - p + k) / 2) * log(wilks_lambda)
    ddl <- p * (k - 1)
    p_value <- pchisq(statistic, df = ddl, lower.tail = FALSE)
  }
  
  # Résumé des résultats du test de Wilks
  wilks_results <- data.frame(
    Wilks_Lambda = wilks_lambda,
    Chi2_Stat = statistic,
    df = ddl,
    p_value = p_value
  )
  
  # Retour des résultats sous forme de liste
  return(list(
    Vecteurs_Propres = as.data.frame(Un),
    Valeurs_Propres = as.data.frame(valeurs_propres),
    Scores = Scores_df,
    Test_Wilks = wilks_results
  ))
}


# Fonction AFD_graph1 pour afficher le plan factoriel avec les couleurs et les centres de gravité
AFD_graph1 <- function(Scores, label = T, center = F) { 
  colnames(Scores)[3] <- 'Qualite'  
  print(head(Scores))  # Vérification du data.frame
  
  gr <- ggplot()  # Initialisation du graphique
  
  if (label == F) {
    gr <- gr + geom_point(data = Scores, aes(x = Axe_1, y = Axe_2), colour = '#0066CC', size = 1)
  } else {
    gr <- gr + geom_point(data = Scores, aes(x = Axe_1, y = Axe_2, colour = Qualite), size = 2)
  }
  
  # Ajouter les centres de gravité des classes si demandé
  if (center == T) {
    df_center <- aggregate(Scores[,-3], by = list(Scores$Qualite), mean)
    gr <- gr + geom_point(data = df_center, aes(x = Axe_1, y = Axe_2, colour = Group.1), shape = 23, size = 2)
  }
  
  # Ajouter les lignes horizontales et verticales (axes)
  gr <- gr + geom_hline(yintercept = 0 , size = 0.1, colour = 'red') +
    geom_vline(xintercept = 0 , size = 0.1, colour = 'red') 
  
  return(gr)  # Retourne le graphique
}





data(iris)
X <- iris[, 1:4]  # Variables prédictives (longueur et largeur des pétales et sépales)
Y <- iris$Species  # Variable à prédire (3 espèces de fleurs)



# Exécuter la fonction MANOVA pour obtenir SS_tot, SS_intra et SS_inter
result_MANOVA <- MANOVA(cbind(X, Y))

# Exécuter la fonction AFD
result_AFD <- AFD(
  X = X, 
  Y = Y, 
  SS_tot = result_MANOVA$SS_tot, 
  SS_intra = result_MANOVA$SS_Intra, 
  SS_inter = result_MANOVA$SS_Inter, 
  nb_axes = 2
)

# Affichage des résultats
print(result_AFD$Valeurs_Propres)  # Afficher les valeurs propres
print(result_AFD$Vecteurs_Propres) # Afficher les vecteurs propres (loading factors)
print(result_AFD$Test_Wilks)       # Afficher le test de Wilks

# Visualisation des résultats dans le plan factoriel
print(AFD_graph1(result_AFD$Scores, center = TRUE))


library(MASS)

# Exécuter l'Analyse Discriminante Linéaire (LDA) avec MASS
lda_model <- lda(Species ~ ., data = iris)

# Comparer les valeurs propres
print("Valeurs propres AFD :")
print(result_AFD$Valeurs_Propres)

print("Valeurs propres LDA :")
print(lda_model$svd^2)  # LDA retourne les racines carrées des valeurs propres, donc on élève au carré.

# Comparer les vecteurs propres (coefficients)
print("Vecteurs propres AFD :")
print(result_AFD$Vecteurs_Propres)

print("Coefficients de LDA :")
print(lda_model$scaling)

# Comparer les scores
scores_lda <- as.matrix(X) %*% lda_model$scaling
scores_afd <- result_AFD$Scores[, 1:2]  # On prend les 2 premières colonnes

print("Scores AFD :")
print(head(scores_afd))

print("Scores LDA :")
print(head(scores_lda))

# Test de Wilks avec la fonction manova
manova_res <- manova(as.matrix(X) ~ Y)
wilks_test <- summary(manova_res, test = "Wilks")

print("Test de Wilks - AFD :")
print(result_AFD$Test_Wilks)

print("Test de Wilks - manova() :")
print(wilks_test)

# Appliquation à VIN_QUALITES.txt
data<-read.table("G:/Mon Drive/E4/Fouille_de_donnees_avec_R/td_3/VIN_QUALITE.txt",header = TRUE, sep = "")
str(data)
print(head(data))
print(table(data[, 5]))


result<-MANOVA(data)

# Exécuter la fonction AFD
result_AFD <- AFD(
  X = X, 
  Y = Y, 
  SS_tot = result_MANOVA$SS_tot, 
  SS_intra = result_MANOVA$SS_Intra, 
  SS_inter = result_MANOVA$SS_Inter, 
  nb_axes = 2
)

# Affichage des résultats
print(result_AFD$Valeurs_Propres)  # Afficher les valeurs propres
print(result_AFD$Vecteurs_Propres) # Afficher les vecteurs propres (loading factors)
print(result_AFD$Test_Wilks)       # Afficher le test de Wilks

# Visualisation des résultats dans le plan factoriel
print(AFD_graph1(result_AFD$Scores, center = TRUE))


