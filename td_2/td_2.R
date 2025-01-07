MANOVA <- function(data) {  
  #effacer toutes les variables en mémoire
  rm(list=ls())
  
  #permet de réaliser des “beaux” tableaux au format html ou pdf.
  library(kableExtra)
  library(dplyr)
  
  
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
  
  # Informations sur les données
  N <- nrow(data)  # Nombre d'individus
  P <- ncol(data) - 1  # Nombre de variables prédictives (sans la colonne Class)
  K <- length(unique(data$Class))  # Nombre de groupes
  print(c(N, P, K))
  
  # Séparation des variables prédictives (X) et catégorielles (Y)
  X <- data[, 1:P]
  Y <- data$Class
  
  # Vérification des classes des colonnes
  sapply(data, class)
  
  # Moyenne générale des variables
  G <- colMeans(X)
  
  # Calcul des inerties partielles intraclasse
  XK <- split(seq_len(N), Y)  # Liste des indices par groupe
  GK <- lapply(split(X, data$Class), colMeans)  # Moyennes par groupe
  #vecteur qui correspond aux nombre d'individu par groupe
  NK<-table(data$Class)
  SS_partiel_Intra <- function(GK, X, XK) {
    ss <- numeric(length(GK))
    for (k in seq_along(GK)) {
      indices <- XK[[k]]
      group_data <- X[indices, ]
      for (i in seq_len(nrow(group_data))) {
        diff <- group_data[i, ] - GK[[k]]  # Différence par rapport au centre du groupe
        ss[k] <- ss[k] + sum(diff^2)  # Somme des carrés des distances
      }
    }
    return(ss)
  }
  
  SS_intra_partiel <- SS_partiel_Intra(GK, X, XK)
  SS_Intra <- sum(SS_intra_partiel)
  SS_tot <- sum(rowSums((X - matrix(rep(G, times = N), nrow = N, byrow = TRUE))^2))
  SS_Inter <- SStot - SS_Intra
  
  # Vérification des inerties
  print(c(SStot, SS_Intra, SS_Inter))
  
  # Inférence statistique
  lambda <- abs(SS_Intra) / abs(SS_Intra + SS_Inter)
  corr <- (N - 1 - (P + K) / 2) * log(lambda)
  valeur_critique <- qchisq(0.95, df = P * (K - 1))
  
  print(c(lambda, corr, valeur_critique))
  # Calcul de la probabilité associée au test de Wilks Lambda
  # On utilise ici la distribution F pour obtenir la p-value
  v <- N - P - 1
  u <- P * (K - 1)
  F_stat <- ((1 - lambda^(1 / u)) / lambda^(1 / u)) * (v / P)
  p_value <- pf(F_stat, df1 = P * (K - 1), df2 = v, lower.tail = FALSE)
  
  # Application de la MANOVA
  validation <- manova(cbind(Al, Fe, Mg, Ca, Na) ~ Class, data = data)
  summary(validation)
  # Création de la liste contenant toutes les informations
  result <- list(
    SS_tot = SS_tot,
    SS_Intra = SS_Intra,
    SS_Inter = SS_Inter,
    Gk = GK,
    G = G,
    NK = NK,
    P = P,
    N = N,
    Lambda = lambda,
    p_value = p_value
  )
  
  return(result)
}
#charger datasets
data<-read.csv("G:/Mon Drive/E4/Fouille_de_donnees_avec_R/td_2/MANOVA_DATASET(in).csv",header=TRUE,sep=",",fill=TRUE,quote="",check.names = FALSE )
results<-MANOVA(data)
print(results)
