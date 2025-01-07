#effacer toutes les variables en mémoire
rm(list=ls())
#permet de réaliser des “beaux” tableaux au format html ou pdf.
library(kableExtra)
#le package caret pour les matrices de confusion
library('caret')
library(caret)
#le paquest e1071 qui réalise la classification Bayésienne. Les résultats développés dans vos scripts seront comparés à ceux fournis par cette librairie
library('e1071')

df = data.frame('Salaire' = c('< 30', '30 - 50', '30 - 50', '30 - 50', '> 50'),
                
                'Impots' = c('< 20','< 20','< 20','> 20','> 20'),
                
                'Etudiant' = c('Oui','Non','Oui','Non','Non') ,
                
                'Controle' = c('Non','Oui','Oui','Non','Oui')
                
)
#Affichage du tableau avec kableExtra
df %>% kbl(digits=3) %>%    
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped') %>% 
  scroll_box( height = "250px")
# Créer le DataFrame des prédicteurs (X)
X<-df[,-4]
# Créer le vecteur des catégories à prédire (Y)
Y<-df$Controle

X %>% kbl(digits=3) %>%    
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped') %>% 
  scroll_box( height = "250px")

# Créer la liste des tableaux avec lapply
contingence<-lapply(X, function(x){temp_1<-table(Y,x)})
df %>%
  kable() %>%
  kable_styling()
contingence
# Paramètre m pour la correction
m <- 1
k<-length(unique(Y))
k
#Correction des tableaux et calcul des probabilités
prob_cond <- lapply(X, function(x){  
  temp_1 <- table(Y,x)         # tableau initiale 
  s      <- apply(temp_1,1, sum) +  m * k  # calcul des sommes + pondération

  mat    <- matrix(s, ncol = ncol(temp_1), nrow = nrow(temp_1))  # matrice des sommes
  out <- (temp_1 + m)/mat  # calcul des nk = pondération de Laplace
}) # somme des controles

prob_cond 
data_test <- data.frame( 'Salaire' = c('30 - 50', '> 50','< 30'),
                         'Impots'  = c('< 20', '< 20' ,'< 20')  ,
                         'Etudiant' = c('Oui','Non' ,'Oui')
)
data_test %>% kbl(digits=3) %>%    
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped')

priori <- prop.table(table(Y))
priori_labels  <- names(priori)
priori
priori_labels

Test    <- NULL
LnTest  <- NULL
for (i_test in 1:nrow(data_test)){    # sélection des individus du jeux test
  V    <- NULL
  lnV  <- NULL
  for(ic in 1:length(priori))        # sélection des Ck Oui
    {   somme <- 0 ; prod <- 1
        for(i in 1:ncol(X))          # sélection des tableau de proba conditionnelles
          {
            ar    <-  prob_cond[[i]]  # on extrait les tableau des probailiyés conditionnelles de la liste 
            somme <- somme + log( ar[priori_labels[ic],data_test[i_test,i]])
            prod  <- prod *   ar[priori_labels[ic],data_test[i_test,i]]
  
          }
        V     <- c(V, priori[ic] * prod )
        lnV   <- c(lnV, log(priori[ic]) + somme ) 
        }  
}
Test  <- rbind(Test, V)    
LnTest <- rbind(LnTest, lnV)
data.frame(LnTest)%>% kbl(digits=4) %>%    
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped')

resulTest <- data.frame(data_test, 'Pred' = colnames(LnTest)[apply(LnTest,1,which.max)] ) 

resulTest %>% kbl(digits=4) %>%    
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped')


Naive_Bayes <- function(X, Y, m = 1) {
  k <- length(unique(Y)) # Nombre de catégories (labels)
  
  # Calcul des probabilités conditionnelles corrigées
  prob_cond <- lapply(X, function(x) {
    temp_1 <- table(Y, x)         # Tableau initial
    s <- apply(temp_1, 1, sum) + m * k # Somme + correction
    mat <- matrix(s, ncol = ncol(temp_1), nrow = nrow(temp_1))  # Matrice des sommes
    (temp_1 + m) / mat  # Probabilités conditionnelles corrigées
  })
  
  # Calcul des probabilités a priori
  priori <- prop.table(table(Y))
  
  # Résultat
  out <- list('prob_cond' = prob_cond, 'priori' = priori)
  return(out)
}
model_1 <- Naive_Bayes(X, Y, m=1)
print(model_1)

Predict_Bayes <- function(model, X_pred) {
  # Extraire les composantes du modèle
  prob_cond <- model$prob_cond
  priori <- model$priori
  priori_labels <- names(priori)  # Labels des catégories
  
  LnTest <- NULL
  
  for (i_test in 1:nrow(X_pred)) {  # Parcourir les individus du jeu test
    lnV <- NULL
    for (ic in 1:length(priori)) {  # Parcourir les catégories (Oui, Non, etc.)
      somme <- 0
      for (i in 1:ncol(X_pred)) {   # Parcourir les variables
        ar <- prob_cond[[i]]  # Extraire le tableau des probas conditionnelles
        somme <- somme + log(ar[priori_labels[ic], X_pred[i_test, i]])
      }
      lnV <- c(lnV, log(priori[ic]) + somme)
    }
    LnTest <- rbind(LnTest, lnV)
  }
  
  # Identifier les prédictions maximales
  preds <- colnames(LnTest)[apply(LnTest, 1, which.max)]
  result <- data.frame(X_pred, 'Pred' = preds)
  
  return(result)
}

result_pred <- Predict_Bayes(model_1, data_test)

result_pred %>% kbl(digits=3) %>%    
  kable_styling(bootstrap_options = "striped", full_width = F, position = "center", latex_options = 'stripped') %>% 
  scroll_box( height = "250px")

load('data_test.Rda')

X <- data_test[,1:11]
obs <- data_test[,12] 
model <- Naive_Bayes(X, obs, m = 1)
print(model)

pred_1 <- Predict_Bayes(model, X)

model_2 <- naiveBayes(event ~ ., data = data_test)
pred_2 <- predict(model_2, newdata = X) 
confusionMatrix(pred_2, obs)
table(pred_1$pred, obs)