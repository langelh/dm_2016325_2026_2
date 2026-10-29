install.packages(c("rpart", "randomForest", "xgboost"))
library(rpart)
library(randomForest)
library(xgboost)

#  Datos 
set.seed(42); n <- 600
df <- data.frame(
  turbiedad  = rexp(n, 0.1),
  pH         = rnorm(n, 7.2, 0.8),
  coliformes = rpois(n, 80),
  DBO        = rexp(n, 0.2),
  poblacion  = sample(5000:500000, n, replace = TRUE),
  altitud    = runif(n, 100, 3000)
)
df$apta <- as.integer(df$turbiedad < 5 & df$pH > 6.5 & df$pH < 8.5 &
                        df$coliformes < 100 & df$DBO < 5)
vars <- setdiff(names(df), "apta")
cat("Proporción de aptas:", round(mean(df$apta), 3), "\n")

# Funciones auxiliares 
auc <- function(y, p) {                      # AUC por rangos (Mann-Whitney)
  r <- rank(p); n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
metricas <- function(y, p, umbral = 0.5) {
  pred <- as.integer(p >= umbral)
  tp <- sum(pred == 1 & y == 1); fp <- sum(pred == 1 & y == 0); fn <- sum(pred == 0 & y == 1)
  f1 <- if (tp == 0) 0 else 2 * tp / (2 * tp + fp + fn)
  c(accuracy = mean(pred == y), f1 = f1, auc = auc(y, p))
}
folds_estrat <- function(y, k = 5) {         # pliegues estratificados
  f <- integer(length(y))
  for (c in unique(y)) {
    i <- which(y == c)
    f[i] <- sample(rep(1:k, length.out = length(i)))
  }
  f
}

# entrena con 'tr' y devuelve P(apta = 1) en 'te'
fit_tree <- function(tr, te) {
  m <- rpart(factor(apta) ~ ., data = tr, method = "class",
             control = rpart.control(maxdepth = 5))
  predict(m, te, type = "prob")[, "1"]
}
fit_rf <- function(tr, te) {
  m <- randomForest(factor(apta) ~ ., data = tr, ntree = 200)
  predict(m, te, type = "prob")[, "1"]
}
fit_xgb <- function(tr, te, nrounds = 200, eta = 0.1) {
  dtr <- xgb.DMatrix(as.matrix(tr[, vars]), label = tr$apta)
  m <- xgb.train(params = list(objective = "binary:logistic", eval_metric = "logloss",
                               eta = eta, max_depth = 4),
                 data = dtr, nrounds = nrounds, verbose = 0)
  predict(m, as.matrix(te[, vars]))
}

# (a) Comparación con CV de 5 pliegues
set.seed(42)
fold <- folds_estrat(df$apta, 5)
modelos <- list("Árbol" = fit_tree, "Random Forest" = fit_rf, "XGBoost" = fit_xgb)

cv_modelo <- function(fit_fun) {
  res <- t(sapply(1:5, function(k) {
    tr <- df[fold != k, ]; te <- df[fold == k, ]
    metricas(te$apta, fit_fun(tr, te))
  }))
  colMeans(res)
}
tabla <- t(sapply(modelos, cv_modelo))
cat("\n(a) Validación cruzada 5 pliegues\n")
print(round(tabla, 3))

#  (b) Early stopping en XGBoost (paciencia = 20) 
set.seed(42)
dall <- xgb.DMatrix(as.matrix(df[, vars]), label = df$apta)
cv_es <- xgb.cv(params = list(objective = "binary:logistic", eval_metric = "logloss",
                              eta = 0.05, max_depth = 4),
                data = dall, nrounds = 1000, nfold = 5, stratified = TRUE,
                early_stopping_rounds = 20, verbose = 0)
log_ev <- as.data.frame(cv_es$evaluation_log)
n_opt <- which.min(log_ev$test_logloss_mean)
cat("\n(b) Número óptimo de árboles:", n_opt, "\n")
cat("    Logloss de validación en el óptimo:", round(min(log_ev$test_logloss_mean), 4), "\n")

# ---------- (c) Importancia: RF (impureza) vs XGBoost (Gain) ----------
set.seed(42)
rf_full <- randomForest(factor(apta) ~ ., data = df, ntree = 200, importance = TRUE)
imp_rf <- importance(rf_full)[, "MeanDecreaseGini"]
imp_rf <- imp_rf / sum(imp_rf)

xgb_full <- xgb.train(params = list(objective = "binary:logistic", eval_metric = "logloss",
                                    eta = 0.05, max_depth = 4),
                      data = dall, nrounds = n_opt, verbose = 0)
imp_x <- as.data.frame(xgb.importance(model = xgb_full))
gain <- setNames(rep(0, length(vars)), vars)
gain[imp_x$Feature] <- imp_x$Gain

imp <- data.frame(RF_impureza = imp_rf[vars], XGB_gain = gain[vars])
imp$rank_RF  <- rank(-imp$RF_impureza)
imp$rank_XGB <- rank(-imp$XGB_gain)
cat("\n(c) Importancias (normalizadas)\n")
print(round(imp[order(-imp$RF_impureza), ], 3))
top3_rf  <- rownames(imp)[order(-imp$RF_impureza)][1:3]
top3_xgb <- rownames(imp)[order(-imp$XGB_gain)][1:3]
cat("¿Mismo top-3?:", setequal(top3_rf, top3_xgb), "\n")

barplot(t(as.matrix(imp[order(imp$RF_impureza), c("RF_impureza", "XGB_gain")])),
        beside = TRUE, horiz = TRUE, las = 1, col = c("steelblue", "darkorange"),
        legend.text = c("RF (impureza)", "XGBoost (Gain)"),
        args.legend = list(x = "bottomright"),
        main = "Importancia de variables")

# (d) Stacking: árbol + RF, meta-modelo logística (con OOF)
fit_stack <- function(tr, te, k_int = 5) {
  # 1) Predicciones OOF de los modelos base sobre el set de entrenamiento
  f_int <- folds_estrat(tr$apta, k_int)
  oof <- data.frame(tree = numeric(nrow(tr)), rf = numeric(nrow(tr)))
  for (j in 1:k_int) {
    tr_in <- tr[f_int != j, ]; va_in <- tr[f_int == j, ]
    oof$tree[f_int == j] <- fit_tree(tr_in, va_in)
    oof$rf[f_int == j]   <- fit_rf(tr_in, va_in)
  }
  # 2) Meta-modelo logístico entrenado sobre las OOF
  meta <- suppressWarnings(glm(apta ~ tree + rf, family = binomial,
                               data = cbind(apta = tr$apta, oof)))
  # 3) Modelos base reentrenados con todo el train -> test -> meta
  base_te <- data.frame(tree = fit_tree(tr, te), rf = fit_rf(tr, te))
  predict(meta, base_te, type = "response")
}
set.seed(42)
res_stack <- t(sapply(1:5, function(k) {
  tr <- df[fold != k, ]; te <- df[fold == k, ]
  metricas(te$apta, fit_stack(tr, te))
}))
tabla <- rbind(tabla, "Stacking (Árbol+RF->LR)" = colMeans(res_stack))
cat("\n(d) Comparación final (CV 5 pliegues)\n")
print(round(tabla, 3))

mejor_ind <- max(tabla[1:3, "auc"])
cat("\n¿Stacking supera al mejor individual en AUC?",
    tabla["Stacking (Árbol+RF->LR)", "auc"] > mejor_ind, "\n")
cat("Modelo recomendado (mayor AUC):", rownames(tabla)[which.max(tabla[, "auc"])], "\n")