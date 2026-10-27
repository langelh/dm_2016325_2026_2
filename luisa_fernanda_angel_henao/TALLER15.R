# ============================================================
# Práctica: Random Forest y métodos de ensamble 
# ============================================================
paquetes <- c("rpart", "randomForest", "pROC", "ggplot2")
for (p in paquetes) if (!requireNamespace(p, quietly = TRUE)) install.packages(p)
library(rpart); library(randomForest); library(pROC); library(ggplot2)

# ============================================================
# 0. Generación de datos reproducibles
# ============================================================
set.seed(2026)
n <- 1000
historial <- sample(c("bueno", "regular", "malo"), n,
                    replace = TRUE, prob = c(.55, .30, .15))
empleo  <- sample(c("formal", "informal"), n,
                  replace = TRUE, prob = c(.60, .40))
estrato <- sample(1:6, n, replace = TRUE,
                  prob = c(.18, .32, .28, .13, .06, .03))
ingreso <- round(exp(rnorm(n, log(1.2 + 0.5 * estrato),
                           0.35)), 2)
monto   <- round(runif(n, 1, 25), 1)
edad    <- sample(20:70, n, replace = TRUE)
antiguedad <- pmin(edad - 18, rpois(n, 6))
cod_asesor <- sample(1:60, n, replace = TRUE)

carga <- monto / (12 * ingreso)  # crédito vs. ingreso anual
z <- -3.0 + 2.2 * (historial == "malo") +
  1.0 * (historial == "regular") +
  1.0 * (empleo == "informal") + 3.0 * (carga > 0.5) +
  0.8 * (edad < 26) - 0.10 * antiguedad
mora <- ifelse(runif(n) < plogis(z), "Si", "No")
cooperativa <- data.frame(ingreso, monto, edad, antiguedad,
                          estrato, empleo, historial,
                          cod_asesor, mora)
dir.create("data", showWarnings = FALSE)
write.csv(cooperativa, "data/cooperativa.csv",
          row.names = FALSE)

# ---------------- Lectura ----------------
datos <- read.csv("data/cooperativa.csv", stringsAsFactors = TRUE)
datos$mora <- factor(datos$mora, levels = c("No", "Si"))
niveles <- levels(datos$mora); neg <- "No"; pos <- "Si"
form <- mora ~ .
p_pred <- ncol(datos) - 1   


vars_regla <- c("historial", "empleo", "monto", "ingreso", "edad", "antiguedad")

# ============================================================
# a. Tasa de mora y exactitud de predecir siempre "No"
# ============================================================
tasa_mora <- mean(datos$mora == pos)
acc_base  <- mean(datos$mora == neg)
cat("\n[a] Tasa de mora:", round(tasa_mora, 4),
    "| Exactitud de predecir siempre 'No':", round(acc_base, 4), "\n")

# ============================================================
# b. Partición 70/30 (semilla 2026) y comparación de modelos
# ============================================================
set.seed(2026)
idx   <- sample(seq_len(nrow(datos)), size = round(0.7 * nrow(datos)))
train <- datos[idx, ]
test  <- datos[-idx, ]

metricas <- function(prob, real, umbral = 0.5) {
  pred <- factor(ifelse(prob >= umbral, pos, neg), levels = niveles)
  c(Exactitud = mean(pred == real),
    AUC = as.numeric(auc(roc(real, prob, levels = niveles, quiet = TRUE))))
}

# Árbol sin restricciones
arbol_full <- rpart(form, data = train, method = "class",
                    control = rpart.control(cp = 0, minsplit = 2,
                                            minbucket = 1, xval = 10))
p_full <- predict(arbol_full, test, type = "prob")[, pos]

# Árbol podado (cp con mínimo error de validación cruzada)
cp_opt <- arbol_full$cptable[which.min(arbol_full$cptable[, "xerror"]), "CP"]
arbol_pod <- prune(arbol_full, cp = cp_opt)
p_pod <- predict(arbol_pod, test, type = "prob")[, pos]

# Bosque de 500 árboles
set.seed(2026)
rf <- randomForest(form, data = train, ntree = 500, importance = TRUE)
p_rf <- predict(rf, test, type = "prob")[, pos]

tabla_b <- rbind(
  "Árbol sin restricciones" = metricas(p_full, test$mora),
  "Árbol podado"            = metricas(p_pod,  test$mora),
  "Bosque (500 árboles)"    = metricas(p_rf,   test$mora)
)
cat("\n[b] Comparación en prueba:\n"); print(round(tabla_b, 4))

# ============================================================
# c. Error OOB vs número de árboles
# ============================================================
oob <- rf$err.rate[, "OOB"]
df_oob <- data.frame(arboles = seq_along(oob), oob = oob)
print(
  ggplot(df_oob, aes(arboles, oob)) +
    geom_line(color = "darkgreen") +
    labs(title = "Error OOB vs número de árboles",
         x = "Número de árboles", y = "Error OOB") +
    theme_minimal()
)

# Estabilización: desde qué árbol el OOB queda a +/- 0.01 del valor final
tol <- 0.01
fuera <- which(abs(oob - mean(tail(oob, 100))) > tol)
n_estable <- if (length(fuera) == 0) 1 else max(fuera) + 1
cat("\n[c] El error OOB se estabiliza aprox. desde", n_estable, "árboles\n")

# ============================================================
# d. Variar mtry de 1 a 8 (mtry = 8 es Bagging)
# ============================================================
res_d <- data.frame()
for (m in 1:p_pred) {
  set.seed(2026)
  rf_m <- randomForest(form, data = train, ntree = 500, mtry = m)
  pr   <- predict(rf_m, test, type = "prob")[, pos]
  mt   <- metricas(pr, test$mora)
  res_d <- rbind(res_d, data.frame(mtry = m,
                                   OOB = tail(rf_m$err.rate[, "OOB"], 1),
                                   Exactitud = mt["Exactitud"],
                                   AUC = mt["AUC"]))
}
rownames(res_d) <- NULL
cat("\n[d] Resultados por mtry (mtry = 8 es Bagging):\n"); print(round(res_d, 4))
cat("Mejor mtry por AUC:", res_d$mtry[which.max(res_d$AUC)],
    "| por OOB:", res_d$mtry[which.min(res_d$OOB)], "\n")

print(
  ggplot(res_d, aes(mtry)) +
    geom_line(aes(y = AUC, color = "AUC")) + geom_point(aes(y = AUC, color = "AUC")) +
    geom_line(aes(y = 1 - OOB, color = "1 - OOB")) +
    geom_point(aes(y = 1 - OOB, color = "1 - OOB")) +
    scale_x_continuous(breaks = 1:p_pred) +
    labs(title = "Desempeño según mtry", y = "Valor", color = "") +
    theme_minimal()
)

# ============================================================
# e. Importancia Gini vs permutación
# ============================================================
imp <- importance(rf)
imp_df <- data.frame(variable = rownames(imp),
                     Gini = imp[, "MeanDecreaseGini"],
                     Permutacion = imp[, "MeanDecreaseAccuracy"])
imp_df$rank_Gini <- rank(-imp_df$Gini)
imp_df$rank_Perm <- rank(-imp_df$Permutacion)
imp_df <- imp_df[order(imp_df$rank_Gini), ]
cat("\n[e] Importancia de variables:\n"); print(imp_df, row.names = FALSE)
cat("\ncod_asesor (ruido puro, sin efecto en la regla) -> ranking Gini:",
    imp_df$rank_Gini[imp_df$variable == "cod_asesor"],
    "| ranking permutación:",
    imp_df$rank_Perm[imp_df$variable == "cod_asesor"], "\n")

varImpPlot(rf, main = "Importancia de variables (Gini y permutación)")

# ============================================================
# f. ¿El bosque recupera las variables de la regla?
# ============================================================
cat("\n[f] Posición de las variables de la regla:\n")
imp_regla <- imp_df[imp_df$variable %in% vars_regla, ]
print(imp_regla, row.names = FALSE)
cat("Variable de la regla que más le cuesta (menor importancia por permutación):",
    as.character(imp_regla$variable[which.min(imp_regla$Permutacion)]), "\n")

# Extra: la regla usa carga = monto / (12 * ingreso), variable que el bosque
# no ve directamente. Se compara agregándola como predictor.
train2 <- train; test2 <- test
train2$carga <- train2$monto / (12 * train2$ingreso)
test2$carga  <- test2$monto  / (12 * test2$ingreso)
set.seed(2026)
rf2 <- randomForest(form, data = train2, ntree = 500, importance = TRUE)
cat("\nAUC sin 'carga':", round(metricas(p_rf, test$mora)["AUC"], 4),
    "| AUC con 'carga':",
    round(metricas(predict(rf2, test2, type = "prob")[, pos], test2$mora)["AUC"], 4), "\n")
print(round(importance(rf2)[, c("MeanDecreaseGini", "MeanDecreaseAccuracy")], 3))

# ============================================================
# g. Umbral 0.5 y umbral alternativo para la cooperativa
# ============================================================
eval_umbral <- function(prob, real, u) {
  pred <- factor(ifelse(prob >= u, pos, neg), levels = niveles)
  tp <- sum(pred == pos & real == pos); fn <- sum(pred == neg & real == pos)
  fp <- sum(pred == pos & real == neg); tn <- sum(pred == neg & real == neg)
  c(Umbral = u,
    Detecta_morosos = tp / (tp + fn),
    Especificidad   = tn / (tn + fp),
    Precision       = ifelse(tp + fp == 0, NA, tp / (tp + fp)),
    Exactitud       = (tp + tn) / length(real),
    Falsos_positivos = fp, Falsos_negativos = fn)
}

umbrales <- c(0.5, 0.4, 0.3, 0.25, 0.2, 0.15, 0.1)
tabla_g <- do.call(rbind, lapply(umbrales, function(u) eval_umbral(p_rf, test$mora, u)))
cat("\n[g] Efecto del umbral:\n"); print(round(tabla_g, 4))
cat("\nCon umbral 0.5 el bosque detecta",
    round(100 * tabla_g[1, "Detecta_morosos"], 2), "% de los morosos.\n")

# Umbral propuesto: el más alto que detecte al menos 70% de los morosos
meta <- 0.70
cand <- tabla_g[tabla_g[, "Detecta_morosos"] >= meta, , drop = FALSE]
u_prop <- if (nrow(cand) > 0) max(cand[, "Umbral"]) else min(umbrales)
cat("Umbral propuesto:", u_prop, "\n")
print(round(tabla_g[tabla_g[, "Umbral"] == u_prop, ], 4))

roc_rf <- roc(test$mora, p_rf, levels = niveles, quiet = TRUE)
plot(roc_rf, print.auc = TRUE, main = "Curva ROC - Bosque aleatorio")
points(1 - tabla_g[, "Especificidad"], tabla_g[, "Detecta_morosos"], pch = 19, col = "red")
text(1 - tabla_g[, "Especificidad"], tabla_g[, "Detecta_morosos"],
     labels = tabla_g[, "Umbral"], pos = 4, cex = 0.7)