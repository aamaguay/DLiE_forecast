setwd("C:/Users/amaguaya/OneDrive - Kienzle Automotive GmbH/Desktop/DLiE/repo/DLiE_forecast_13_06_25")

##################################################################################################
##################################################################################################
##################################################################################################



library(parallel)
library(arrow)
library(mgcv)
library(lubridate)
library(dplyr)
library(tidyr)

############################ FUNCTIONS #######################################

build_gam_formula <- function(feature_names, apply_smooth_list, apply_cyclic_list, apply_interactions, train_df) {
  
  ytarget <- feature_names[1]
  feature_names <- feature_names[-1]
  terms <- c()
  
  for (name in feature_names) {
    
    unique_vals <- length(unique(train_df[[name]]))
    
    if (name %in% apply_cyclic_list) {
      terms <- c(terms, paste0("ti(", name, ", bs = 'cc')"))
      
    } else if (name %in% apply_smooth_list) {
      if (unique_vals < 5) {
        terms <- c(terms, name)  # Use linear term if not enough unique values
      } else {
        terms <- c(terms, paste0("ti(", name, ", bs = 'cs')"))
      }
      
    } else {
      terms <- c(terms, name)
    }
  }
  
  # Add interactions if provided
  if (!is.null(apply_interactions)) {
    for (triple in apply_interactions) {
      var1 <- triple[1]
      var2 <- triple[2]
      basis <- triple[3]
      terms <- c(terms, paste0("ti(", var1, ", ", var2, ", bs = '", basis, "')"))
    }
  }
  
  formula_str <- paste(ytarget, "~", paste(terms, collapse = " + "))
  return(as.formula(formula_str))
}



############################ PARAMETERS #######################################

INIT_DATE_EXPERIMENTS <- as.Date("2019-01-01")
INIT_TEST_DATE <- as.Date("2023-01-01")
FINAL_DATE_EXPERIMENTS <- as.Date("2024-12-31")

n_days_test <- as.integer(FINAL_DATE_EXPERIMENTS - INIT_TEST_DATE) + 1
full_date_series <- seq.Date(INIT_DATE_EXPERIMENTS, FINAL_DATE_EXPERIMENTS, by = "day")

train_start_idx <- which(full_date_series == INIT_DATE_EXPERIMENTS)
id_init_eval <- which(full_date_series == INIT_TEST_DATE)
id_end_eval <- which(full_date_series == FINAL_DATE_EXPERIMENTS)
N_s <- (id_end_eval - id_init_eval) + 1

LS_VAR_APPLY_SMOOTH_SPLINE <- c("Load_DA_lag_0", "pct_chg_Load_DA", "lag168_Load_DA",
                                "Coal_lag_2", "NGas_lag_2", "Oil_lag_2", "EUA_lag_2",
                                "WindOn_DA_lag_0", "volatility_pct_24h_lg1", "volatility_24h_lg1",
                                "Temp_lag_1", "Solar_lag_1", "WindS_lag_1", "WindDir_lag_1",
                                "Press_lag_1", "Humid_lag_1")

LS_VAR_APPLY_CYCLECUBIC_SPLINE <- c("sin_hour", "cos_hour", "sin_week", "cos_week", "sin_year", "cos_year")

LS_VAR_APPLY_INTERACTION <- list(c("sin_hour", "cos_hour", "cc"),
                                 c("sin_week", "cos_week", "cc"),
                                 c("sin_year", "cos_year", "cc"))

zones <- c("NO1", "NO2", "NO3", "NO4", "NO5")
MODEL_USING_FIRST_DIFF <- TRUE
S <- 24
Z <- length(zones)

pred_array <- array(NA, dim = c(N_s, S, Z))
true_array <- array(NA, dim = c(N_s, S, Z))
rmse_train_matrix <- matrix(NA, nrow = N_s, ncol = Z)
rmse_test_matrix <- matrix(NA, nrow = N_s, ncol = Z)

############################ PARALLEL SETUP #######################################

num_cores <- detectCores() - 1
cl <- makeCluster(num_cores)

clusterExport(cl, c("build_gam_formula",
                    "LS_VAR_APPLY_SMOOTH_SPLINE",
                    "LS_VAR_APPLY_CYCLECUBIC_SPLINE",
                    "LS_VAR_APPLY_INTERACTION",
                    "train_start_idx", "id_init_eval", "N_s",
                    "S", "MODEL_USING_FIRST_DIFF", "zones"))

clusterEvalQ(cl, {
  library(mgcv)
  library(arrow)
})

############################ MAIN PARALLEL LOOP #######################################

start_time <- Sys.time()

zone_results <- parLapply(cl, seq_along(zones), function(z_idx) {
  
  z <- zones[z_idx]
  cat("\n--- Processing Zone:", z, "---\n")
  
  Xy_flat <- as.matrix(read_parquet(paste0("data_for_R/Xy_t_", z, ".parquet")))
  full_price_original <- Xy_flat[, ncol(Xy_flat)]
  Xy_flat <- Xy_flat[, 1:(ncol(Xy_flat) - 1)]
  colnames_mat <- colnames(Xy_flat)
  
  F <- ncol(Xy_flat)
  T <- nrow(Xy_flat) / S
  Xy_t <- array(Xy_flat, dim = c(S, T, F))
  Xy_t <- aperm(Xy_t, c(2, 1, 3))
  
  full_price_original <- matrix(full_price_original, nrow = T, ncol = S, byrow = TRUE)
  flat_price <- as.vector(t(full_price_original))
  first_lag_flat <- c(rep(NA, S), flat_price[1:(length(flat_price) - S)])
  full_price_first_lag <- matrix(first_lag_flat, nrow = T, ncol = S, byrow = TRUE)
  
  pred_z <- matrix(NA, nrow = N_s, ncol = S)
  true_z <- matrix(NA, nrow = N_s, ncol = S)
  rmse_train_z <- rep(NA, N_s)
  rmse_test_z <- rep(NA, N_s)
  
  for (n in seq_len(N_s)) {
    
    id_start <- train_start_idx + n - 1
    id_end <- id_init_eval + n - 2
    
    train <- Xy_t[id_start:id_end, , ]
    test_x <- Xy_t[id_end + 1, , ]
    
    train_flat <- matrix(aperm(train, c(2, 1, 3)), nrow = dim(train)[1] * dim(train)[2], ncol = dim(train)[3])
    train_df <- as.data.frame(train_flat)
    colnames(train_df) <- colnames_mat
    
    valid_rows <- complete.cases(train_df)
    train_df <- train_df[valid_rows, ]
    
    flat_first_lag_tr <- as.vector(t(full_price_first_lag[id_start:id_end, ]))[valid_rows]
    flat_original_tr <- as.vector(t(full_price_original[id_start:id_end, ]))[valid_rows]
    
    mean_vec <- apply(train_df, 2, mean)
    sd_vec <- apply(train_df, 2, sd)
    sd_vec[sd_vec == 0] <- 1
    
    train_df_scaled <- sweep(train_df, 2, mean_vec, "-")
    train_df_scaled <- sweep(train_df_scaled, 2, sd_vec, "/")
    
    gam_formula <- build_gam_formula(
      feature_names = colnames_mat,
      apply_smooth_list = LS_VAR_APPLY_SMOOTH_SPLINE,
      apply_cyclic_list = LS_VAR_APPLY_CYCLECUBIC_SPLINE,
      apply_interactions = LS_VAR_APPLY_INTERACTION,
      train_df = train_df
    )
    
    bam_model <- bam(gam_formula, data = train_df_scaled, discrete = TRUE, select = TRUE)
    
    y_train_pred <- predict(bam_model, newdata = train_df_scaled)
    y_train_pred <- (y_train_pred * sd_vec[1]) + mean_vec[1]
    if (MODEL_USING_FIRST_DIFF) y_train_pred <- y_train_pred + flat_first_lag_tr
    rmse_train <- sqrt(mean((y_train_pred - flat_original_tr)^2, na.rm = TRUE))
    
    test_df <- as.data.frame(test_x)
    colnames(test_df) <- colnames_mat
    test_df_scaled <- sweep(test_df, 2, mean_vec, "-")
    test_df_scaled <- sweep(test_df_scaled, 2, sd_vec, "/")
    
    y_test_pred <- predict(bam_model, newdata = test_df_scaled[, 2:ncol(test_df_scaled)])
    y_test_pred <- (y_test_pred * sd_vec[1]) + mean_vec[1]
    if (MODEL_USING_FIRST_DIFF) y_test_pred <- y_test_pred + full_price_first_lag[id_end + 1, ]
    
    pred_z[n, ] <- y_test_pred
    true_z[n, ] <- full_price_original[id_end + 1, ]
    rmse_train_z[n] <- rmse_train
    rmse_test_z[n] <- sqrt(mean((y_test_pred - full_price_original[id_end + 1, ])^2, na.rm = TRUE))
    
    cat(sprintf("Zone: %s | Day %d/%d | RMSE train = %.4f | RMSE test = %.4f\n",
                z, n, N_s, rmse_train_z[n], rmse_test_z[n]))
  }
  
  return(list(pred = pred_z, true = true_z, rmse_train = rmse_train_z, rmse_test = rmse_test_z))
  
})

stopCluster(cl)

############################ ASSEMBLE RESULTS #######################################

for (z_idx in seq_along(zones)) {
  pred_array[, , z_idx] <- zone_results[[z_idx]]$pred
  true_array[, , z_idx] <- zone_results[[z_idx]]$true
  rmse_train_matrix[, z_idx] <- zone_results[[z_idx]]$rmse_train
  rmse_test_matrix[, z_idx] <- zone_results[[z_idx]]$rmse_test
}

############################ FINAL TIMING #######################################

end_time <- Sys.time()
diff_time <- difftime(end_time, start_time, units = "mins")
cat(paste("\nExecution took", round(as.numeric(diff_time), 2), "minutes\n"))

############################ FINAL RESULTS #######################################
for (z_idx in seq_along(zones)) {
  rmse_all_test_days <- sqrt(mean((pred_array[, , z_idx] - true_array[, , z_idx])^2))
  avg_rmse_train <- mean(rmse_train_matrix[, z_idx] )
  
  cat(sprintf("Zone: %s | mean for RMSE of all training days = %.4f | RMSE test of all test days= %.4f\n",
              zones[z_idx], avg_rmse_train, rmse_all_test_days))
}




