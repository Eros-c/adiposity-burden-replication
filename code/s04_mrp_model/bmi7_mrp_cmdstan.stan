// Translated to new Stan array syntax for cmdstan 2.36 (stanc3 >= 2.33).
// Semantically identical to bmi7_logistic_normal_mrp_auxiliary.stan used by the
// production rstan fit; only the data-block declarations changed.
data {
  int<lower=1> N;
  int<lower=2> K;
  int<lower=1> P;
  int<lower=1> C;
  int<lower=1> D;
  int<lower=1> F;
  int<lower=1> S;
  array[N] int<lower=1, upper=C> country_id;
  array[N] int<lower=1, upper=D> stage_id;
  array[N] int<lower=1, upper=F> family_id;
  array[N] int<lower=1, upper=S> survey_id;
  array[N] int<lower=0, upper=1> is_train;
  array[N, K] int<lower=0> y;
  matrix[N, K - 1] alr_offset;
  matrix[N, P] X;
  vector[N] ntl_z;
  vector[N] pop_z;
}

parameters {
  matrix[P, K - 1] beta;

  matrix[K - 1, C] z_country_intercept;
  matrix[K - 1, C] z_country_ntl;
  matrix[K - 1, C] z_country_pop;
  vector<lower=0>[K - 1] sigma_country_intercept;
  vector<lower=0>[K - 1] sigma_country_ntl;
  vector<lower=0>[K - 1] sigma_country_pop;

  matrix[K - 1, D] z_stage_ntl;
  matrix[K - 1, D] z_stage_pop;
  vector<lower=0>[K - 1] sigma_stage_ntl;
  vector<lower=0>[K - 1] sigma_stage_pop;

  matrix[K - 1, F] z_family_intercept;
  matrix[K - 1, F] z_family_ntl;
  matrix[K - 1, F] z_family_pop;
  matrix[K - 1, S] z_survey_intercept;
  vector<lower=0>[K - 1] sigma_family_intercept;
  vector<lower=0>[K - 1] sigma_family_ntl;
  vector<lower=0>[K - 1] sigma_family_pop;
  vector<lower=0>[K - 1] sigma_survey_intercept;
}

model {
  to_vector(beta) ~ normal(0, 0.35);

  to_vector(z_country_intercept) ~ std_normal();
  to_vector(z_country_ntl) ~ std_normal();
  to_vector(z_country_pop) ~ std_normal();
  sigma_country_intercept ~ normal(0, 0.30);
  sigma_country_ntl ~ normal(0, 0.20);
  sigma_country_pop ~ normal(0, 0.20);

  to_vector(z_stage_ntl) ~ std_normal();
  to_vector(z_stage_pop) ~ std_normal();
  sigma_stage_ntl ~ normal(0, 0.20);
  sigma_stage_pop ~ normal(0, 0.20);

  to_vector(z_family_intercept) ~ std_normal();
  to_vector(z_family_ntl) ~ std_normal();
  to_vector(z_family_pop) ~ std_normal();
  to_vector(z_survey_intercept) ~ std_normal();
  sigma_family_intercept ~ normal(0, 0.25);
  sigma_family_ntl ~ normal(0, 0.15);
  sigma_family_pop ~ normal(0, 0.15);
  sigma_survey_intercept ~ normal(0, 0.20);

  for (n in 1:N) {
    if (is_train[n] == 1) {
      vector[K] eta;
      int q;
      eta[3] = 0;
      q = 1;
      for (k in 1:K) {
        if (k != 3) {
          eta[k] =
            alr_offset[n, q] +
            X[n] * beta[, q] +
            sigma_country_intercept[q] *
              z_country_intercept[q, country_id[n]] +
            ntl_z[n] * (
              sigma_country_ntl[q] *
                z_country_ntl[q, country_id[n]] +
              sigma_stage_ntl[q] *
                z_stage_ntl[q, stage_id[n]] +
              sigma_family_ntl[q] *
                z_family_ntl[q, family_id[n]]
            ) +
            pop_z[n] * (
              sigma_country_pop[q] *
                z_country_pop[q, country_id[n]] +
              sigma_stage_pop[q] *
                z_stage_pop[q, stage_id[n]] +
              sigma_family_pop[q] *
                z_family_pop[q, family_id[n]]
            ) +
            sigma_family_intercept[q] *
              z_family_intercept[q, family_id[n]] +
            sigma_survey_intercept[q] *
              z_survey_intercept[q, survey_id[n]];
          q += 1;
        }
      }
      y[n] ~ multinomial_logit(eta);
    }
  }
}
