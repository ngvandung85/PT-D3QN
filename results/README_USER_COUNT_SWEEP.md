# User-count scaling outputs

Each `users_NNN` directory is a complete, independent experiment for one user
count. Random PT traits and evaluation references are resolved once per user
and Monte Carlo replicate at the maximum population, then exact prefixes are
frozen for all nine methods and all episodes. The root `user_count_*.csv`
files concatenate replicate-level or plot-ready values across user counts; raw
episode data remain in the corresponding subdirectory to avoid duplication.

For statistical inference, use replicate-level rows, not evaluation episodes or
users as independent samples.
