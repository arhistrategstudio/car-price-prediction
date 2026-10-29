| model                        | target   |   MAE |       MSE |   RMSE |    R2 | MedAPE   |   train_time_s |
|:-----------------------------|:---------|------:|----------:|-------:|------:|:---------|---------------:|
| RandomForest                 | log      | 1,000 | 5.404e+06 |  2,325 | 0.914 | 11.6%    |             27 |
| HistGradientBoosting         | log      | 1,023 | 5.376e+06 |  2,319 | 0.914 | 11.9%    |              7 |
| HistGradientBoosting (raw $) | raw      | 1,051 | 4.948e+06 |  2,225 | 0.921 | 12.2%    |              5 |
| GroupMedian lookup (bez ML)  | raw      | 1,181 | 1.028e+07 |  3,207 | 0.836 | 12.2%    |              0 |
| DecisionTree                 | log      | 1,261 | 8.407e+06 |  2,899 | 0.866 | 13.9%    |              0 |
| Formula: base + 4 člana      | log      | 1,277 | 9.212e+06 |  3,035 | 0.853 | 14.3%    |              0 |
| Symbolic GP (gplearn)        | log      | 1,318 | 1.112e+07 |  3,334 | 0.823 | 14.3%    |            151 |
| Ridge                        | log      | 1,500 | 1.168e+07 |  3,418 | 0.814 | 16.6%    |              0 |
| LinearRegression             | raw      | 2,453 | 2.029e+07 |  4,504 | 0.676 | 29.9%    |              1 |