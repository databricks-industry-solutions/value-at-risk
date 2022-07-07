<img src=https://d1r5llqwmkrl74.cloudfront.net/notebooks/fsi/fs-lakehouse-logo-transparent.png width="600px">

[![DBR](https://img.shields.io/badge/DBR-10.4ML-red?logo=databricks&style=for-the-badge)](https://docs.databricks.com/release-notes/runtime/10.4ml.html)
[![CLOUD](https://img.shields.io/badge/CLOUD-ALL-blue?logo=googlecloud&style=for-the-badge)](https://cloud.google.com/databricks)
[![POC](https://img.shields.io/badge/POC-10_days-green?style=for-the-badge)](https://databricks.com/try-databricks)

*This solution has two parts. First, it shows how Delta Lake and MLflow can be used for value-at-risk calculations – showing how banks can modernize their risk management practices by back-testing, aggregating and scaling simulations by using a unified approach to data analytics with the Lakehouse. Secondly. the solution uses alternative data to move towards a more holistic, agile and forward looking approach to risk management and investments.*

___
<antoine.amend@databricks.com>

___

<img src='https://raw.githubusercontent.com/databricks-industry-solutions/value-at-risk/master/images/reference_architecture.png' width=800>

___

&copy; 2022 Databricks, Inc. All rights reserved. The source in this notebook is provided subject to the Databricks License [https://databricks.com/db-license-source].  All included or referenced third party libraries are subject to the licenses set forth below.

| library                                | description             | license    | source                                              |
|----------------------------------------|-------------------------|------------|-----------------------------------------------------|
| Yfinance                               | Yahoo finance           | Apache2    | https://github.com/ranaroussi/yfinance              |
| tempo                                  | Timeseries library      | Databricks | https://github.com/databrickslabs/tempo             |
| PyYAML                                 | Reading Yaml files      | MIT        | https://github.com/yaml/pyyaml                      |
