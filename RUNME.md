# User Guide

This guide walks you through the configuration and execution of the value at risk solution accelerator.
Based on previous implementations, we estimate the time for an overall POC duration to be around 10 days.

## [00_configure_notebook](config/configure_notebook.py)

[![POC](https://img.shields.io/badge/_-ARBITRARY_FILE-lightgray?style=for-the-badge)]()

Although executed through multiple notebooks, this solution is configured using external configuration file. 
See [application.yaml](config/application.yaml) for more information about each configuration item. 
This configuration will be used in [configure_notebook](config/configure_notebook.py) and 'injected' in each notebook 
as follows. Please note that the portfolio used in that POC is provided as an external [file](config/portfolio.json) 
and accessible through the configuration variable `portfolio`. In real-life scenario, this would be read from an 
external table or as job argument. 

```
%run config/configure_notebook
```

# Issues

For any issue, please raise ticket to github project directly.