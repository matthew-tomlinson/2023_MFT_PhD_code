# 2023_MFT_PhD_code
Python libraries and notebooks. Tomlinson, Matthew F., PhD thesis: 'Asymmetries and interactions between two tails of extreme returns in financial price time series', University of Bath (2023).


=====================================================================================================
SUBDIRECTORIES

Core

    notebooks
    --------
    # .ipynb Jupiter notebooks used to perform experiments and generate results, tables, and figures.

        B20_results
        ----
        # Generates results of main experiments.

        T23_Thesis
        ----
        # Generates tables and figures for thesis chapters.

    py
    --------
    # .py Python scripts of core library.

    settings
    --------
    # Master settings for programs run in 'notebooks'.


Input

    data
    --------
    # .csv files of historic daily log returns data sourced from 'stooq.com'.


Output

    exports
    --------
        # Figures and tables generated from 'T23_thesis' notebooks for inclusion in LaTeX documents.

        figures
        --------
        # .pdf and .png figures generated from 'T23_thesis' notebooks.

        tables
        --------
        # .sty files for LaTeX-formatted tables generated from 'T23_thesis' notebooks.

    results
    --------
    # .csv files for results of main experiments generated from 'B20_results' notebooks.

        fit
        ----
        # Results of parameter estimation and associated goodness of fit measurement.

        vares
        ----
        # Results of conditional quantile (VaR) and conditional violation expectation (ES) analysis.


=====================================================================================================