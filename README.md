# Sports-Analytics
This repository documents my research and experiments in sports analytics, focusing primarily on football. It includes predictive models, match data analysis, tactical insights, and machine learning approaches to understand and forecast game outcomes.


# Premier League Final Position Predictor
A probabilistic framework to predict final Premier League positions using betting odds, Poisson modeling, and Monte Carlo simulation.

#Project Overview
This project combines:

Betting market data from The Odds API

Historical match results from [Football-Data.org]((https://www.football-data.org/))

Poisson distribution modeling for match outcomes

Monte Carlo simulation (20,000 iterations) to generate probability distributions over final league positions

The model produces a probability matrix showing each team's likelihood of finishing in each position (1st through 20th).

#Required APIs
We will need API keys from:

[The Odds API]([url](https://the-odds-api.com/)) - for current betting odds

[Football-Data.org]((https://www.football-data.org/)) - for fixtures and historical results
