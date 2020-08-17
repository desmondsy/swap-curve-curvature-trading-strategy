# swap-curve-curvature-trading-strategy
Final Year Project

## Abstract
This project concerns the research and design of a novel yield curve trading strategy that involves elements within graph theory, signal processing and diffusion processes. In particular, the strategy is designed by considering the swap curve, a line that plots swap rates against its respective maturity, to have an underlying graph node structure, which allows us to associate it with a Laplacian matrix. Combined with heat diffusion theory, which states that the Laplacian operator is equal to the negative of the Laplacian matrix, it provides a pathway to extract information regarding the swap curve’s curvature. This is then pieced together to formulate an optimization problem in order to estimate the Laplacian matrix for each day. Variants of co-ordinate descent are used to tackle this optimization problem. 

To backtest the trading strategy, the Laplacian matrix is viewed as different sets of portfolio weights that sum to zero, creating ‘zero net investment portfolios’ that are traded each day based on various curvature conditions and thresholds. The idea is that the swap curve has a diffusive component to correct for short term blips and volatility that contribute to the overall curve, which combined with the Laplacian dictate the curvature conditions, which are our trading signals. Finally, we show that the strategy is profitable with a range of swap curves in different regions, and that the strategy can also be extended to other asset classes such as derivatives, where instead of the swap curve we can use the VIX term structure to create a profitable derivatives trading strategy.

## Requirements
```
pip3 install -r requirements.txt
```
## Dataset
Contains swap rates for 15 different maturities ranging from 1Y-30Y, for a total of 8 swap denominations (USD, EUR, GBP, CAD, NZD, AUD, CHF, JPY). Set1 ranges from Jan 2015 to Jul 2019, and set2 ranges from Jan 2000 to May 2020. Data is retrieved via the Bloomberg Terminal Excel Add-in tool.

## Theory
- The `baseline/theory.ipynb` notebook contains a brief summary of the theory and rationale behind the creation of the strategy.
