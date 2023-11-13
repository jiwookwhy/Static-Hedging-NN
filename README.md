# Static-Hedging-NN
Static hedging of equity index derivatives using nearby contracts using feed-forward neural networks. Two neural networks  (an evaluation network and a policy network) take as input a target option to hedge and jointly return a set of three nearby option contracts on the volatility surface and portfolio weights such that the portfolio formed by these nearby with the returned weights hedges price fluctations in the target option over a user-defined holding period. 

It performs similiarly to other modern static hedging algorithms in the literature such as those devloped by [Wu and Zhu (2019)](https://academic.oup.com/jfec/article/15/1/1/2548347 ) or [Carr and Wu (2016)](https://academic.oup.com/jfec/article/12/1/3/815302) and outperforms standard Black-Scholes delta hedging in the presence of jumps in the underlying.

Two pairs of pre-trained nueral networks on SPX and VIX option prices repsectively are provided as .keras files.

