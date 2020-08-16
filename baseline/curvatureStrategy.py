#!/usr/local/bin/python3.7
from utils import *
import networkx as nx
import numpy as np
import pandas as pd
import cvxpy as cp
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as datetime
import rcparams
import warnings
warnings.filterwarnings('ignore')

rcparams.set_rcparams()

class CurvatureStrat:
    """Creates an instance of CurvatureStrat.

    The strategy is designed by considering the swap curve, a line that plots swap rates
    against its respective maturity, to have anunderlying graph node structure, which allows 
    us to associate it with a Laplacian matrix. Combined with heat diffusion theory, which 
    states that the Laplacian operator is equal to the negative of the Laplacian matrix, 
    provides a pathway to extract information regarding the swap curve’s curvature. 
    This is then pieced together to formulate an optimization problem in order to estimate 
    the Laplacian matrix for each day. Variants of co-ordinate descent are used to tackle this 
    optimization problem. To backtest the trading strategy, the Laplacian matrix is viewed as
    different sets of portfolio weights that sum to zero, creating ‘zero net investment portfolios’
    that are traded each day based on various curvature conditions and thresholds. The idea is that
    the swap curve has a diffusive component to correct for short term blips and volatility that 
    contribute to the overall curve, which combined withthe Laplacian dictate the curvature conditions. 

    Args:
        df (pandas.DataFrame): A pandas dataframe of the swaps data created with getData() 
            in utils.py. 
        cd_iter (int): The number of coordinate descent iterations to be used.
        cd_weight (str): The coordinate descent scheme to be used.
    """    

    def __init__(self, df, cd_iter, cd_weight):
        self.df = df
        self.currency = "".join(df.columns.get_level_values(0).unique())
        self.cd_iter = cd_iter
        self.cd_weight = cd_weight

    def coordinate_descent(self, X, Y):
        """Performs coordinate descent using cvxpy.

        Args:
            X (array-like): Swap curve at time t.
            Y (array-like): Swap curve at time t + 1.

        Returns:
            m_each_iter: Returns m for each iteration of coordinate descent.
            L_each_iter: Returns L for each iteration of coordinate descent.
            m_latest: Returns the m from the last iteration of coordinate descent.
            L_latest: Returns the L from the last iteration of coordinate descent.
        """        
        # generalize input
        if X.shape[1] != 15:
            X = X.T
            Y = Y.T
        else:
            pass
        
        # optimization init
        N = X.shape[1]
        T = X.shape[0]
        I = np.eye(N)
        e = np.ones((N,1))
        z = np.zeros((N,1))

        m_each_iter = np.zeros((N,self.cd_iter))
        L_each_iter = np.zeros((N,N,self.cd_iter))

        sample_mean = np.mean(X,0).reshape(-1,1)
        mu = np.mean(sample_mean)
        m_tilde = sample_mean - mu # m_tilde is zero mean vector
        
        # Normal least squares
        if self.cd_weight == None:
            for i in range(self.cd_iter):
                print('CD iterations: ', i+1)

                # solve for L
                L = cp.Variable((N,N),symmetric=True)
                cost = 0
                for t in range(T):
                    cost += cp.sum_squares(Y[t,:].reshape(-1,1) - (I-L)@X[t,:].reshape(-1,1) - L@m_tilde)
                prob = cp.Problem(cp.Minimize(cost),[L@e == z])
                prob.solve()
                L = L.value
                L_each_iter[:,:,i] = L

                # solve for m
                m_tilde = cp.Variable((N,1))
                cost = 0
                for t in range(T):
                    cost += cp.sum_squares(Y[t,:].reshape(-1,1) - (I-L).dot(X[t,:].reshape(-1,1)) - L@m_tilde)
                prob = cp.Problem(cp.Minimize(cost),[cp.sum(m_tilde) == 0])
                prob.solve()
                m_tilde = m_tilde.value

                m_each_iter[:,[i]] = m_tilde + mu
        
        # Exponential-weighted least squares
        elif self.cd_weight == 'exp':
            gamma = 0.9
            for i in range(self.cd_iter):
                print('CD iterations: ', i+1)

                # solve for L
                L = cp.Variable((N,N),symmetric=True)
                cost = 0
                for t in range(T):
                    cost += gamma**t*cp.sum_squares(Y[t,:].reshape(-1,1) - (I-L)@X[t,:].reshape(-1,1) - L@m_tilde)
                prob = cp.Problem(cp.Minimize(cost),[L*e == z])
                prob.solve()
                L = L.value
                L_each_iter[:,:,i] = L

                # solve for m
                m_tilde = cp.Variable((N,1))
                cost = 0
                for t in range(T):
                    cost += gamma**t*cp.sum_squares(Y[t,:].reshape(-1,1) - (I-L).dot(X[t,:].reshape(-1,1)) - L@m_tilde)
                prob = cp.Problem(cp.Minimize(cost),[cp.sum(m_tilde) == 0])
                prob.solve()
                m_tilde = m_tilde.value

                m_each_iter[:,[i]] = m_tilde + mu

        else:
            raise Exception('weight must be specified as either None (for normal coordinate descent) or exp (for exponential coordinate descent)')
                
        m_latest=m_each_iter[:,[self.cd_iter-1]]
        L_latest=L_each_iter[:,:,self.cd_iter-1]
        
        return m_each_iter, L_each_iter, m_latest, L_latest

    def getLaplacian(self, w, rollingMSE=False):
        """Uses coordinate descent to solve for the Laplacian matrix L.
        
        Given a rolling backtest window size w, it uses coordinate descent to solve 
        for the Laplacian matrix L, rolls the window forward by 1 stride and repeat.

        Args:
            w (int): The rolling backtest window size to be used.

        Returns:
            daily_L: Returns the evolution of L for each stride looped through.
            daily_m: Returns the evolution of m for each stride looped through.
        """        
        strides = len(self.df) - w + 1
        L_evol = np.zeros((len(self.df.columns.get_level_values(0)), len(self.df.columns.get_level_values(0)), strides))
        m_evol = np.zeros((len(self.df.columns.get_level_values(0)), strides))
        
        print('window size: ', w)
        
        for i in range(strides-1):

            print('periods rolled: {}/{}\r'.format(i+1, strides))

            X = self.df.T.values[:,0+i:w-1+i]
            Y = self.df.T.values[:,1+i:w+i]

            _, _, m_latest, L_latest = self.coordinate_descent(X, Y)

            L_evol[:,:,i] = L_latest
            m_evol[:,[i]] = m_latest

        return L_evol, m_evol
    
    def swapCurvePrediction(self, L, m, w, rollingMSE=False):
        """Calculates the swap curve predictions based on the proposed optimization problem.

        Args:
            L (array-like): L_evol from the getLaplacian method.
            m (array-like): m_evol from the getLaplacian method.
            w (int): The rolling backtest window size to be used.
            rollingMSE (bool, optional): If True, calculate the MSE for each daily 
                swap curve prediction. Defaults to False.

        Returns:
            yield_pred_evol: Returns a pandas.DataFrame object of the swap curve predictions, 
                calculated based on the proposed optimization equation. (See readme.)
            mse: Returns a pandas.DataFrame object of the MSE calculation if rollingMSE is asserted.
        """        
        strides = len(self.df) - w + 1
        last_date = datetime.datetime.strptime('2019-07-23','%Y-%m-%d').date()
        #yield_pred_evol = np.zeros((len(df.columns.get_level_values(1)), strides-1))
        yield_pred_evol = pd.DataFrame(columns=self.df.columns.get_level_values(1),index=self.df.index[w:])
        yield_pred_evol = yield_pred_evol.append(pd.DataFrame(index=[last_date]))
        mse = pd.DataFrame(index=self.df.index[w:],columns=['MSE'])

        for i in range(strides):
            yield_pred = np.matmul( np.eye(len(m[:,i])) - L[:,:,i], self.df.T.values[:,[w-1+i]] ) + np.matmul(L[:,:,i], m[:,i]).reshape(len(m[:,i]),1)
            yield_pred_evol.iloc[[i],:] = yield_pred.T
        
            if rollingMSE:
                    mse.iloc[i,:] = mean_squared_error(yield_pred_evol.iloc[[i],:],self.df.iloc[[w+i],:])
        #         print(yield_pred_evol)
        #         print('')
        #         print(df.T.values[:,windowsize:])
            
        return yield_pred_evol, mse

    def plotSwapCurvePredictionsByMaturity(self, ype, w, maturity_index=0, view_all=False):
        """Plots the swap rate predictions for a specified maturity, or for all maturities.

        Args:
            ype (pandas.DataFrame): The yield predictions from the swapCurvePredictions method.
            w (int): The rolling backtest window size to be used.
            maturity_index (int, optional): The maturity index to plot. Valid range is from 
                0 to 14. Defaults to 0.
            view_all (bool, optional): If True, generate a 5x3 subplot to visualize
                for all maturities. Defaults to False.
        """        
        maturity_index_mapping = {
                                0 : 1, 
                                1 : 2, 
                                2 : 3, 
                                3 : 4,
                                4 : 5,
                                5 : 6, 
                                6 : 7, 
                                7 : 8, 
                                8 : 9,
                                9 : 10, 
                                10 : 12,
                                11 : 15,
                                12 : 20,
                                13 : 25,
                                14 : 30
        }

        if not view_all:
            plt.plot(self.df[self.currency].iloc[w:,maturity_index], label='Actual')
            plt.plot(ype.iloc[:,maturity_index], label='Prediction')
            plt.xticks([ype.index[i] for i in np.linspace(0,len(ype.index)-1,10).astype(int)]);
            plt.title(f'Swap Rate Prediction vs Actual, {self.currency}-{maturity_index_mapping[maturity_index]}')
            plt.xlabel('Date')
            plt.ylabel('Swap rate (%)')
            plt.legend()
            plt.grid()
            plt.show()

        else:
            plt.figure(figsize=(15,15))

            for i in range(15):
                plt.subplot(5,3,i+1)
                plt.plot(self.df[self.currency].iloc[w:,i], label='Actual')
                plt.plot(ype.iloc[:,i], label='Prediction')
                plt.xticks([ype.index[i] for i in np.linspace(0,len(ype.index)-1,5).astype(int)]);
                plt.xlabel('Date', fontsize=10)
                plt.ylabel('Swap Rate (%)', fontsize=10)
                plt.title(f'Swap Rate Prediction vs Actual, {self.currency}-{maturity_index_mapping[i]}', fontsize=10)
                plt.legend()
                plt.grid()
                plt.tight_layout()
            
            plt.show()

    def plotSwapCurvePredictionsByDate(self, ype, w, i=0):
        """Plots the swap curve predictions for a particular date.

        Args:
            ype (pandas.DataFrame): The yield predictions from the swapCurvePredictions method.
            w (int): The rolling backtest window size to be used.
            i (int, optional): The index corresponds to a date. Defaults to 0.

        Note:
            Run the method to see the valid index range given the particular swaps dataframe used.
        """        
        stride = len(self.df) - w + 1
        print('valid date range -', self.df[self.currency].index.strftime('%Y-%m-%d')[w], '(i=0)','to', self.df[self.currency].index.strftime('%Y-%m-%d')[-1], f'(i={stride-2})')
        print('')
        
        plt.plot(self.df[self.currency].iloc[w+i,:],label='Actual')
        plt.plot(ype.iloc[0+i,:],label='Prediction')
        plt.xlabel('Maturity')
        plt.ylabel('Swap rate (%)')
        plt.title('Swap Curve Prediction vs Actual - ' + self.df[self.currency].iloc[w+i,:].name.date().strftime('%Y-%m-%d'))
        plt.grid()
        plt.legend()
        plt.show()
    
    def plotLaplacian(self, L, whichStride):
        """Plots each row of the Laplacian matrix in subplots.

        Args:
            L (array-like): L_evol from the getLaplacian method.
            whichStride (int): The particular stride of the Laplacian matrix to plot.

        Note:
            The stride calculation can be referred to in the getLaplacian method.
        """        
        last_L = pd.DataFrame(L[:,:,whichStride])
        plt.figure(figsize=(15,20))
        for row in range(len(L)):
            plt.subplot(5,3,row+1)
            last_L.iloc[row,:].plot()
            plt.axhline(0,alpha=0.3)
            plt.title('Laplacian row {}'.format(row))
            plt.tight_layout()
        
        plt.show()

    def plot_m(self, m, whichStride, title):
        """[summary]

        Args:
            m (array-like): The entire m matrix.
            whichStride (int): The particular stride of the m matrix to plot
            title (str): The title of the generated plot.
        """        
        plt.figure(figsize=(7,5))
        plt.plot(self.df.columns.get_level_values(1), m[:,whichStride])
        plt.title(title)
        plt.xlabel('Maturity')
        plt.ylabel('$\mathbf{m}$')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def get_learnedWeightMatrix(self, L, whichStride, wthresh=0.1, showGraph=False):
        """Generates a graph network of the underlying weight matrix recovered from the Laplacian.

        The underlying weight matrix is recovered as W = diag(L) - L.

        Args:
            L (array-like): The entire Laplacian Matrix.
            whichStride (int): The particular stride of the Laplacian Matrix to use
            wthresh (float, optional): Weight matrix threshold to use, below which the 
                weights are considered to be insignificant. Defaults to 0.1.
            showGraph (bool, optional): If True, show the generated graph. Defaults to False.

        Returns:
            G: Returns a Graph object if showGraph=False.
            weights: Returns the weights associated to the Graph object if showGraph=False.
        """        
        L = L[:,:,whichStride]
        W = pd.DataFrame(np.diag(np.diag(L)) - L)
        W = W[W > wthresh].fillna(0).round(2)
        W.index = list(self.df.columns.get_level_values(1))
        W.columns= list(self.df.columns.get_level_values(1))
        
        G = nx.from_pandas_adjacency(W)
        
        weights = []
        for _, _, d in G.edges(data=True):
            weights.append(d['weight'])
        
        if showGraph:
            fig = plt.figure(figsize=(25,8))
            plt.title('Recovered Weight Matrix via Learned Laplacian',fontsize=35, color='white')
            nx.draw(G, with_labels=True, node_color='skyblue', node_size=2000, edge_color=weights, width=8, edge_cmap=plt.cm.Blues, font_size=15)
            fig.set_facecolor("#00000F")
            plt.show()

        else:
            return G, weights

    def backtest(self, w, m, L, cash=1, normalize=True, curvatureThreshold=0.002):
        """Backtests the strategy with a long/short trading signal given by a curvature threshold.

        The Laplacian matrix L is a 15x15 matrix and acts as a set of portfolio weights to trade on.
        This means that each day 15 portfolios are traded on, with each portfolio consisting of 15 
        different maturities. As a result, in this context 'trading the ith maturity' refers 
        to trading the particular portfolio (dictated by the ith row of L) associated
        with that maturity.

        Args:
            w (int): Backtest window size to be used.
            m (array-like): The entire m matrix to be used for backtesting.
            L (array-like): The entire Laplacian matrix to be used for backtesting.
            cash (int, optional): Daily investment amount for each zero-investment 
                portfolio each day. Defaults to 1.
            normalize (bool, optional): If True, normalize the Laplacian. Defaults to True.
            curvatureThreshold (float, optional): The curvature condition used to generate 
                trading signals. Defaults to 0.002.

        Returns:
            profits: Returns a pandas.DataFrame object of the daily profits for each portfolio.
            daily_agg: Returns a pandas.DataFrame object of the cumulative profits for each portfolio.
            curvature: Returns a pandas.DataFrame object of the daily curvature of each portfolio.
            short_long_stats: Returns a pandas.DataFrame object of the number of short and long trades taken, 
                and the respective profits.

        """        
        L = L[:,:,:-1]
        m = m[:,:-1]
        df = self.df.iloc[w:,:]

        if normalize:
            L = normalize_L(L)

        today = np.zeros((L.shape[0],L.shape[1],L.shape[2]))
        evolve = np.zeros((L.shape[0],L.shape[1],L.shape[2]))
        profits = pd.DataFrame(index=df.index,columns=df.columns).fillna(0)
        curvature = pd.DataFrame(index=df.index,columns=df.columns)
        short_long_stats = pd.DataFrame(index=['long trades','short trades','long profits','short profits'],columns=df.columns).fillna(0)
        
        for i in range(L.shape[2]-1):
            today[:,:,i] = np.multiply(L[:,:,i],df.iloc[i,:].values) # portfolio value today (rows)
            evolve[:,:,i] = np.multiply(L[:,:,i],df.iloc[i+1,:].values) # portfolio value tomorrow (rows)
            curvature.iloc[i,:] = L[:,:,i].dot(df.iloc[i,:] - m[:,i])

            for j in range(L.shape[0]):
                if curvature.iloc[i,j] > curvatureThreshold: 
                    #short
                    profits.iloc[i,j] = np.sum(cash*today[j,:,i] - cash*evolve[j,:,i])
                    short_long_stats.loc['short trades'][j] += 1
                    short_long_stats.loc['short profits'][j] += np.sum(cash*today[j,:,i] - cash*evolve[j,:,i])
                if curvature.iloc[i,j] < -curvatureThreshold:
                    #long
                    profits.iloc[i,j] = np.sum(cash*evolve[j,:,i] - cash*today[j,:,i])
                    short_long_stats.loc['long trades'][j] += 1
                    short_long_stats.loc['long profits'][j] += np.sum(cash*evolve[j,:,i] - cash*today[j,:,i])
                    
            # original condition
    #         for j in range(L.shape[0]):
    #             if today[j,j,i] > curvatureThreshold: 
    #                 # short
    #                 profits.iloc[i,j] = np.sum(cash*today[j,:,i] - cash*evolve[j,:,i])
    #             if today[m,m,i] < -curvatureThreshold:
    #                 # long
    #                 profits.iloc[i,j] = np.sum(cash*evolve[j,:,i] - cash*today[j,:,i])
            
        daily_agg = profits.cumsum().sum(axis=1)

        return profits, daily_agg, curvature, short_long_stats

    def cumulativeProfitsByMaturity(self, profits):
        """Plots the cumulative profits for each portfolio.

        Args:
            profits (pd.DataFrame): Takes in a profits dataframe from the backtest method.
        """        
        if self.cd_weight == None:
            title = f'{self.currency} Swaps - Cumulative profits for each maturity: normal CD scheme'
        if self.cd_weight == 'exp':
            title = f'{self.currency} Swaps - Cumulative profits for each maturity: exponential CD scheme'

        plt.plot(profits.cumsum())
        plt.legend(self.df.columns.get_level_values(1),ncol=15)
        plt.xlabel('Date')
        plt.ylabel('Absolute profit')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def endingWealth(self, profits):
        """Plots the final profits for each portfolio in a bar chart format.

        Args:
            profits (pd.DataFrame): Takes in a profits dataframe from the backtest method.
        """       
        if self.cd_weight == None:
            title = f'{self.currency} Swaps - Ending wealth by maturity: normal CD scheme'
        if self.cd_weight == 'exp':
            title = f'{self.currency} Swaps - Ending wealth by maturity: exponential CD scheme'

        profits.sum().plot(kind='bar',legend=False,title=title,rot=0,color='b')
        plt.xlabel('Maturity')
        plt.ylabel('Absolute profit')

    def cumulativeProfitsAggregated(self, profits):
        """Plots the cumulative profits aggregated across all portfolios.

        Args:
            profits (pd.DataFrame): Takes in a profits dataframe from the backtest method.
        """       
        if self.cd_weight == None:
            title = f'{self.currency} Swaps - Daily cumulative profits aggregated across all portfolios: normal CD scheme'
        if self.cd_weight == 'exp':
            title = f'{self.currency} Swaps - Daily cumulative profits aggregated across all portfolios: exponential CD scheme'

        plt.plot(profits.cumsum().sum(axis=1))
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Absolute profit')

    def tradeAnalytics(self, profits, figsize=(5,5)):
        """Bar chart breakdowns of the trades taken for each portfolio.

        Args:
            profits (pd.DataFrame): Takes in a profits dataframe from the backtest method.
            figsize (tuple, optional): Matplotlib figure size. Defaults to (5,5).
        """        
        if self.cd_weight == None:
            title = f'{self.currency} Swaps - Trades Breakdown: normal CD scheme'
        if self.cd_weight == 'exp':
            title = f'{self.currency} Swaps - Trades Breakdown: exponential CD scheme'

        profits_sum = pd.DataFrame(index=['profits sum'],columns=profits.columns)
        loss_sum = pd.DataFrame(index=['losses sum'],columns=profits.columns)
        successful_trades = pd.DataFrame(index=['successful trades'],columns=profits.columns)
        number_of_trades_made = pd.DataFrame(data=(profits != 0).sum(axis=0))
        number_of_trades_made.columns = ['trades made']
        total_days = profits.shape[0] - 1
        count = 0
        
        for j in range(profits.shape[1]):
            profits_sum.iloc[:,j] = profits.iloc[:,j].where(profits.iloc[:,j] > 0).sum()
            loss_sum.iloc[:,j] = profits.iloc[:,j].where(profits.iloc[:,j] < 0).sum()
            
        pl_sum = pd.concat([profits_sum,loss_sum],axis=0)
            
        for j in range(profits.shape[1]):
            for i in range(profits.shape[0]):
                if profits.iloc[i,j] > 0:
                    count += 1
            successful_trades.iloc[:,j] = count
            count = 0
            trades_made = pd.concat([number_of_trades_made.T,successful_trades],axis=0)
            
        fig = plt.figure(figsize=figsize)
        ax1 = fig.add_axes([0.1, 0.5, 1.1, 0.5])
        pl_sum.iloc[0,:].plot(kind='bar',stacked=True,rot=0,ax=ax1,color='g')
        pl_sum.iloc[1,:].plot(kind='bar',stacked=True,rot=0,ax=ax1,color='r')
        plt.legend(['Profit','Loss'])
        plt.title(title)
        plt.ylabel('Absolute profit')
        ax2 = fig.add_axes([0.1, 0.0, 1.1, 0.5])
        trades_made.iloc[0,:].plot(kind='bar',stacked=True,rot=0,ax=ax2,color='b')
        trades_made.iloc[1,:].plot(kind='bar',stacked=True,rot=0,ax=ax2,color='y',width=0.4)
        plt.hlines(total_days, xmin=-1, xmax=8,linestyle='dashed',alpha=0.5,color='r')
        plt.legend(['Total days','Total trades','Successful trades'],ncol=3,loc='best')
        plt.xlabel('Maturity')
        plt.ylabel('Number of trades')
        plt.show()

    def successRatio(self, profits):
        """Calculates the success ratios of each portfolio.

        The success ratio is defined as the number of profitable trades taken as a fraction of 
        the total number of trades taken over the time period. 

        Args:
            profits (pd.DataFrame): Takes in a profits dataframe from the backtest method.

        Returns:
            successratio: Returns a pandas.DataFrame object with the success ratios of each portfolio.
        """        
        count = 0
        count_0 = 0
        successratio = pd.DataFrame(index=['success ratio'],columns=profits.columns)
        
        for j in range(profits.shape[1]):
            for i in range(profits.shape[0]):
                if profits.iloc[i,j] > 0:
                    count += 1
                if profits.iloc[i,j] == 0:
                    count_0 += 1
            successratio.iloc[:,j] = count/(profits.shape[0] - 1 - count_0)
            count = 0
            count_0 = 0

        return successratio
    
    def stressTest_w_c(self):
        """Performs a stress test of the strategy under various configurations of w and c.

        w is the backtest window size and c is the curvature threshold.

        Returns:
            agg_table: Returns a pandas.DataFrame object with the final cumulative profits for each stress test
                configuration.
            largest_pl: Returns a pandas.DataFrame object with the portfolio that generated the largest final 
                cumulative profit for each stress test configuration.
            mse_table: Returns a pandas.DataFrame object with the mean squared error of the swap curve predictions
                for each configuration.
        """        
        w_range = [20,30,50,70,100]
        c_thresh = [0, 0.0005, 0.001, 0.002, 0.005]
        agg_table = pd.DataFrame(index=[['w','w','w','w','w'],['20','30','50','70','100']],columns=[['$\tau$','$\tau$','$\tau$','$\tau$','$\tau$'],['0','0.0005','0.001','0.002','0.005']])
        largest_pl = pd.DataFrame(index=[['w','w','w','w','w'],['20','30','50','70','100']],columns=[['$\tau$','$\tau$','$\tau$','$\tau$','$\tau$'],['0','0.0005','0.001','0.002','0.005']])
        mse_table = pd.DataFrame(index=[['w','w','w','w','w'],['20','30','50','70','100']],columns=[['$\tau$','$\tau$','$\tau$','$\tau$','$\tau$'],['0','0.0005','0.001','0.002','0.005']])

        for i in range(len(w_range)):
            count = 0
            for j in range(len(c_thresh)):
                if count == 0:
                    L, m = self.getLaplacian(w_range[i])
                    y, _ = self.swapCurvePrediction(L, m, w_range[i])
                    count += 1
                
                mse_table.iloc[i,j] = mean_squared_error(y.iloc[:-1,:], self.df.iloc[w_range[i]:,:])
                p, agg, _, _ = self.backtest(w_range[i], m[:,:-1], L[:,:,:-1], cash=1, normalize=True, curvatureThreshold=c_thresh[j])
                agg_table.iloc[i,j] = agg.iloc[-1]
                largest_pl.iloc[i,j] = p.sum().idxmax()
        
        return agg_table, largest_pl, mse_table

    def getHeatmap(self, title):
        """Plots a heatmap based on the stress test results from the stressTest_w_c method.

        Args:
            title (str): Specifies the title of the heatmap.
        """        
        agg_table, _, _ = self.stressTest_w_c()
        plt.figure(figsize=(15,10))
        sns.heatmap(agg_table.loc['w','$\tau$'].astype(int),cmap = "YlGnBu",annot=True,fmt='d',annot_kws={'size':13})
        plt.xlabel(r'$\tau$')
        plt.ylabel('$w$')
        plt.title(title)
        plt.show()


