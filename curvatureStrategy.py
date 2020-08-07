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
    def __init__(self, df, cd_iter, cd_weight):
        self.df = df
        self.currency = "".join(df.columns.get_level_values(0).unique())
        self.cd_iter = cd_iter
        self.cd_weight = cd_weight

    def coordinate_descent(self, X, Y, iterations,weight):
        #generalize input
        if X.shape[1] != 15:
            X = X.T
            Y = Y.T
        else:
            pass
        
        #initialization
        N = X.shape[1]
        T = X.shape[0]
        I = np.eye(N)
        e = np.ones((N,1))
        z = np.zeros((N,1))

        m_evol = np.zeros((N,iterations))
        L_evol = np.zeros((N,N,iterations))

        sample_mean = np.mean(X,0).reshape(-1,1)
        mu = np.mean(sample_mean)
        m_tilde = sample_mean - mu # m_tilde is zero mean vector
        
        #normal least squares
        if weight == None:
            for i in range(iterations):
                print('CD iterations: ', i+1)

                # solve for L
                L = cp.Variable((N,N),symmetric=True)
                cost = 0
                for t in range(T):
                    cost += cp.sum_squares(Y[t,:].reshape(-1,1) - (I-L)@X[t,:].reshape(-1,1) - L@m_tilde)
                prob = cp.Problem(cp.Minimize(cost),[L@e == z])
                prob.solve()
                L = L.value
                L_evol[:,:,i] = L

                # solve for m
                m_tilde = cp.Variable((N,1))
                cost = 0
                for t in range(T):
                    cost += cp.sum_squares(Y[t,:].reshape(-1,1) - (I-L).dot(X[t,:].reshape(-1,1)) - L@m_tilde)
                prob = cp.Problem(cp.Minimize(cost),[cp.sum(m_tilde) == 0])
                prob.solve()
                m_tilde = m_tilde.value

                m_evol[:,[i]] = m_tilde + mu
        
        #exponential-weighted least squares
        if weight == 'exp':
            gamma = 0.9
            for i in range(iterations):
                print('CD iterations: ', i+1)

                # solve for L
                L = cp.Variable((N,N),symmetric=True)
                cost = 0
                for t in range(T):
                    cost += gamma**t*cp.sum_squares(Y[t,:].reshape(-1,1) - (I-L)@X[t,:].reshape(-1,1) - L@m_tilde)
                prob = cp.Problem(cp.Minimize(cost),[L*e == z])
                prob.solve()
                L = L.value
                L_evol[:,:,i] = L

                # solve for m
                m_tilde = cp.Variable((N,1))
                cost = 0
                for t in range(T):
                    cost += gamma**t*cp.sum_squares(Y[t,:].reshape(-1,1) - (I-L).dot(X[t,:].reshape(-1,1)) - L@m_tilde)
                prob = cp.Problem(cp.Minimize(cost),[cp.sum(m_tilde) == 0])
                prob.solve()
                m_tilde = m_tilde.value

                m_evol[:,[i]] = m_tilde + mu
                
        m_latest=m_evol[:,[iterations-1]]
        L_latest=L_evol[:,:,iterations-1]
        
        return m_evol, L_evol, m_latest, L_latest

    def getLaplacian(self, w, plot=False, rollingMSE=False):
        last_date = datetime.datetime.strptime('2019-07-23','%Y-%m-%d').date()
        strides = len(self.df) - w + 1
        #yield_pred_evol = np.zeros((len(df.columns.get_level_values(1)), strides-1))
        yield_pred_evol = pd.DataFrame(columns=self.df.columns.get_level_values(1),index=self.df.index[w:])
        yield_pred_evol = yield_pred_evol.append(pd.DataFrame(index=[last_date]))
        daily_L = np.zeros((len(self.df.columns.get_level_values(0)), len(self.df.columns.get_level_values(0)), strides))
        daily_m = np.zeros((len(self.df.columns.get_level_values(0)), strides))
        mse = pd.DataFrame(index=self.df.index[w:],columns=['MSE'])
        
        print('window size: ', w)
        
        for i in range(strides-1):

            print('periods rolled: {}/{}\r'.format(i,strides))

            X = self.df.T.values[:,0+i:w-1+i]
            Y = self.df.T.values[:,1+i:w+i]

            _, _, m_latest, L_latest = self.coordinate_descent(X, Y, self.cd_iter, self.cd_weight)

            daily_L[:,:,i] = L_latest
            daily_m[:,[i]] = m_latest
            
            yield_pred = np.matmul( np.eye(len(m_latest)) - L_latest, self.df.T.values[:,[w-1+i]] ) + np.matmul(L_latest,m_latest)
            yield_pred_evol.iloc[[i],:] = yield_pred.T
            
            if rollingMSE:
                mse.iloc[i,:] = mean_squared_error(yield_pred_evol.iloc[[i],:],self.df.iloc[[w+i],:])
    #         print(yield_pred_evol)
    #         print('')
    #         print(df.T.values[:,windowsize:])
        
            #plot yield curves
            if plot:
                plt.figure(figsize=(15,5))
                plt.plot(self.df.T.values[:,w+i])
                plt.plot(yield_pred)
                plt.legend(['actual','prediction'])
                titlestring = 'index ' + str(w+i)
                plt.title(titlestring)
                plt.show()

        return yield_pred_evol, daily_L, daily_m, mse
    
    def plotLaplacian(self, L, whichStride):
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
        plt.figure(figsize=(7,5))
        plt.plot(self.df.columns.get_level_values(1), m[:,whichStride])
        plt.title(title)
        plt.xlabel('Maturity')
        plt.ylabel('$\mathbf{m}$')
        plt.grid()
        plt.tight_layout()
        plt.show()

    def get_learnedWeightMatrix(self, L, whichStride, wthresh=0.1, showGraph=False):
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
        if self.cd_weight == None:
            title = f'{self.currency} Swaps - Ending wealth by maturity: normal CD scheme'
        if self.cd_weight == 'exp':
            title = f'{self.currency} Swaps - Ending wealth by maturity: exponential CD scheme'

        profits.sum().plot(kind='bar',legend=False,title=title,rot=0,color='b')
        plt.xlabel('Maturity')
        plt.ylabel('Absolute profit')

    def cumulativeProfitsAggregated(self, profits):
        if self.cd_weight == None:
            title = f'{self.currency} Swaps - Daily cumulative profits aggregated across all portfolios: normal CD scheme'
        if self.cd_weight == 'exp':
            title = f'{self.currency} Swaps - Daily cumulative profits aggregated across all portfolios: exponential CD scheme'

        plt.plot(profits.cumsum().sum(axis=1))
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Absolute profit')

    def tradeAnalytics(self, profits, figsize=(5,5)):
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
        w_range = [20,30,50,70,100]
        c_thresh = [0, 0.0005, 0.001, 0.002, 0.005]
        agg_table = pd.DataFrame(index=[['w','w','w','w','w'],['20','30','50','70','100']],columns=[['$\tau$','$\tau$','$\tau$','$\tau$','$\tau$'],['0','0.0005','0.001','0.002','0.005']])
        largest_pl = pd.DataFrame(index=[['w','w','w','w','w'],['20','30','50','70','100']],columns=[['$\tau$','$\tau$','$\tau$','$\tau$','$\tau$'],['0','0.0005','0.001','0.002','0.005']])
        mse_table = pd.DataFrame(index=[['w','w','w','w','w'],['20','30','50','70','100']],columns=[['$\tau$','$\tau$','$\tau$','$\tau$','$\tau$'],['0','0.0005','0.001','0.002','0.005']])

        for i in range(len(w_range)):
            count = 0
            for j in range(len(c_thresh)):
                if count == 0:
                    y, L, m, _ = self.getLaplacian(w_range[i], plot=False, rollingMSE=False)
                    count += 1
                
                mse_table.iloc[i,j] = mean_squared_error(y.iloc[:-1,:], self.df.iloc[w_range[i]:,:])
                p, agg, _, _ = self.backtest(m[:,:-1], L[:,:,:-1], cash=1, normalize=True, curvatureThreshold=c_thresh[j])
                agg_table.iloc[i,j] = agg.iloc[-1]
                largest_pl.iloc[i,j] = p.sum().idxmax()
        
        return agg_table, largest_pl, mse_table

    def getHeatmap(self, title):
        agg_table, _, _ = self.stressTest_w_c()
        plt.figure(figsize=(15,10))
        sns.heatmap(agg_table.loc['w','$\tau$'].astype(int),cmap = "YlGnBu",annot=True,fmt='d',annot_kws={'size':13})
        plt.xlabel(r'$\tau$')
        plt.ylabel('$w$')
        plt.title(title)
        plt.show()


