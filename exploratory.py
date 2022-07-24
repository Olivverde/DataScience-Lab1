# Universidad del Valle de Guatemala
# Mineria de Datos
# HDT1-Exploratory Analysis
#------------------------------------
# Oliver de Leon 19270

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from reader import reader as Reader

class main(object):

    def __init__(self, csvDoc):
        # Universal Doc
        self.csvDoc = csvDoc
        # Classes
        R = Reader(csvDoc)
        self.df = R.pricing
        self.df = self.rangeAdding(self.percentile())
        self.remodel_bool()

    def percentile(self):
        x = self.df['SalePrice']
        self.Qs = x.quantile([0.25,0.5,0.75,1])
        threshold = x.quantile([0.33,0.67])
        firstRange, secondRange = threshold.iloc[0], threshold.iloc[1]

        return firstRange, secondRange

    def rangeAdding(self, ranges):
        fR, sR = ranges
        df = self.df
        
        df['SaleRange'] = df['SalePrice'].apply(
            lambda x: 'Low' if x <= fR 
            else ('Medium' if (x > fR and x <= sR) else 'High'))
    
        return df
    
    def remodel_bool(self):
        self.df['RemodelBool'] = (self.df.YearBuilt == self.df.YearRemodAdd)


    def avgPricing(self):

        df = self.df
        df = df[['SaleRange','SalePrice']]
        df = df.groupby('SaleRange').mean()

        """
        ax = df.plot.bar(y='SalePrice', use_index=True)
        plt.title('House Average Valuation')
        plt.ylabel('Average Value')
        plt.xlabel('House Valuation Type')
        plt.show()
        """
        return df
    
    def pricing_Boxplot(self):
        df = self.df
        print("------------------------------------------------------")
        print("MINIMUM HOUSE VALUATION: " + str(df['SalePrice'].min()))
        print("MAXIMUM HOUSE VALUATION: " + str(df['SalePrice'].max()))
        print("------------------------------------------------------")
        print("Q1 -> ", self.Qs.iloc[0])
        print("Q2 -> ", self.Qs.iloc[1])
        print("Q3 -> ", self.Qs.iloc[2])
        print("Q4 -> ", self.Qs.iloc[3])

        """        
        bp = self.df.boxplot(['SalePrice'])
        plt.title('Distribution & Central Tendency of House Pricing')
        plt.ylabel('House Value')
        plt.xlabel('Houses for Sale')
        plt.show()
        """

    def zoning_Count(self):
        df = self.df
        df = df.groupby(df['MSZoning']).size().to_frame('count')
        df = df.sort_values('count', ascending=False)

        ax = df.plot.bar(y='count', use_index=True)
        plt.title('Amount of Houses per Zone')
        plt.ylabel('Amount of Houses')
        plt.xlabel('Zones')
        plt.show()

    def lotArea_Zone_Pricing(self):

        df = self.df
        rl = df.copy()[df['MSZoning']=='RL'] 
        rm = df.copy()[df['MSZoning']=='RM']
        fv = df.copy()[df['MSZoning']=='FV']
        rh = df.copy()[df['MSZoning']=='RH']
        cAll = df.copy()[df['MSZoning']=='C (all)']

        ax1 = rl.plot.scatter(x = 'LotArea', y = 'SalePrice', label = 'RL', c = 'r')
        ax2 = rm.plot.scatter(x = 'LotArea', y = 'SalePrice', c = 'b', label = 'RM', ax=ax1)
        ax3 = fv.plot.scatter(x = 'LotArea', y = 'SalePrice', c = 'g', label = 'FV', ax=ax1)
        ax4 = rh.plot.scatter(x = 'LotArea', y = 'SalePrice', c = 'yellow', label = 'RH', ax=ax1)
        ax5 = cAll.plot.scatter(x = 'LotArea', y = 'SalePrice', c = 'pink', label = 'C (All)', ax=ax1)
        plt.title('House Pricing according to Lot Area per Zone')
        plt.ylabel('House Pricing')
        plt.xlabel('Lot Area')
        plt.show()

    def lotArea_slope_Pricing(self):
        df = self.df
        gtl = df.copy()[df['LandSlope']=='Gtl'] 
        mod = df.copy()[df['LandSlope']=='Mod']
        sev = df.copy()[df['LandSlope']=='Sev']

        ax1 = gtl.plot.scatter(x = 'LotArea', y = 'SalePrice', label = 'Gentle', c = 'r')
        ax2 = mod.plot.scatter(x = 'LotArea', y = 'SalePrice', c = 'b', label = 'Moderate', ax=ax1)
        ax3 = sev.plot.scatter(x = 'LotArea', y = 'SalePrice', c = 'g', label = 'Severe', ax=ax1)
        plt.title('House Pricing according to Land Slope & Lot Area')
        plt.ylabel('House Pricing')
        plt.xlabel('Lot Area')
        plt.show()

    def lotArea_Qlty_Pricing(self):
        df = self.df
        tier_A_S = df.copy()[df['OverallQual']>=7] 
        tier_C_B = df.copy()[(df['OverallQual']>=3) & (df['OverallQual']<7)]
        tier_D = df.copy()[df['OverallQual']<3]

        ax1 = tier_A_S.plot.scatter(x = 'LotArea', y = 'SalePrice', label = 'Tier A - S', c = 'yellow')
        ax2 = tier_C_B.plot.scatter(x = 'LotArea', y = 'SalePrice', c = 'orange', label = 'Tier C - B', ax=ax1)
        ax3 = tier_D.plot.scatter(x = 'LotArea', y = 'SalePrice', c = 'purple', label = 'Tier D', ax=ax1)
        plt.title('House Pricing according to Overall Quality & Lot Area')
        plt.ylabel('House Pricing')
        plt.xlabel('Lot Area')
        plt.show()

    def numerical_Corr(self):
        df = self.df
        df = df[['SalePrice','LotFrontage', 'MasVnrArea', 'BsmtFinSF1',
                'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
                '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath',
                'BsmtHalfBath', 'FullBath', 'HalfBath', 'TotRmsAbvGrd',
                'Fireplaces', 'GarageCars','GarageArea', 'WoodDeckSF',
                'OpenPorchSF', 'EnclosedPorch','3SsnPorch', 'ScreenPorch',
                'PoolArea', 'MiscVal']]
        corr_df = df.corr(method='pearson')
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_df, annot=False)
        plt.title('Correlation among Numerical Variables') 
        plt.show()

    def constr_Year(self):
        
        df = self.df
        df = df.groupby(df['YearBuilt']).size().to_frame('count')
        df = df.sort_values('YearBuilt', ascending=True)

        ax = df.plot.bar(y='count', use_index=True)
        plt.title('House Construction Year')
        plt.ylabel('Amount of Built Houses')
        plt.xlabel('Year')
        plt.locator_params(axis='x', nbins=12)
        plt.show()

    def remodel_Year(self):
        
        df = self.df
        df = df.copy()[df['RemodelBool']==True]
        
        df = df.groupby(df['YearRemodAdd']).size().to_frame('count')
        df = df.sort_values('YearRemodAdd', ascending=True)

        ax = df.plot.bar(y='count', use_index=True)
        plt.title('House Remodelation Year')
        plt.ylabel('Amount of Remodeled Houses')
        plt.xlabel('Year')
        plt.locator_params(axis='x', nbins=12)
        plt.show()
        
    def lotArea_Remodel_Pricing(self):
        df = self.df
        remodel = df.copy()[df['RemodelBool']==True] 
        remodel_nt = df.copy()[df['RemodelBool']==False]
        
        ax1 = remodel.plot.scatter(x = 'LotArea', y = 'SalePrice', label = 'Remodeled', c = 'blue')
        ax2 = remodel_nt.plot.scatter(x = 'LotArea', y = 'SalePrice', c = 'red', label = 'Not Remodeled', ax=ax1)

        plt.title('House Pricing according to Remodeling Status & Lot Area')
        plt.ylabel('House Pricing')
        plt.xlabel('Lot Area')
        plt.show()

exp = main('./data/train.csv')
# exp.Pricing_Boxplot()
# print(exp.avgPricing())
# print(exp.zoning_Count())
# exp.lotArea_Zone_Pricing()
# exp.lotArea_slope_Pricing()
# exp.lotArea_Qlty_Pricing()
# exp.numerical_Corr()
# exp.constr_Year()
exp.remodel_Year()
# exp.lotArea_Remodel_Pricing()
