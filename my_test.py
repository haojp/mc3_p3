import StrategyLearner as sl
import pandas as pd

temp = sl.StrategyLearner()
temp.addEvidence()

df_trades = temp.testPolicy()
print df_trades
