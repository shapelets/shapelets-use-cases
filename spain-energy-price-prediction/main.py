import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


from shapelets import init_session
from shapelets.dsl.data_app import DataApp
from shapelets.model import  Sequence,Dataframe
from shapelets.model.image import Image
from shapelets.dsl import dsl_op


from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import lightgbm as lgb
import xgboost as xgb



# Start shapelets process and init session as admin
client = init_session("admin", "admin")

# Create a dataApp
app = DataApp(
    name="Spain Energy Price Prediction",
    description="Use case for predict price of energy with Spanish market."
)


# Charge data with pandas

df = pd.read_csv("shapelets_usecase_resources/data.csv")
df.datetime = pd.to_datetime(df.datetime)
df = df.set_index('datetime')


df_price = pd.read_csv("shapelets_usecase_resources/spot_esp_20150101_20220415.csv")
df_price = df_price[:-1]
df_price.timestamp = pd.to_datetime(df_price.timestamp)
df_price = df_price.set_index('timestamp')

df['precio'] = df_price.SPOTPRICE_españa


df_real = pd.read_csv("shapelets_usecase_resources/data_treal.csv")
df_real.datetime = pd.to_datetime(df_real.datetime)
df_real = df_real.set_index('datetime')




# Create Shapelets collections
collection = client.create_collection(name="OMIE Data (Hour)",description="Generación programada y precio")

for column in df.columns:
    client.create_sequence(dataframe=df.loc[:, column].to_frame(), name=column, collection=collection)


collection2 = client.create_collection(name="OMIE Data (10M)",description="Generación en tiempo real y demanda real")

for column in df_real.columns:
    client.create_sequence(dataframe=df_real.loc[:, column].to_frame(), name=column, collection=collection2)



app.place(app.markdown("""
    # Introduction



    Hello! 

    Would you like to know if it is possible to predict energy prices using Machine Learning? Do you think it is possible? Maybe this DataApp will interest you!

    The electricity market is regulated by REE (Red Electrica Española), and acts as an intermediary between companies that generate energy and 
    companies that buy this energy to distribute it to the customers. 

    Generated energy can come from different forms of generation, such as wind energy, solar energy with photovoltaic panels, nuclear energy... etc. 




    __How does the Spanish market work?__

    The price of electricity is calculated by means of a matching offer.

    Every day at 16:00 the next day's energy prices is calculated, and it is calculated by ordering from lowest to highest the prices at which energy 
    sellers want to sell the energy produced, and by ordering from highest to lowest the prices at which buyers want to buy the energy. The point where the two 
    lines intersect is called the matching point and that price is established as the energy price for that hour. 

    This process is repeated for each hourly segment of the following day and is managed by Operador del Mercado Ibérico-Polo Español (OMIE).


"""))



aux = np.array([0,0,0,1,1,1,1,2,2,2,2,2,2,2,2,2,4,5,5,5,5,6,7,7,7,7,7,7,7,8,8,8,8,8,8])
aux_2 = np.array([8, 8, 8, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 3, 2, 2, 2,2, 2, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])


fig, ax = plt.subplots(figsize=(15,7))
plt.plot(aux,label='Sellers')
plt.plot(aux_2,label='Buyers')

index_union = np.where(aux == aux_2)[0][0]

coordenadas_circulo = (index_union,aux[index_union])

circle2 = plt.Circle(coordenadas_circulo, 0.8, color='r', fill=False)
ax.add_patch(circle2)

plt.axvline(x = index_union, color = 'k', linestyle = '--')

ax.annotate("", xy=(35, -0.2), xytext=(-0.8, -0.2),arrowprops=dict(arrowstyle="->",color="C0"))
ax.annotate("", xy=(-0.7, 8), xytext=(-0.7, -0.2),arrowprops=dict(arrowstyle="->",color="tab:orange"))

ax.annotate("Energy [MWh]",xy=(17,0),color="C0",)
ax.annotate("Price [€]",xy=(-1.4,4),color="tab:orange",rotation=90)

plt.xlabel("Energy [MWh]")
plt.ylabel("Price [€]")

plt.gca().axes.get_yaxis().set_visible(False)
plt.gca().axes.get_xaxis().set_visible(False)

plt.legend()


img = app.image(fig)
app.place(img)







app.place(app.markdown("""



    The left side of the graph is referred to as matched bids (the sell and buy bids that become firm commitments to deliver energy). 



    __The objective of this DataApp is to try to predict the energy matching price for each of the hours of a day.__


    ---

    # Data


    The data used in this project is a dataset extracted from the OMIE API. This API has 1400 indicators available for analysis, 
    among them information on scheduled power generation, real time generation, energy price including intraday market session prices. 

    For this use case a few KPIs have been selected, because they are the most complete data (no missing data) and in the case of power generation 
    indicators correspond to the energy sources that have the greatest impact on the Spanish energy market.



    Here's the list of KPIs used:


    - __Spot Price__                            Daily SPOT market price                             _Freq: Hourly data_
    - __Real Demand__                           Real energy demand in Spanish territory               _Freq: 10 Mins data_ 
    - __Scheduled Generation Hydraulics__       Scheduled hydroelectric power generation         _Freq: Hourly data_
    - __Scheduled Generation Nuclear__          Nuclear programmed power generation            _Freq: Hourly data_
    - __Scheduled Generation Combined cycle__   Combined Cycle programmed power generatio    _Freq: Hourly data_
    - __Scheduled Generation Wind Power__       Scheduled wind energy generation             _Freq: Hourly data_
    - __Scheduled Generation co-generation__    Programmed co-generation power generation         _Freq: Hourly data_
    - __Real time Generation Hydraulics__       Real-time generation of hydroelectric energy     _Freq: 10 Mins data_ 
    - __Real time Generation nuclear__          Real-time nuclear power generation        _Freq: 10 Mins data_ 
    - __Real time Generation Combined cycle__   Real-time combined cycle power generation _Freq: 10 Mins data_ 

    We can visualize the data below in this amazing Shapelets graphs! 


"""))




# Create tabs layout
tabs_fp = app.tabs_flow_panel("OMIE Data (Hour)")
app.place(tabs_fp)

# Add tabs
tab1 = app.vertical_flow_panel()
tabs_fp.place(tab1,"Spot Price")

sequence = next(col for col in client.get_collection_sequences(collection) if col.name == "Spot_Price_Spain")
line_chart1 = app.line_chart(title=sequence.name, sequence=sequence)
tab1.place(line_chart1)

tab2 = app.vertical_flow_panel()
tabs_fp.place(tab2,"Nuclear Energy")

sequence = next(col for col in client.get_collection_sequences(collection) if col.name == "GP_Nuclear")
line_chart2 = app.line_chart(title=sequence.name, sequence=sequence)
tab2.place(line_chart2)



tab3 = app.vertical_flow_panel()
tabs_fp.place(tab3,"Hidraulic Energy")

sequence = next(col for col in client.get_collection_sequences(collection) if col.name == "GP_Hidraulica")
line_chart3 = app.line_chart(title=sequence.name, sequence=sequence)
tab3.place(line_chart3)


tab4 = app.vertical_flow_panel()
tabs_fp.place(tab4,"Eólica Energy")

sequence = next(col for col in client.get_collection_sequences(collection) if col.name == "GP_Eolica")
line_chart4 = app.line_chart(title=sequence.name, sequence=sequence)
tab4.place(line_chart4)


tab5 = app.vertical_flow_panel()
tabs_fp.place(tab5,"Combined Cycle Energy")

sequence = next(col for col in client.get_collection_sequences(collection) if col.name == "GP_Ciclo_Combinado")
line_chart5 = app.line_chart(title=sequence.name, sequence=sequence)
tab5.place(line_chart5)


tab6 = app.vertical_flow_panel()
tabs_fp.place(tab6,"Cogeneracion Energy")

sequence = next(col for col in client.get_collection_sequences(collection) if col.name == "GP_Cogeneracion")
line_chart6 = app.line_chart(title=sequence.name, sequence=sequence)
tab6.place(line_chart6)






# Create tabs layout
tabs_fp = app.tabs_flow_panel("OMIE Data (10Min)")
app.place(tabs_fp)

# Add tabs
tab1 = app.vertical_flow_panel()
tabs_fp.place(tab1,"Demanda real")

sequence = next(col for col in client.get_collection_sequences(collection2) if col.name == "demanda_real")
line_chart1 = app.line_chart(title=sequence.name, sequence=sequence)
tab1.place(line_chart1)

tab2 = app.vertical_flow_panel()
tabs_fp.place(tab2,"GTReal hidráulica ")

sequence = next(col for col in client.get_collection_sequences(collection2) if col.name == "GTR_hidraulica")
line_chart2 = app.line_chart(title=sequence.name, sequence=sequence)
tab2.place(line_chart2)



tab3 = app.vertical_flow_panel()
tabs_fp.place(tab3,"GTReal nuclear")

sequence = next(col for col in client.get_collection_sequences(collection2) if col.name == "GTR_nuclear")
line_chart3 = app.line_chart(title=sequence.name, sequence=sequence)
tab3.place(line_chart3)


tab4 = app.vertical_flow_panel()
tabs_fp.place(tab4,"GTReal ciclo combinado")

sequence = next(col for col in client.get_collection_sequences(collection2) if col.name == "GTR_ciclo_combinado")
line_chart4 = app.line_chart(title=sequence.name, sequence=sequence)
tab4.place(line_chart4)













app.place(app.markdown("""


    ---

    # Data Transform

    The objective of this use case is to __predict the price of energy__, so our target variable will be the Spot Price KPI,
     and our predictor variables, all the others.

    To do this we have to take into consideration that the data is not at the same frequency, we need to resample the data. 


    We have some initial business knowledge about this data, and we know that price evolution throughout the day has a very similar shape over the different days. 
    similar over the different days of the month. We can see it in this chart, which represents the price data for all the days of January 2020. 


"""))


df_fil = df[['Spot_Price_Spain']].loc["2020-01-01":"2020-01-30"]
df_fil = df_fil.reset_index()
df_fil['color'] = df_fil.datetime.dt.day_name()

df_fil['hora'] = df_fil.datetime.dt.time.astype(str).str[:5]
df_fil['semana'] = df_fil.datetime.dt.week

fig, axs = plt.subplots(2,2,figsize=(15,10))


#plt.figure(figsize=(15,5))
sns.lineplot(data=df_fil[df_fil.semana==1],x="hora",y="Spot_Price_Spain",hue="color",ax=axs[0][0]).set_title("Week 1")
axs[0][0].tick_params(axis='x', rotation=90)

sns.lineplot(data=df_fil[df_fil.semana==2],x="hora",y="Spot_Price_Spain",hue="color",ax=axs[0][1]).set_title("Week 2")
axs[0][1].tick_params(axis='x', rotation=90)

handles, labels = axs[0][1].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right')


sns.lineplot(data=df_fil[df_fil.semana==3],x="hora",y="Spot_Price_Spain",hue="color",ax=axs[1][0]).set_title("Week 3")
axs[1][0].tick_params(axis='x', rotation=90)

sns.lineplot(data=df_fil[df_fil.semana==4],x="hora",y="Spot_Price_Spain",hue="color",ax=axs[1][1]).set_title("Week 4")
axs[1][1].tick_params(axis='x', rotation=90)

fig.subplots_adjust(hspace=0.3) 

for axg in axs:
    for ax in axg:
        ax.set(ylabel="Energy price [€]",xlabel="")
        ax.get_legend().remove()


img = app.image(fig)
app.place(img)


app.place(app.markdown("""

    Knowing this, you can use a dimensionality reduction algorithm such as PCA to reduce the dimensionality of the target variable. 


    ## Dimensionality reduction of the target variable.

    Currently, being in hourly format, I have 24 data points for each day. I am going to use the PCA algorithm to reduce the dimensionality for each day.
     The main goal is to be able to summarize the information from the 24 points into a single variable. We can do it easily with python and sci-kit learn!




    ```python
    # Create a dataset for PCA using price data

    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA


    # Standard Scaler
    sc = StandardScaler()
    sc.fit(price_ts_np)
    X_train_std = sc.transform(price_ts_np)

    # PCA
    pca = PCA(1)
    data_fitted = pca.fit_transform(X_train_std)

    ```

    With just a single variable, I can explain 96% of the explanied variance!

    But, With this new target variable exist a problem. The data do not have the same time frequency!
    The target variable has daily data, some variables have hourly data, and another one every 10 minutes. Let's fix that!


    Let's resample the data so that they all have the same frequency, daily data.

    The data that has frequency every 10Mins will first be resampled to hourly.

    ```python
    df.resample('H').agg(['min','mean','max'])
    ```
    Now, for each variable that had frequency 10Min, we will have 3 new columns, indicating the minimum, average and maximum values. 

    With all the variables, repeat the same operation to have the data with daily frequency. 


    ```python
    df.resample('D').agg(['min','mean','max'])
    ```

    Now we can use models that are not specifically designed for time series, such as RandomForest, XGBoost or LightGBM.

    We have added some columns so that the model can know information about the day of the week and the day of the month of the data.

    _TODO: You can try adding information about the season of the year or month information, to see if the results improve!_



"""))

# RESAMPLE DATA
df_real_resampled = df_real.resample('H').agg(['min','mean','max'])


for col in df_real_resampled.columns:
    name = '_'.join(col)
    df[name] = df_real_resampled[col[0]][col[1]]
    


# REMOVE OUTLIER
    
aux_1 = df.loc['2022-03-04':'2022-03-06','precio']
aux_2 = df.loc['2022-03-10':'2022-03-12','precio']

aux = pd.DataFrame(pd.concat([aux_1,aux_2]))
input_values = aux.groupby(aux.index.to_series().dt.hour).max().values

df.loc['2022-03-07','precio'] = input_values.reshape(-1)
df.loc['2022-03-08','precio'] = input_values.reshape(-1)
df.loc['2022-03-09','precio'] = input_values.reshape(-1)



# PROCESSING TARGET

daterange = pd.date_range(df.index.min(), df.index.max(),freq='D')

price_ts = []
for dia in daterange:
        data = df.loc[dia.strftime("%Y-%m-%d"),'precio'].values.reshape(-1)
        price_ts.append(data)

price_ts_np = np.array(price_ts)




# Standard Scaler
sc = StandardScaler()
sc.fit(price_ts_np)
X_train_std = sc.transform(price_ts_np)

# PCA
pca = PCA(1)
data_fitted = pca.fit_transform(X_train_std)


df_pca = pd.DataFrame(data={'pca':data_fitted.reshape(-1)},index=daterange)



df_resample = df[['GP_Hidraulica', 'GP_Nuclear', 'GP_Ciclo_Combinado', 'GP_Eolica',
                   'GP_Cogeneracion', 'GTR_hidraulica_min',
                   'GTR_hidraulica_mean', 'GTR_hidraulica_max', 'GTR_nuclear_min',
                   'GTR_nuclear_mean', 'GTR_nuclear_max', 'GTR_ciclo_combinado_min',
                   'GTR_ciclo_combinado_mean', 'GTR_ciclo_combinado_max', 'GTR_eolica_min',
                   'GTR_eolica_mean', 'GTR_eolica_max', 'demanda_real_min',
                   'demanda_real_mean', 'demanda_real_max']]

df_min = df_resample.resample('D').min()
df_min.columns = [f'min_{x}' for x in df_min.columns]
df_mean = df_resample.resample('D').mean()
df_mean.columns = [f'mean_{x}' for x in df_mean.columns]
df_max = df_resample.resample('D').max()
df_max.columns = [f'max_{x}' for x in df_max.columns]

df_agg = pd.concat([df_min,df_mean,df_max],axis=1)




for columna in df_agg.columns:
    if 'gp48' not in columna.lower():
        nombre = f"{columna}_lag{1}"
        lag = 1
        df_agg[nombre] = df_agg[columna].shift(lag)
        df_agg.drop(columna,axis=1,inplace=True)
    elif 'gp48' in columna.lower():
        for i in range(1,3):
            nombre = f"{columna}_lag{i}"
            lag = i
            df_agg[nombre] = df_agg[columna].shift(lag)
    else:
        pass


df_mod2 = pd.concat([df_agg,df_pca],axis=1)
# TARGET VARIABLE
df_mod2['price_future'] = df_mod2.pca.shift(1)
df_mod2 = df_mod2.dropna()



df_mod2['dia_semana'] = pd.date_range(df_mod2.index.min(),df_mod2.index.max(),freq='H').to_series().dt.dayofweek + 1
df_mod2['ciclo_dia'] = df_mod2['dia_semana'] / 7
df_mod2['dias_mes'] = pd.date_range(df_mod2.index.min(),df_mod2.index.max(),freq='H').to_series().dt.daysinmonth
df_mod2['dia_mes'] = df_mod2.index.to_series().dt.day / df_mod2['dias_mes']
df_mod2['ciclo_mes'] = df_mod2['dia_mes'].apply(lambda x: round(x,3))
df_mod2 = df_mod2.drop(['dia_semana','dias_mes','dia_mes','pca'],axis=1)
df_mod2 = df_mod2.dropna()

train_data = df_mod2[:"2022-03-31"]
prediction_data = df_mod2["2022-04-01":]


y = train_data['price_future']
X = train_data.drop('price_future',axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


app.place(app.markdown("""

    ---

    # Models

        This problem can be approached with different points of view. 

    - As an autoregressive time series, predicting the value of the principal component of the daily data.
    - As a multivariate time series problem, with the different KPIs we have. 
    - As a regression problem.

    For this approach, I have treated the problem as a regression problem, in which, with the processing of the previous point, I predict the value of the principal component that I have predicted. 
    the value of the principal component that I have calculated for the target. 

    We have used 3 algorithms:

    - the RandomForestRegressor algorithm from the `Scikit-Learn` package, and it is implemented like this:

    - The LightGBM algorithm, from microsoft.

    - The XGBoost algorithm. 


    How have we implemented the algorithms? Easy, look at this example.

    ```python
    from sklearn.ensemble import RandomForestRegressor

    model = RandomForestRegressor(random_state=2022)

    # Train model
    model.fit(X_train,y_train)

    # Get predictions
    predictions = model.predict(X_test)

    ```

     With Shapelets you have all the power of python and its different packages, so it is as easy as importing and running them!

    To train the algorithm I have used data from all the KPIs from January 2015 to March 31, 2022.

    To evaluate the performance of the models, we are going to split the training dataset to have a train and a test, selecting 30% of the data randomly, and calculate some metrics. You can see wich model is working better below!





"""))

def regression_metrics(y_true,y_pred):
    from sklearn.metrics import mean_absolute_percentage_error,mean_absolute_error, mean_squared_error
    errors = {"MSE": round(mean_squared_error(y_true,y_pred),4),
              "RMSE": round(mean_squared_error(y_true,y_pred,squared=False),4),
              "MAE": round(mean_absolute_error(y_true,y_pred),4),
              "MAPE": round(mean_absolute_percentage_error(y_true,y_pred),4)}
    return errors




regr = RandomForestRegressor(random_state=0,n_jobs=-1,**{'n_estimators': 400,
                                                         'min_samples_split': 2,
                                                         'min_samples_leaf': 1,
                                                         'max_features': 'sqrt',
                                                         'max_depth': None,
                                                         'bootstrap': False})

# Train model
regr.fit(X_train,y_train)
preds1 = regr.predict(X_test)
errores1 = regression_metrics(y_test,preds1)
errores1['model'] = "RandomForestRegressor"



model_ = lgb.LGBMRegressor(random_state=0,**{'n_estimators': 400,
                                             'min_samples_split': 2,
                                             'min_samples_leaf': 1,
                                             'max_features': 'sqrt',
                                             'max_depth': None,
                                             'bootstrap': False})



# Train model
model_.fit(X_train,y_train)
preds2 = model_.predict(X_test)
errores2 = regression_metrics(y_test,preds2)
errores2['model'] = "LightGBM"





model2_ = xgb.XGBRegressor(random_state=0,**{'n_estimators': 400,
                                             'min_samples_split': 2,
                                             'min_samples_leaf': 1,
                                             'max_features': 'sqrt',
                                             'max_depth': None,
                                             'bootstrap': False})


# Train model
model2_.fit(X_train,y_train)
preds3 = model2_.predict(X_test)
errores3 = regression_metrics(y_test,preds3)
errores3['model'] = "XGBoost"

df_errors = pd.DataFrame([errores1,errores2,errores3])
colsnames =  df_errors.columns.values.tolist()
cols_ordered = [colsnames[-1]] + colsnames[:-1]

df_errors = df_errors[cols_ordered]
df_errors_shapelets = client.create_dataframe(df_errors, name="Metrics Dataframes", description="Example description")


table = app.table(df_errors_shapelets)
app.place(table)


app.place(app.markdown("""

    Here's a short description about errors:
    - _MSE_ is the mean squared error.
    - _RMSE_ is the root of mean squarred error.
    - _MAE_ is the mean of absolute error.
    - _MAPE_ is the sum of the individual absolute errors divided by each period separately. It is the average of the percentage errors.



    --- 
    # Prediction


    Want to see the models in action?

    As of the date of development of this DataApp, we have data up to mid-April, so we will predict spot prices from April 1 to April 14, 2022.

"""))

prediction_data_model_input = prediction_data.drop('price_future',axis=1)

prediction_data['preds_rfr'] = regr.predict(prediction_data_model_input)
prediction_data['preds_lgbm'] = model_.predict(prediction_data_model_input)
prediction_data['preds_xgboost'] = model2_.predict(prediction_data_model_input)



dataset_plot = []


for i in pd.date_range("2022-04-01",periods=14,freq='D'):

    X_val_d = prediction_data.loc[i.strftime("%Y-%m-%d")]

    original_data = sc.inverse_transform(pca.inverse_transform(X_val_d['price_future'])).reshape(-1)
    preds_rfr     = sc.inverse_transform(pca.inverse_transform(X_val_d['preds_rfr'])).reshape(-1)
    preds_lgbm     = sc.inverse_transform(pca.inverse_transform(X_val_d['preds_lgbm'])).reshape(-1)
    preds_xgboost     = sc.inverse_transform(pca.inverse_transform(X_val_d['preds_xgboost'])).reshape(-1)

    new_index = pd.date_range(i.strftime("%Y-%m-%d"),periods=24,freq='H')
    new_df = pd.DataFrame({'real_data':original_data,
                           'prediction_rf':preds_rfr,
                           'prediction_lighgbm':preds_lgbm,
                           'prediction_xgboost':preds_xgboost},
                            index=new_index)

    dataset_plot.append(new_df)

dataset_plot = pd.concat(dataset_plot)
shapelets_dataframe = client.create_dataframe(dataset_plot, name="Dataset for prediction plots", description="Description test")






def plot_selected_day(dataf: Dataframe, dia: int, model:str )-> Image:
    import matplotlib.pyplot as plt
    dataf = dataf.dataframe
    print(dataf)

    date = f"2022-04-{dia}"
    datafil =  dataf.loc[:,date]

    fig = plt.figure(figsize=(12,4)) 

    plt.title(f'Hourly prediction for day {dia}')
    plt.plot(datafil.loc[:, "real_data"],label="Real Price")


    if model == "All Models":
        plt.plot(datafil.loc[:, "prediction_rf"],label="Prediction RandomForestRegressor")
        plt.plot(datafil.loc[:, "prediction_lighgbm"],label="Prediction LightGBM")
        plt.plot(datafil.loc[:, "prediction_xgboost"],label="Prediction XGBoost")
    elif model == "RandomForestRegressor":
        plt.plot(datafil.loc[:, "prediction_rf"],label="Prediction RandomForestRegressor")
    elif model == "LightGBM":
        plt.plot(datafil.loc[:, "prediction_lighgbm"],label="Prediction LightGBM")
    elif model == "XGBoost":
        plt.plot(datafil.loc[:, "prediction_xgboost"],label="Prediction XGBoost")

    plt.ylabel("Price [€]")
    plt.legend()

    return Image(fig)


client.register_custom_function(plot_selected_day)

# Create an horizontal_flow_panel
hf = app.horizontal_flow_panel("")


# Create a slider
slider = app.slider(title="Select a day", min_value=1, max_value=14, step=1, default_value=1)

# Place slider into the Dataapp
hf.place(slider,width=9)


selector = app.selector(["All Models", "RandomForestRegressor", "LightGBM","XGBoost"], value="All Models")
app.place(selector)


fig = dsl_op.plot_selected_day(shapelets_dataframe,slider,selector)

button = app.button(text="Get predictions")
button.on_click([fig])

hf.place(button,width=3)

app.place(hf)

img_fig = app.image(fig)
app.place(img_fig)



app.place(app.markdown("""
    ---

    # Conclussion



    We have reached the end!

    We have seen how we can process time series and how we can apply algorithms with that data.
    In addition, we have seen that the price of electricity is possible to predict, maybe with a more extensive data source, or more data sources, 
     it can be better tuned. 

    I leave that to you! 

"""))


# Register the Dataapp
client.register_data_app(app)
