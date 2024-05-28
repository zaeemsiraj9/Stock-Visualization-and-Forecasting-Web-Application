import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input, State
from dash.exceptions import PreventUpdate
from datetime import datetime as dt
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import nsepy as nse
import random
from sklearn.svm import SVR

app = dash.Dash(__name__, external_stylesheets=['assets\styles.css'])
server = app.server

def prediction(stock, n_days):
    # Import necessary libraries
    import numpy as np
    import yfinance as yf
    from sklearn.model_selection import train_test_split
    from sklearn.model_selection import GridSearchCV
    from sklearn.svm import SVR
    from datetime import date, timedelta
    import plotly.graph_objs as go

    # Load the data

    # Load data based on stock input
    if stock.upper() == "NIFTY50":
        df = pd.read_csv("NIFTY50.csv")
        df.set_index("Date", inplace=True)  # Assuming Date column for indexing
    else:
        df = yf.download(stock, period='6mo')
        df.reset_index(inplace=True)
        df['Day'] = df.index
    
    
    days = list()
    for i in range(len(df.Day)):
     days.append([i])
    X = days
    Y = df[['Close']]
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, shuffle=False)

    # Train and select the model
    gsc = GridSearchCV(estimator=SVR(kernel='rbf'), param_grid={'C': [0.001, 0.01, 0.1, 1, 100, 1000], 'epsilon': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100, 150, 1000], 'gamma': [0.0001, 0.001, 0.005, 0.1, 1, 3, 5, 8, 40, 100, 1000]}, cv=5, scoring='neg_mean_absolute_error', verbose=0, n_jobs=-1)
    y_train = y_train.values.ravel()
    grid_result = gsc.fit(x_train, y_train)
    best_params = grid_result.best_params_
    best_svr = SVR(kernel='rbf', C=best_params["C"], epsilon=best_params["epsilon"], gamma=best_params["gamma"], max_iter=-1)

    # Train the model
    best_svr.fit(x_train, y_train)

    # Predict and visualize the results
    output_days = list()
    last_day = x_test[-1][0]
    for i in range(1, n_days):
        last_day += 1
        # Introduce some randomness for a more dynamic forecast
        last_day += random.uniform(-1, 1)
        output_days.append([last_day])
    dates = []
    current = date.today()
    for i in range(n_days):
        current += timedelta(days=1)
        dates.append(current)

    # Plot the results
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.array(x_test).flatten(), y=y_test.values.flatten(), mode='markers', name='data'))
    fig.add_trace(go.Scatter(x=np.array(x_test).flatten(), y=best_svr.predict(x_test), mode='lines+markers', name='test'))
    fig = go.Figure()  # [Warning] Overwriting previous figure without reason

    fig.add_trace(go.Scatter(x=dates, y=best_svr.predict(output_days), mode='lines+markers', name='data'))
    fig.update_layout(title=f"Predicted Close Price of next " + str(n_days - 1) + " days of ", xaxis_title="Date", yaxis_title="Close Price")
    return fig


# Define the layout components
header = html.Div(
    [
        html.Img(src="assets/image7.png", width="640px", height="106px", style={"margin": "0 auto", "margin-top":"20px"}),
        html.P("CSE S7 Minor Mini-Project", style={"font-family": "Microsoft YaHei UI Light","font-size": "55px", "color": "white", "text-shadow": "2px 2px 4px rgba(0, 0, 128)"}),

        
    ],
    style={"text-align": "center"},
    className="header"
)
# Navigation component
item1 = html.Div(
    [
        html.Div([
            
            html.P("Enter Stock Ticker Below:", style={"margin-top":"420px","font-family": "Microsoft YaHei UI Light","font-size": "35px", "color": "white", "text-shadow": "2px 2px 4px rgba(0, 0, 128)"}),
        ]),

        html.Div([
            # stock code input
            dcc.Input(id='stock-code', type='text', placeholder='Enter stock code'),
            html.Button('Submit', id='submit-button')
        ], className="stock-input"),

        html.Div([
            # Date range picker input
            dcc.DatePickerRange(
                id='date-range', start_date=dt(2020, 1, 1).date(), end_date=dt.now().date(), className='date-input')
        ]),
        html.Div([
            html.Button('Get Stock Price', id='stock-price-button')
    ], className="selectors"),
    html.Div([
        html.Button('Get Indicators', id='indicators-button')
    ], className="selectors"),
    html.Div([
        dcc.Input(id='forecast-days', type='number', placeholder='Enter number of days')
    ], className="selectors"),
    html.Div([
        html.Button('Get Forecast', id='forecast-button')
    ], className="selectors")

    
    ],
    className="nav"
)

item3 = html.Div([
        html.P("In modern financial market, the most crucial problem is to find essential approach to outline and visualizing the predictions in stock-markets to be made by individuals in order to attain maximum profit by investments. The stock market is a transformative, on straight dynamical and complex system. Long term investment is one of the major investment decisions. Developing this simple project idea using dash library from Python, we can make dynamic plots of financial data of a specific company using tabular data provided by Yahoo Finance (yfinance) Python LIbrary. On top of it, we can use machine learning algorithm to predict the upcoming stock prices through SVR Model")
    ], className="description")
item4 = html.Div([
    html.Img(src="assets/image9.png", width="700px", height="356px", style={"opacity": 0.9,"border": "3px solid #B9EBFF","box-shadow": "2px 2px 30px rgba(165, 229, 255, 0.5)"})
    ], className="image")
item5 = html.Div([
    html.Img(src="assets/image9.png", width="470px", height="550px", style={"border": "3px solid #B9EBFF","box-shadow": "2px 2px 30px rgba(165, 229, 255, 0.5)"})
    ], className="image2")
item6 = html.Div([
    html.Img(src="assets/image2.jpg", width="400px", height="355px", style={"border": "3px solid #B9EBFF","box-shadow": "2px 2px 30px rgba(165, 229, 255, 0.5)"})], className="image3")
item7 = html.Div([
        html.P("CSE S7 Minor Mini-Project", style={"font-family": "Microsoft YaHei UI Light","font-size": "30px", "color": "white", "text-shadow": "2px 2px 4px rgba(0, 0, 128)"})
    ], className="image4")

# Content component
item2 = html.Div(
    [
        html.Div([html.Br()]),
        html.Div([html.Br()]),
        html.Div([html.Br()]),
        html.Div([], id="graphs-content"),
        html.Div([], id="main-content"),
        html.Div([], id="forecast-content")
    ],
    className="content"
)

# Set the layout
app.layout = html.Div(className='container', children=[header, item1, item3, item2, item4, item5, item6])

# Callbacks


# Callback for displaying stock price graphs
@app.callback(
    [Output("graphs-content", "children")],
    [
        Input("stock-price-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def stock_price(n, start_date, end_date, val):
  if n is None:
        return [""]
  if val is None:
        raise PreventUpdate
  else:
        if val.upper() == "NIFTY50":
            df = pd.read_csv("assets/NIFTY50.csv")  # Load from CSV
            df.set_index("Date", inplace=True)  # Assuming Date column for indexing
        else:
            if start_date is not None:
                df = yf.download(val, str(start_date), str(end_date))
            else:
                df = yf.download(val)

        df.reset_index(inplace=True)
        fig = px.line(df, x="Date", y=["Close", "Open"], title=f"Closing and Opening Price vs Date of {val}")
  first_pred_close = fig.data[0].y[0]
  last_pred_close = fig.data[0].y[-1]
  text_elements = [
        html.Div(
            dcc.Graph(figure=fig),style={"border": "3px solid #B9EBFF","box-shadow": "2px 2px 30px rgba(165, 229, 255, 0.5)","margin-right":"570px"}
            
        ),
        html.Div(
            [
                html.H5("Stock Prices:"),
                
                html.P(f"This graph is used to visualize the stock prices of an input company for an input date range. This specific graph shows the stock price of {val} from {start_date} to {end_date}"),
                html.P(f"First Day: {first_pred_close:.2f}"),
                html.P(f"Last Day: {last_pred_close:.2f}"),
            ],
            style={"font-family": "Microsoft YaHei UI Light","font-size": "20px", "color": "white",
        "background-color": "rgba(0, 0, 139, 0.5)",  # Translucent blue fill
        "border": "3px solid #B9EBFF","box-shadow": "2px 2px 30px rgba(165, 229, 255, 0.5)", "position": "absolute","top":"1350px","right":"40px","padding": "15px", "width":"500px"},
        ),
    ]
   
  return [html.Div(text_elements)]
# Callback for displaying indicators
@app.callback(
    [Output("main-content", "children")],
    [
        Input("indicators-button", "n_clicks"),
        Input('date-range', 'start_date'),
        Input('date-range', 'end_date')
    ],
    [State("stock-code", "value")]
)
def indicators(n, start_date, end_date, val):
  if n is None:
    return [""]
  if val is None:
    return [""]
  if start_date is None:
    df_more = yf.download(val)
  else:
    df_more = yf.download(val, str(start_date), str(end_date))

  df_more.reset_index(inplace=True)
  fig = get_more(df_more)
  text_elements = [
        html.Div(
            dcc.Graph(figure=fig),style={"border": "3px solid #B9EBFF","box-shadow": "2px 2px 30px rgba(165, 229, 255, 0.5)","margin-right":"570px"}
            
        ),
        html.Div(
            [
                html.H5("Exponential Average:"),
                html.P("An exponential moving average (EMA) is a widely used technical chart indicator that tracks changes in the price of a financial instrument over a certain period. Unlike simple moving average (SMA), EMA puts more emphasis on recent data points like the latest prices. Hence, the latter responds to a change in price points faster than the former."),
                
            ],
            style={"font-family": "Microsoft YaHei UI Light","font-size": "20px", "color": "white",
        "background-color": "rgba(0, 0, 139, 0.5)",  # Translucent blue fill
       "border": "3px solid #B9EBFF","box-shadow": "2px 2px 30px rgba(165, 229, 255, 0.5)", "position": "absolute", "bottom": "600px","right":"40px","padding": "15px","width":"500px"},
        ),
    ]
   
  return [html.Div(text_elements)]

def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x="Date", y="EWA_20", title=f"Exponential Moving Average vs Date")
    fig.update_traces(mode='lines+markers')
    return fig


# Callback for displaying forecast
@app.callback(
    [Output("forecast-content", "children")],
    [Input("forecast-button", "n_clicks")],
    [State("forecast-days", "value"),
     State("stock-code", "value")]
)
def forecast(n, n_days, val):
    if n is None:
        return [""]
    if val is None:
        raise PreventUpdate
    fig = prediction(val, int(n_days) + 1)  # No need to overwrite fig
    first_pred_close = fig.data[0].y[0]
    last_pred_close = fig.data[0].y[-1]

    # Create additional text elements to display the predicted close prices
    text_elements = [
        html.Div(
            dcc.Graph(figure=fig),style={"border": "3px solid #B9EBFF","box-shadow": "2px 2px 30px rgba(165, 229, 255, 0.5)","margin-right":"570px"}
            
        ),
        html.Div(
            [
                html.H5("Predicted Close Prices:"),
                html.P(f"This graph is used to predict the stock prices of an input company for an input number of days . This specific graph predicts the stock price for the next {n_days} days "),
                html.P(f"First Day: {first_pred_close:.2f}"),
                html.P(f"Last Day: {last_pred_close:.2f}"),
            ],
            style={"font-family": "Microsoft YaHei UI Light","font-size": "20px", "color": "white",
        "background-color": "rgba(0, 0, 139, 0.5)",  # Translucent blue fill
       "border": "3px solid #B9EBFF","box-shadow": "2px 2px 30px rgba(165, 229, 255, 0.5)", "position": "absolute", "bottom": "100px","right":"40px","padding": "15px","width":"500px"},
        ),
    ]
   
    return [html.Div(text_elements)]
    


if __name__ == '__main__':
    app.run_server(debug=True)

