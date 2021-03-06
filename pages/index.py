# Imports from 3rd party libraries
import dash
import re
import dash_bootstrap_components as dbc
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import tensorflow as tf
from dash.dependencies import Input, Output, State
import joblib
# Imports from this application
from app import app
from assets.variables import STOPWORDS, COLUMN_NAMES, TIME_PERIODS, DAYS, MONTHS, CATEGORY_NAMES, SUB_CATEGORY_NAMES, COUNTRY_NAMES, LOCATION_NAMES


class BlurbsBlock(tf.keras.Model):
    """
    Class inheriting from tf.keras.Model that defines an lstm block for the blurbs input

    inputs: units for layer 1, dropout for layer 1, max words, embedding dimension, and the input length
    """

    def __init__(self, units, do, max_words, embedding_dim):
        super(BlurbsBlock, self).__init__(name='BlurbsBlock')

        self.embed = tf.keras.layers.Embedding(max_words, embedding_dim)

        self.lstm1 = tf.keras.layers.LSTM(
            units, dropout=do, kernel_initializer='glorot_normal')

    def call(self, input_tensor):
        embed = self.embed(input_tensor)
        x = self.lstm1(embed)
        return x


class KickstarterModel(tf.keras.Model):
    """
    Class defining KickstarterModel

    Inputs: tokenized name,
    a tokenized blurb, and the tabular Kickstarter data.

    Outputs: 0-1 value where if it's > 0.5 then the kickstarter is predicted
    to be successful and < 0.5 the kickstarter is predicted to be non-successful}
    """

    def __init__(self, max_words, embedding_dim):
        super(KickstarterModel, self).__init__(name='NamesBlock')

        self.dense1 = tf.keras.layers.Dense(
            64, kernel_initializer='glorot_normal', kernel_regularizer='l2')
        self.do1 = tf.keras.layers.Dropout(0.25)

        self.dense2 = tf.keras.layers.Dense(
            32, kernel_initializer='glorot_normal', kernel_regularizer='l2')
        self.do2 = tf.keras.layers.Dropout(0.85)

        self.dense3 = tf.keras.layers.Dense(
            16, kernel_initializer='glorot_normal', kernel_regularizer='l2')
        self.do3 = tf.keras.layers.Dropout(0.25)

        self.blurbs_block = BlurbsBlock(
            units=64, do=0.95, max_words=max_words, embedding_dim=embedding_dim)

        self.concat = tf.keras.layers.Concatenate()
        self.act = tf.keras.layers.ReLU(threshold=0.078)
        self.act2 = tf.keras.layers.ReLU(threshold=0.045)
        self.act3 = tf.keras.layers.ReLU(threshold=0.03)

        self.classifier = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs):
        blurb = self.blurbs_block(inputs[0])
        tabular = self.dense1(inputs[1])
        x = self.do1(tabular)
        x = self.concat([blurb, x])
        x = self.act(x)

        x = self.dense2(x)
        x = self.do2(x)
        x = self.act2(x)

        x = self.dense3(x)
        x = self.do3(x)
        x = self.act3(x)
        return self.classifier(x)


max_words = 50000
embedding_dim = 100

# model = KickstarterModel(max_words, embedding_dim)
# # model.compile(optimizer=tf.keras.optimizers.Nadam(),
# #               loss='binary_crossentropy')
# model.load_weights('assets/Kickstarter_model/')


def get_prediction(name, blurb, usd_goal, creation_to_launch_days,
                   campaign_days, category, country, sub_category,
                   launch_day, deadline_day, launch_month, deadline_month,
                   launch_time, deadline_time, location):
    """
    Function for checking if all features are correctly entered and to return
    a message to the user to fix their input if not. Otherwise returns a price
    prediction by passing features to the price prediction model.
    """
    df = pd.DataFrame(columns=COLUMN_NAMES,
                      index=[0]
                      )
    df = df.fillna(0)

    try:
        df.iloc[0, df.columns.get_loc('name_length')] = len(name.split())
        df.iloc[0, df.columns.get_loc('blurb_length')] = len(blurb.split())
        df.iloc[0, df.columns.get_loc('usd_goal')] = usd_goal
        df.iloc[0, df.columns.get_loc(
            'creation_to_launch_days')] = creation_to_launch_days
        df.iloc[0, df.columns.get_loc('campaign_days')] = campaign_days
        df.iloc[0, df.columns.get_loc('category_{}'.format(category))] = 1
        df.iloc[0, df.columns.get_loc('country_{}'.format(country))] = 1
        df.iloc[0, df.columns.get_loc('sub_category_{}'.format(sub_category))] = 1
        df.iloc[0, df.columns.get_loc('launch_day_{}'.format(launch_day))] = 1
        df.iloc[0, df.columns.get_loc('deadline_day_{}'.format(deadline_day))] = 1
        df.iloc[0, df.columns.get_loc('launch_month_{}'.format(launch_month))] = 1
        df.iloc[0, df.columns.get_loc(
            'deadline_month_{}'.format(deadline_month))] = 1
        df.iloc[0, df.columns.get_loc('launch_time_{}'.format(launch_time))] = 1
        df.iloc[0, df.columns.get_loc(
            'deadline_time_{}'.format(deadline_time))] = 1
        df.iloc[0, df.columns.get_loc('locations_{}'.format(location))] = 1
    except Exception as e:
        return dcc.Markdown('All fields must be filled!')
    if df.isnull().values.any() or len(blurb) < 1:
        return dcc.Markdown('All fields must be filled!')
    else:
        # text pre-processing
        REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
        BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
        NEWLINE_RE = re.compile('\n')
        SPACES_RE = re.compile(' +')

        def clean_text(text):
            """
                text: a string

                return: modified initial string
            """
            text = str(text).lower()
            text = REPLACE_BY_SPACE_RE.sub(' ', text)
            text = NEWLINE_RE.sub(' ', text)
            text = BAD_SYMBOLS_RE.sub('', text)
            text = SPACES_RE.sub(' ', text)
            # remove stopwords from text
            text = ' '.join(word for word in text.split()
                            if word not in STOPWORDS)
            return text
        blurb = clean_text(blurb)
        # tokenize blurbs
        max_seq_length = 200
        tokenizer = joblib.load('assets/tokenizer.pickle')
        scaler = joblib.load('assets/scaler.pickle')
        blurb = tokenizer.texts_to_sequences([blurb])
        blurb = tf.keras.preprocessing.sequence.pad_sequences(
            blurb, maxlen=max_seq_length)
        df = scaler.transform(df)
        model = KickstarterModel(max_words, embedding_dim)
        model.compile(optimizer=tf.keras.optimizers.Nadam(),
                      loss='binary_crossentropy')
        model.load_weights('assets/kickstarter_model/')
        prediction = model.predict([blurb, df])
        del(model)
        del(tokenizer)
        del(scaler)
        if prediction[0][0] > 0.5:
            return dcc.Markdown('{}% Sure your project will be successful!'.format(round(prediction[0][0]*100, 2)))
        else:
            return dcc.Markdown('{}% Sure your project will not be successful.'.format(round(100-prediction[0][0]*100, 2)))


def get_options(options):
    """
    Get options for dcc.Dropdown from a list of options
    """
    opts = []
    for opt in options:
        opts.append({'label': opt, 'value': opt})
    return opts


def get_options_title_case(options):
    """
    Get options for dcc.Dropdown from a list of options
    """
    opts = []
    for opt in options:
        opts.append({'label': opt.title(), 'value': opt})
    return opts


# Column 1
column1 = dbc.Col(
    [
        dcc.Markdown("""
                     ## Kickstarter Success Prediction Model

                     The model has a precision of 0.86, a recall of 0.72, and an f1-score of 0.78 using 14594 test set samples for the positive class.

                     The precision means that when the model guesses a project will succeed, it is right roughly 86% of the time.

                     The recall means that it misses around 28% of sucessful projects. So if the model predicts your project will

                     succeed, there is a high likely hood that it will. But if the model predicts it won't succeed, your project

                     could still be in the 28% misclassified by the model.

                     The model also has a precision of 0.75, a recall of 0.88, and an f1-score of 0.81 using 13997 test set samples for the negative class.
                     
                     The macro and weighted average precision is 0.81 for a total of 28591 test set samples.

                     The macro and weighted average recall is 0.80 for a total of 28591 test set samples.

                     The macro and weighted average f1-score is 0.80 for a total of 28591 test set samples.

                     ##### After hitting 'Make Prediction' it can take a minute or two
                     """),
        html.Br(), html.Br(),
        dcc.Input(id='name', type='text',
                  placeholder='Name'),
        html.Br(), html.Br(),
        dcc.Input(id='blurb', type='text',
                  placeholder='Blurb'),
        html.Br(), html.Br(),
    ]
)

# Column 2
column2 = dbc.Col(
    [
        dcc.Input(id='usd_goal', type='number',
                  placeholder='USD Goal', min=1),
        html.Br(), html.Br(),
        dcc.Input(id='creation_to_launch_days', type='number',
                  placeholder='Creation to Launch Days', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Input(id='campaign_days', type='number',
                  placeholder='Campaign Days', min=1, step=1),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='category',
            options=get_options_title_case(CATEGORY_NAMES),
            searchable=True,
            clearable=True,
            placeholder='Category'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='sub_category',
            options=get_options_title_case(SUB_CATEGORY_NAMES),
            searchable=True,
            clearable=True,
            placeholder='Sub-Category'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='country',
            options=get_options(COUNTRY_NAMES),
            searchable=True,
            clearable=True,
            placeholder='Country'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='location',
            options=get_options(LOCATION_NAMES),
            searchable=True,
            clearable=True,
            placeholder='Location'
        ),
        html.Br(), html.Br(),

    ]
)

# Column 3
column3 = dbc.Col(
    [
        dcc.Dropdown(
            id='launch_time',
            options=get_options(TIME_PERIODS),
            searchable=True,
            clearable=True,
            placeholder='Launch Time'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='deadline_time',
            options=get_options(TIME_PERIODS),
            searchable=True,
            clearable=True,
            placeholder='Deadline Time'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='launch_day',
            options=get_options(DAYS),
            searchable=True,
            clearable=True,
            placeholder='Launch Day'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='deadline_day',
            options=get_options(DAYS),
            searchable=True,
            clearable=True,
            placeholder='Deadline Day'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='launch_month',
            options=get_options(MONTHS),
            searchable=True,
            clearable=True,
            placeholder='Launch Month'
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(
            id='deadline_month',
            options=get_options(MONTHS),
            searchable=True,
            clearable=True,
            placeholder='Deadline Month'
        ),
        html.Br(), html.Br(),
    ]
)

# Column for displaying the app callback result after clicking
# on the prediction button
prediction_column = dbc.Col(
    html.Center(id='output-submit')
)

# Column for price prediction button
column_button = dbc.Col(
    [
        html.Hr(),
        html.Center((dbc.Button('Make Prediction', color='primary',
                                id='btn-submit', n_clicks=0)))
    ]
)

# Webpage layout
layout = dbc.Container(
    fluid=True,
    children=[
        dbc.Row([column1]),
        dbc.Row([column2, column3]),
        dbc.Row([prediction_column]),
        dbc.Row([column_button])
    ],
    style={'margin': 'auto'}
)


# App callback to get values from user and return a prediction
@ app.callback(
    Output('output-submit', 'children'),
    Input('btn-submit', 'n_clicks'),
    State('name', 'value'),
    State('blurb', 'value'),
    State('usd_goal', 'value'),
    State('creation_to_launch_days', 'value'),
    State('campaign_days', 'value'),
    State('category', 'value'),
    State('country', 'value'),
    State('sub_category', 'value'),
    State('launch_day', 'value'),
    State('deadline_day', 'value'),
    State('launch_month', 'value'),
    State('deadline_month', 'value'),
    State('launch_time', 'value'),
    State('deadline_time', 'value'),
    State('location', 'value')
)
def update_output(clicks, name, blurb, usd_goal, creation_to_launch_days,
                  campaign_days, category, country, sub_category,
                  launch_day, deadline_day, launch_month, deadline_month,
                  launch_time, deadline_time, location):
    if clicks:
        # Return prediction when button is clicked
        return get_prediction(name, blurb, usd_goal, creation_to_launch_days,
                              campaign_days, category, country, sub_category,
                              launch_day, deadline_day, launch_month, deadline_month,
                              launch_time, deadline_time, location)
