from dash import Dash, html, dcc, dash_table
import dash_bootstrap_components as dbc

from dash.dependencies import Input, Output, State
import plotly.express as px
import numpy as np
import pandas as pd

from dash.exceptions import PreventUpdate
from dash_extensions.enrich import Output, DashProxy, Input, MultiplexerTransform, html
import plotly.graph_objects as go

import joblib
from assets.AE import AE
import torch
from sklearn.metrics import mean_squared_error
from scipy import signal

class AE(torch.nn.Module):
  def __init__(self):
    super(AE, self).__init__()
    self.encoder = torch.nn.Sequential(
            torch.nn.Linear(2048, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 256),
            torch.nn.Tanh(),
        )
    self.decoder = torch.nn.Sequential(
            torch.nn.Linear(256, 512),
            torch.nn.Tanh(),
            torch.nn.Linear(512, 1024),
            torch.nn.Tanh(),
            torch.nn.Linear(1024, 2048),
            torch.nn.Tanh(),
        )
  def forward(self, x):
          encoded = self.encoder(x)
          decoded = self.decoder(encoded)
          return decoded

global f
f = np.linspace(0, 1000, 2048)

global lrp_FE
global lrp_DE

def softmax_calculator(arr):
  temp_list = []

  for value in arr:
    temp_list.append(np.exp(value))

  summation = np.sum(np.array(temp_list))

  normalized_output_diction = np.round(temp_list / summation, 4)

  return normalized_output_diction

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
# app = Dash(__name__, external_stylesheets=external_stylesheets)
app = DashProxy(transforms=[MultiplexerTransform()], external_stylesheets=external_stylesheets)
server = app.server

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

importing_path = r'assets/'

sig_df = pd.read_csv(importing_path + 'dual_subsampled_df.csv', index_col=0)
lrp_df = pd.read_csv(importing_path + 'LRP_dual_subsampled_df.csv', index_col=0)
DE_scaler = joblib.load(open(importing_path + 'DE_scaler.pkl', 'rb'))
DE_AE = torch.load(importing_path + 'DE_ae.pt', map_location=torch.device('cpu'))
FE_scaler = joblib.load(open(importing_path + 'FE_scaler.pkl', 'rb'))
FE_AE = torch.load(importing_path + 'FE_ae.pt', map_location=torch.device('cpu'))

rpm_state_ratio_dict = {
    'Fan-End': {'1730':
            {
                'Inner-Race': 142.63,
                'Outer-Race': 88.02,
                'Ball': 114.97
            },
            '1750':
            {
                'Inner-Race': 144.28,
                'Outer-Race': 89.04,
                'Ball': 116.29
            },
            '1772':
            {
                'Inner-Race': 146.09,
                'Outer-Race': 90.16,
                'Ball': 117.76
            },
            '1797':
            {
                'Inner-Race': 148.15,
                'Outer-Race': 91.43,
                'Ball': 119.42
            }
            },
    'Drive-End': {'1730':
            {
                'Inner-Race': 156.14,
                'Outer-Race': 103.36,
                'Ball': 135.91
            },
            '1750':
            {
                'Inner-Race': 157.94,
                'Outer-Race': 104.56,
                'Ball': 137.48
            },
            '1772':
            {
                'Inner-Race': 159.93,
                'Outer-Race': 105.87,
                'Ball': 139.21
            },
            '1797':
            {
                'Inner-Race': 162.19,
                'Outer-Race': 107.36,
                'Ball': 141.17
            }
}
}

fault_freq_ratios = {
    'Fan-End': {
            'Inner-Race': 4.9469,
            'Ball': 3.9874,
            'Outer-Race': 3.053,
},
    'Drive-End': {
        'Inner-Race': 5.4152,
        'Ball': 4.7135,
        'Outer-Race': 3.5848,
}
}


def annotation_maker(f_axis, fault_freq_ratios, bearing, load, state, harmonics, harmonic_severity_ratio):

  annotation = np.zeros(f_axis.shape)
  for i in range(1, harmonics+1):

    freq_range = [int(load)/60 * i * fault_freq_ratios[bearing][state] - 10, int(load)/60 * i * fault_freq_ratios[bearing][state] + 10]
    freq_range_indices = np.where(np.logical_and(f_axis >= freq_range[0], f_axis <= freq_range[1]))[0]
    annotation[freq_range_indices] = harmonic_severity_ratio[i - 1] * 1

  return annotation

app.layout = html.Div(children = [
    html.Div(
        children = [
            html.Div(children = [

                dcc.Graph(id='FE_signal_lrp_graph',),
                html.H5(id = 'FE_health_index', style={'text-align':'center'}),
                html.Br(),
            ],className="six columns"),
        
            html.Div(children = [
                dcc.Graph(id='DE_signal_lrp_graph',),
                html.H5(id = 'DE_health_index', style={'text-align':'center'}),
                html.Br(),
            ],className="six columns"),
    
    
    ], className = "row"),

    html.Div(children = [

        # html.Div(children = [], className="one columns"),

        html.Div(
                children = [
                    
                    html.H4('Select Signal Properties'),
                    
                    html.Br(),

                    html.Div(children=[
                    html.Label('Rotational Speed (RPM)'),
                    dcc.Dropdown(id = 'rpm', options = ['1730', '1750', '1772', '1797']),
                    ]),

                    html.Br(),

                    html.Div(children=[
                    html.Label('Fan-End/Drive-End Health State'),
                    dcc.Dropdown(id = 'state', options = [
                        'Normal-Normal',
                        'Normal-InnerRace',
                        'Normal-OuterRace',
                        'Normal-Ball',
                        'InnerRace-Normal',
                        'OuterRace-Normal',
                        'Ball-Normal'
                    ]),
                    ]),

                    html.Br(),

                    html.Div(children=[
                        html.Label('Repeatation'),
                        dcc.Dropdown(id = 'rep', options = [str(rep) for rep in range(1,11)]),
                        ]),

                    html.Br(),

                    html.Div(children=[
                        html.Button(id='submit-button-state', n_clicks=0, children='Submit'),
                    ]),

                ], className="three columns", style={'text-align':'center'}
            ),

        # html.Div(children = [], className="one columns"),

        html.Div(children = [
            
            html.H4('Annotation Settings'),

            html.Br(),

            html.Label('Annotation Bearing'),
            dcc.Dropdown(id = 'annotation_bearing', options = ['Fan-End', 'Drive-End']),
            
            html.Br(),

            html.Div(children=[
            html.Label('Annotation Rotational Speed (RPM)', style={'text-align':'center'}),
            dcc.Dropdown(id = 'annotation_rpm', options = ['1730', '1750', '1772', '1797']),
            ]),

            html.Br(),

            html.Div(children=[
            html.Label('Annotation Health State'),
            dcc.Dropdown(id = 'annotation_state', options = ['Inner-Race', 'Outer-Race', 'Ball']),
            ]),

            html.Br(),

            html.Div(children=[
                html.Label('Fault Frequency', style={'text-align':'center'}),
                html.Div(id = 'fault_freq', style={'text-align':'center'}),
            ]),

        ], className="three columns", style={'text-align':'center'}),

        # html.Div(children = [], className="one columns"),

        html.Div(children = [
            html.H4('Normalized Similarity Scores', style = {'text-align':'center'}),

            html.Div(
                
                html.Table(children = [
                    html.Tr(children = [html.Th('Fault'), html.Th('FE Signal'), html.Th('DE Signal')], style = {'text-align':'center'}),
                    html.Tr(children = [html.Th('FE-IR'), html.Td(html.Div(id='FE_fe_IR_NSS')), html.Td(html.Div(id='DE_fe_IR_NSS'))]),
                    html.Tr(children = [html.Th('FE-OR'), html.Td(html.Div(id='FE_fe_OR_NSS')), html.Td(html.Div(id='DE_fe_OR_NSS'))]),
                    html.Tr(children = [html.Th('FE-Ball'), html.Td(html.Div(id='FE_fe_Ball_NSS')), html.Td(html.Div(id='DE_fe_Ball_NSS'))]),
                    html.Tr(children = [html.Th('DE-IR'), html.Td(html.Div(id='FE_de_IR_NSS')), html.Td(html.Div(id='DE_de_IR_NSS'))]),
                    html.Tr(children = [html.Th('DE-OR'), html.Td(html.Div(id='FE_de_OR_NSS')), html.Td(html.Div(id='DE_de_OR_NSS'))]),
                    html.Tr(children = [html.Th('DE-Ball'), html.Td(html.Div(id='FE_de_Ball_NSS')), html.Td(html.Div(id='DE_de_Ball_NSS'))])]
                , style = {'text-align':'center', 'margin-left': 'auto', 'margin-right': 'auto'})
            , style = {'text-align':'center'})
        ], className = "three columns", style = {'text-align':'center'}),

        html.Div(children = [
            html.H4('Bearings Specifications'),
            
            html.Br(),

            html.Div('Drive-End Bearing'),
            html.Div(
                html.A(
                    children = html.Img(src = r'assets/qr-code_DE.png', alt='image', style={'height':'25%', 'width':'25%', }),
                    href = 'https://www.skf.com/group/products/rolling-bearings/ball-bearings/deep-groove-ball-bearings/productid-6205',
                    target="_blank",
                ),
                style = {'text-align':'center'}
            ),

            html.Br(),

            html.Div('Fan-End Bearing'),
            html.Div(
                html.A(
                    children = html.Img(src = r'assets/qr-code_FE.png', alt='image', style={'height':'25%', 'width':'25%', }),
                    href = 'https://www.skf.com/id/productinfo/productid-6203%202RSJEM',
                    target="_blank",
                ),
                style = {'text-align':'center'}
            ),

        ], className = "three columns", style = {'text-align':'center'})
        
    ], className = "row"),

    ])

@app.callback(
    Output(component_id='FE_health_index', component_property='children'),
    Output(component_id='FE_health_index', component_property='style'),
    Output(component_id='DE_health_index', component_property='children'),
    Output(component_id='DE_health_index', component_property='style'),
    Output(component_id='FE_signal_lrp_graph', component_property='figure'),
    Output(component_id='DE_signal_lrp_graph', component_property='figure'),
    Input('submit-button-state', 'n_clicks'),
    State(component_id='rpm', component_property='value'),
    State(component_id='state', component_property='value'),
    State(component_id='rep', component_property='value'),
)
def select_draw_signal(n_clicks,rpm,state,rep):

    if rpm and state and rep:
        selected_sig = sig_df.loc[(sig_df['state'] == state) & (sig_df['load'] == int(rpm))].iloc[int(rep), :]
        selected_lrp = lrp_df.loc[(lrp_df['state'] == state) & (lrp_df['load'] == int(rpm))].iloc[int(rep), :]

        global x_FE
        global x_DE
        x_FE = selected_sig[:2048].to_numpy()
        x_DE = selected_sig[2048:4096].to_numpy()
        
        global lrp_FE
        global lrp_DE
        lrp_FE = selected_lrp[:2048].to_numpy()
        lrp_DE = selected_lrp[2048:4096].to_numpy()

        x_FE_scaled = FE_scaler.transform(x_FE.reshape(1,-1))
        x_DE_scaled = DE_scaler.transform(x_DE.reshape(1,-1))

        x_FE_scaled = torch.autograd.Variable(torch.Tensor(x_FE_scaled).float())
        x_DE_scaled = torch.autograd.Variable(torch.Tensor(x_DE_scaled).float())
        x_FE_recons =  FE_AE(x_FE_scaled).cpu().detach().numpy()
        x_DE_recons =  DE_AE(x_DE_scaled).cpu().detach().numpy()

        mse_FE = np.round(mean_squared_error(x_FE_scaled, x_FE_recons), 4)
        mse_DE = np.round(mean_squared_error(x_DE_scaled, x_DE_recons), 4)

        FE_health_index = 'FE HSI: ' + str(mse_FE)
        if mse_FE < 0.5:
            color = 'green'
        else:
            color = 'red'

        FE_style = {
            'color': color,
            'text-align':'center'
        }

        DE_health_index = 'DE HSI: ' + str(mse_DE)
        if mse_DE < 0.5:
            color = 'green'
        else:
            color = 'red'

        DE_style = {
            'color': color,
            'text-align':'center'
        }

        FE_fig_signal = px.line(x = list(f), y = [list(x_FE), list(lrp_FE)],)
        series_name = ['Signal', 'LRP']

        for idx, name in enumerate(series_name):
            FE_fig_signal.data[idx].name = name
        
        FE_fig_signal.update_layout(title='Fan-End Original Signal & LRP',
                                xaxis_title='Hz',
                                yaxis_title='Amplitude',
                                title_x=0.5,
                                legend_title_text='Variable')

        DE_fig_signal = px.line(x = list(f), y = [list(x_DE), list(lrp_DE)],)
        series_name = ['Signal', 'LRP']

        for idx, name in enumerate(series_name):
            DE_fig_signal.data[idx].name = name
        
        DE_fig_signal.update_layout(title='Drive-End Original Signal & LRP',
                                xaxis_title='Hz',
                                yaxis_title='Amplitude',
                                title_x=0.5,
                                legend_title_text='Variable')


    return FE_health_index, FE_style, DE_health_index, DE_style, FE_fig_signal, DE_fig_signal

@app.callback(
    Output(component_id='FE_signal_lrp_graph', component_property='figure'),
    Output(component_id='DE_signal_lrp_graph', component_property='figure'),
    Output(component_id='fault_freq', component_property='children'),
    State(component_id='rpm', component_property='value'),
    State(component_id='state', component_property='value'),
    State(component_id='rep', component_property='value'),
    Input(component_id='annotation_bearing', component_property='value'),
    Input(component_id='annotation_rpm', component_property='value'),
    Input(component_id='annotation_state', component_property='value'),
    
)
def FE_annotation(rpm, state, rep, FE_annotation_bearing, FE_annotation_rpm, FE_annotation_state,):
    
    if (FE_annotation_rpm and FE_annotation_state and FE_annotation_bearing) and (rpm and state and rep):
        annotation_freq = rpm_state_ratio_dict[FE_annotation_bearing][FE_annotation_rpm][FE_annotation_state]
        harmonics = 3

        FE_fig_signal = px.line(x = list(f), y = [list(x_FE), list(lrp_FE)],)
        series_name = ['Signal', 'LRP']
        for idx, name in enumerate(series_name):
            FE_fig_signal.data[idx].name = name
        FE_fig_signal.update_layout(title='Fan-End Original Signal & LRP',
                                xaxis_title='Hz',
                                yaxis_title='Amplitude',
                                title_x=0.5,
                                legend_title_text='Variable')

        DE_fig_signal = px.line(x = list(f), y = [list(x_DE), list(lrp_DE)],)
        series_name = ['Signal', 'LRP']

        for idx, name in enumerate(series_name):
            DE_fig_signal.data[idx].name = name
        
        DE_fig_signal.update_layout(title='Drive-End Original Signal & LRP',
                                xaxis_title='Hz',
                                yaxis_title='Amplitude',
                                title_x=0.5,
                                legend_title_text='Variable')                        
        
        for i in range(1, harmonics + 1):
            FE_fig_signal.add_vline(x = i*annotation_freq, line_dash = 'dash', annotation_text = str(i) + 'X' + FE_annotation_state)
        
        for i in range(1, harmonics + 1):
            DE_fig_signal.add_vline(x = i*annotation_freq, line_dash = 'dash', annotation_text = str(i) + 'X' + FE_annotation_state)

        return FE_fig_signal, DE_fig_signal, str(annotation_freq) + ' Hz'
    
    elif (rpm and state and rep) and not (FE_annotation_rpm and FE_annotation_state and FE_annotation_bearing):
        FE_fig_signal = px.line(x = list(f), y = [list(x_FE), list(lrp_FE)],)
        series_name = ['Signal', 'LRP']
        for idx, name in enumerate(series_name):
            FE_fig_signal.data[idx].name = name
        FE_fig_signal.update_layout(title='Fan-End Original Signal & LRP',
                                xaxis_title='Hz',
                                yaxis_title='Amplitude',
                                title_x=0.5,
                                legend_title_text='Variable')

        DE_fig_signal = px.line(x = list(f), y = [list(x_DE), list(lrp_DE)],)
        series_name = ['Signal', 'LRP']
        for idx, name in enumerate(series_name):
            DE_fig_signal.data[idx].name = name
        DE_fig_signal.update_layout(title='Drive-End Original Signal & LRP',
                                xaxis_title='Hz',
                                yaxis_title='Amplitude',
                                title_x=0.5,
                                legend_title_text='Variable')

        return FE_fig_signal, DE_fig_signal, ''

    else:
        return {}, {}, ''

#naming convention: signal + _ + fault eg.: FE_de_IR -> normalized similarity score of IR @ DE bearing, according to the FE signal

@app.callback(
    Output(component_id='FE_fe_IR_NSS', component_property='children'),
    Output(component_id='FE_fe_OR_NSS', component_property='children'),
    Output(component_id='FE_fe_Ball_NSS', component_property='children'),
    Output(component_id='FE_de_IR_NSS', component_property='children'),
    Output(component_id='FE_de_OR_NSS', component_property='children'),
    Output(component_id='FE_de_Ball_NSS', component_property='children'),
    Output(component_id='DE_fe_IR_NSS', component_property='children'),
    Output(component_id='DE_fe_OR_NSS', component_property='children'),
    Output(component_id='DE_fe_Ball_NSS', component_property='children'),
    Output(component_id='DE_de_IR_NSS', component_property='children'),
    Output(component_id='DE_de_OR_NSS', component_property='children'),
    Output(component_id='DE_de_Ball_NSS', component_property='children'),
    Input(component_id='annotation_rpm', component_property='value'),
)
def nomralized_similarity_score_updater(annotation_rpm):

    if annotation_rpm:

        harmonics = 3
        harmonic_severity_ratio = [1, 0.66, 0.33]

        annotation_matrix = np.array([
            annotation_maker(f, fault_freq_ratios, 'Fan-End', annotation_rpm, 'Inner-Race', harmonics, harmonic_severity_ratio),
            annotation_maker(f, fault_freq_ratios, 'Fan-End', annotation_rpm, 'Outer-Race', harmonics, harmonic_severity_ratio),
            annotation_maker(f, fault_freq_ratios, 'Fan-End', annotation_rpm, 'Ball', harmonics, harmonic_severity_ratio),
            annotation_maker(f, fault_freq_ratios, 'Drive-End', annotation_rpm, 'Inner-Race', harmonics, harmonic_severity_ratio),
            annotation_maker(f, fault_freq_ratios, 'Drive-End', annotation_rpm, 'Outer-Race', harmonics, harmonic_severity_ratio),
            annotation_maker(f, fault_freq_ratios, 'Drive-End', annotation_rpm, 'Ball', harmonics, harmonic_severity_ratio)
        ])

        fe_similarity_scores = np.matmul(x_FE, annotation_matrix.T)
        de_similarity_scores = np.matmul(x_DE, annotation_matrix.T)

        fe_normalized_similarity_scores = softmax_calculator(fe_similarity_scores)
        de_normalized_similarity_scores = softmax_calculator(de_similarity_scores)
        
        # # FE Similarity Scores

        # fe_similarity_annotation_vectors = {}
        # fe_similarity_scores = {}

        # for state in list(fault_freq_ratios['Fan-End'].keys()):
        #     ratio = fault_freq_ratios['Fan-End'][state]
        #     temp_annotation = np.zeros(f.shape)
        #     for i in range(1, harmonics+1):
        #         freq_range = [i * int(annotation_rpm)/60 * ratio - 10, i * int(annotation_rpm)/60 * ratio + 10]
        #         temp_annotation[np.where(np.logical_and(f >= freq_range[0], f <= freq_range[1]))] = harmonic_severity_ratio[i - 1]
        #         fe_similarity_annotation_vectors[state] = temp_annotation

        #     fe_similarity_scores[state] = np.inner(x_FE, fe_similarity_annotation_vectors[state])


        # fe_normalized_similarity_scores = softmax_calculator(fe_similarity_scores)

        # # DE Similarity Scores

        # de_similarity_annotation_vectors = {}
        # de_similarity_scores = {}

        # for state in list(fault_freq_ratios['Drive-End'].keys()):
        #     ratio = fault_freq_ratios['Drive-End'][state]
        #     temp_annotation = np.zeros(f.shape)
        #     for i in range(1, harmonics+1):
        #         freq_range = [i * int(annotation_rpm)/60 * ratio - 10, i * int(annotation_rpm)/60 * ratio + 10]
        #         temp_annotation[np.where(np.logical_and(f >= freq_range[0], f <= freq_range[1]))] = harmonic_severity_ratio[i - 1]
        #         de_similarity_annotation_vectors[state] = temp_annotation

        #     de_similarity_scores[state] = np.inner(x_DE, de_similarity_annotation_vectors[state])


        # de_normalized_similarity_scores = softmax_calculator(de_similarity_scores)


        # return fe_normalized_similarity_scores['Inner-Race'], fe_normalized_similarity_scores['Outer-Race'], fe_normalized_similarity_scores['Ball'], de_normalized_similarity_scores['Inner-Race'], de_normalized_similarity_scores['Outer-Race'], de_normalized_similarity_scores['Ball']

        return fe_normalized_similarity_scores[0], fe_normalized_similarity_scores[1], fe_normalized_similarity_scores[2], fe_normalized_similarity_scores[3], fe_normalized_similarity_scores[4], fe_normalized_similarity_scores[5], de_normalized_similarity_scores[0], de_normalized_similarity_scores[1], de_normalized_similarity_scores[2], de_normalized_similarity_scores[3], de_normalized_similarity_scores[4], de_normalized_similarity_scores[5]

    else:
        return '', '', '', '', '', '', '', '', '', '', '', ''


if __name__ == '__main__':
    app.run_server(debug=True)