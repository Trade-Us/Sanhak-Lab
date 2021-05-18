import dash_html_components as html
import dash_core_components as dcc

# from result_graph import num_clus/ter
def detailGraphOption(num_cluster):
    return [
        html.Label("Choose cluster"),
        dcc.RadioItems(id="nth-cluster", 
        options=[
                    {'label': str(i+1), 'value': i}
                    for i in range(num_cluster)
                ], value=0),
        html.Label("Choose Type"),
        dcc.Dropdown(id="detail-graph-input", 
        options= [
                    {'label': 'Graph Detail', 'value': 'GrDt'},
                    {'label': 'Graph Big', 'value': 'GrBg'}
                ], value='GrDt'),
        html.Label("Number of data graphs per clusters", id="label-n-graphs"),
        dcc.Input(id='num-of-graphs', min=1, value=1, type='number'),
    ]