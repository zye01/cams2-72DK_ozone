import pandas as pd
import numpy as np
import streamlit as st
from datetime import date
import plotly.express as px
import plotly.graph_objects as go


def run():
    st.set_page_config(layout="wide")
    st.markdown("# CAMS2_72DK Ozone Alert")

    # Initiate the state
    state = st.session_state
    initiate_state(state)

    # Sidebar configuration
    sidebar_configuration()

    # Load data
    load_data(state)

    tab1, tab2 = st.tabs(['Time series','Target'])

    with tab1:
        # Divide the page into two columns
        cm0, cm1 = st.columns([2,5])

        # Plot the stations map
        plot_station_map(state,cm0)

        # select period and stations
        make_selections(state,cm1)

        # plot scatters
        plot_scatters(state,cm0)

        # Show metrics
        show_metrics(state,cm0)

        # Plot the time series
        download_csv(state,cm1)
        plot_time_series(state,cm1)

        # Show the count of the limit
        show_lim_count(state,cm1)
    
    with tab2:
        st.markdown('This page shows the target plots and summary report for the entire period.')
        st.markdown("## Target Plots")
        # Divide the page into four columns
        cms = st.columns([1,1,1,1])

        for i,icase in enumerate(state.clist[1:]):
            cms[i].markdown(f"### {icase}")
            cms[i].image(f'data/TargetPlot_For2_{icase}_O3.png',use_column_width=True)

        st.markdown("## Summary report for the 100 $\mu g/m^3$ alert")
        cm0, cm1 = st.columns([1,1])
        cm0.markdown("### Regional stations")
        cm0.image('data/SummaryReport_regional.png',use_column_width=True)

        cm1.markdown("### Urban stations")
        cm1.image('data/SummaryReport_urban.png',use_column_width=True)

            

def make_selections(state,cm):
    # Select the time period
    cm1, cm2  = cm.columns([1,1])
    state.st = cm1.date_input('Start date', state.first_date)
    state.ed = cm2.date_input('End date', state.last_date)

    # Select the stations
    select_station(state,cm)

    # Get the selected data
    get_selected_data(state)

def plot_scatters(state,cm):
    # Plot the scatter plots
    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )
    for icase in state.clist[1:]:
        fig.add_trace(go.Scatter(
                x=state.sel_df['obs'],
                y=state.sel_df[icase],
                name=icase,
                mode='markers',
            ))
    # Add 1-1 line
    fig.add_trace(go.Scatter(
        x=[10,150],
        y=[10,150],
        mode='lines',
        line=dict(color='black',dash='dash'),
        name='1:1 line'
    ))
    fig.update_layout(xaxis_range=[10,150],yaxis_range=[10,150])

    fig.update_layout(
                    xaxis_title='Observed O\u2083 (\u03BCg/m^3)',
                    yaxis_title='Simulated O\u2083 (\u03BCg/m^3)',
                    margin=dict(l=2, r=2, t=2, b=2),
                    #    width=1000,
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="left",
                        x=0.01
                        ),
                    font=dict(size=18),
                    xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey', showgrid=False,\
                        ticks='inside',title_font=dict(size=15),tickfont=dict(size=12)),
                    yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey', showgrid=False,\
                        ticks='inside',title_font=dict(size=15),tickfont=dict(size=12)),
                   )

    cm.plotly_chart(fig, use_container_width=True)

def plot_time_series(state,cm):
    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )
    for icase in state.clist:
        if icase=='obs':
            ldict = {'color':'black','width':2}
        else:
            ldict = {'width':1.5}
        fig.add_trace(go.Scatter(
                x=state.sel_df['time'],
                y=state.sel_df[icase],
                name=icase,
                line=ldict,
            ))
            

    fig.add_hline(y=120, line_color='gray',line_dash="dash")
    fig.add_hline(y=100, line_color='grey',line_dash="dot")
    fig.update_layout(yaxis_range=[10,150])
    
    # Count and plot the limit
    plot_limits(state,fig)

    fig.update_layout(
                    xaxis_title='Date',
                    yaxis_title='Daily 8h-maximum O\u2083 (\u03BCg/m^3)',
                    margin=dict(l=2, r=2, t=2, b=2),
                    #    width=1000,
                    height=500,
                    legend=dict(
                        orientation="h",
                        yanchor="top",
                        y=-0.2,
                        xanchor="left",
                        x=0.01,
                        font = dict(size=15)
                        ),
                    font=dict(size=18),
                    xaxis=dict(showline=True, linewidth=1, linecolor='lightgrey', showgrid=False,\
                        ticks='inside',title_font=dict(size=20),tickfont=dict(size=15)),
                    yaxis=dict(showline=True, linewidth=1, linecolor='lightgrey', showgrid=False,\
                        ticks='inside',title_font=dict(size=20),tickfont=dict(size=15)),
                   )

    cm.plotly_chart(fig, use_container_width=True)
    
def plot_limits(state,fig):
    state.lim = {'Alert_100': {'lower':100,'upper':1000,'color':'cyan','num':0,'js':[]},
           'Alert_120': {'lower':120,'upper':1000,'color':'deeppink','num':0,'js':[]}}
    # included = 0
    for ilim,ilimp in state.lim.items():
        ilimp['catches'] = {icase:0 for icase in state.clist[1:]}
        ilimp['falses'] = {icase:0 for icase in state.clist[1:]}
    
    for i, d in enumerate(fig.data):
        if d.name=='obs':
            for j, y in enumerate(d.y):
                for ilim,ilimp in state.lim.items():
                    if y > ilimp['lower'] and y < ilimp['upper']:
                        fig.add_traces(go.Scatter(x=[fig.data[i]['x'][j]],
                                                y=[fig.data[i]['y'][j]],
                                                mode = 'markers',
                                                marker = dict(color=ilimp['color']),
                                                name = f'Over {ilim}',
                                                legendgroup = f'Over {ilim}',
                                                showlegend = False if ilimp['num'] >0 else True
                                                ))
                        ilimp['num'] = ilimp['num'] + 1
                        ilimp['js'].append(j)
        else:
            for j, y in enumerate(d.y):
                for ilim,ilimp in state.lim.items():
                    if y > ilimp['lower'] and y < ilimp['upper']:
                        if j in ilimp['js']:
                            ilimp['catches'][d.name] = ilimp['catches'][d.name] + 1
                        else:
                            ilimp['falses'][d.name] = ilimp['falses'][d.name] + 1



def show_lim_count(state,cm):
    # get limit dataframe
    cm1, cm2 = cm.columns([1,1])
    cm1.markdown("### Alert 100 $\u03BCg/m^3$")
    alert = 'Alert_100'
    cm1.write(f"Observed {state.lim[alert]['num']} alerts")
    lim_df = get_lim_df(state,alert)
    plot_lim_counts(lim_df,cm1)

    cm2.markdown("### Alert 120 $\u03BCg/m^3$")
    alert = 'Alert_120'
    cm2.write(f"Observed {state.lim[alert]['num']} alerts")
    lim_df = get_lim_df(state,alert)
    plot_lim_counts(lim_df,cm2)



def plot_lim_counts(lim_df,cm):
    # Plot the stacked bar chart using plotly
    fig = go.Figure()
    fig.update_layout(
        template = 'plotly_white'
    )
    colors = ['blue','grey','red']
    for icase in lim_df.columns[1:]:
        fig.add_trace(go.Bar(
            x=lim_df['Simulation'],
            y=lim_df[icase],
            marker_color=colors.pop(0),
            name=icase,
            text=lim_df[icase],
        ))

    maximum = lim_df[['Catch','Missing','False']].sum(axis=1).max()
    fig.update_layout(barmode='stack')
    fig.update_layout(
        # xaxis_title='Simulation',
        yaxis_title='Count',
        yaxis_range=[0,maximum+1],
        margin=dict(l=2, r=2, t=2, b=2),
        width=400,
        height=300,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.1,
            xanchor="left",
            x=0.01,
            # font 
            font = dict(size=15)
        ),
        font=dict(size=18),
        xaxis=dict(showline=False, showgrid=False,\
                        ticks='inside',tickfont=dict(size=12)),
        yaxis=dict(showline=False, showgrid=False,\
                        ticks='inside',title_font=dict(size=15),tickfont=dict(size=12)),
    )
    cm.plotly_chart(fig)





def get_lim_df(state,alert):
    observed = state.lim[alert]['num']
    lim_dic = {'Simulation':state.clist[1:],'Catch':[],'Missing':[],'False':[]}
    for sim in state.clist[1:]:
        lim_dic['Catch'].append(state.lim[alert]['catches'][sim])
        lim_dic['Missing'].append(observed - state.lim[alert]['catches'][sim])
        lim_dic['False'].append(state.lim[alert]['falses'][sim])
    lim_df = pd.DataFrame.from_dict(lim_dic)
    return lim_df

def download_csv(state,cm):
    csv = state.adf.to_csv(index=False).encode('utf-8')
    # strst, stred = state.st.strftime('%Y%m%d'), state.ed.strftime('%Y%m%d')
    cm.download_button(
        label="Download all data",
        data = csv,
        file_name = f'CAMS2_72DK.csv',
        mime='text/csv',
    )


def show_metrics(state,cm):
    get_metric_df(state)
    # cm.markdown("## Metrics")
    cm.table(state.mtr_df)

def get_metric_df(state):
    sims = state.clist[1:]
    mtr_dic = {'Simulation':sims}
    for imtr in state.mtrs:
        mtr_dic[imtr] = []

    
    for sim in sims:
        seldf = state.sel_df[[sim, 'obs']].dropna().reset_index(drop=True)
        dsim, dobs = seldf[sim].values, seldf['obs'].values
        if len(dsim) > 0 and len(dobs)>0:
            res = calc_metrics(dsim,dobs)
            for imtr in state.mtrs:
                mtr_dic[imtr].append(res[imtr])
        else:
            for imtr in state.mtrs:
                mtr_dic[imtr].append(np.nan)
        

    typedict,formatdict = {},{}
    for imtr in state.mtrs:
        typedict[imtr] = float
        formatdict[imtr] = "{:.2f}"
    outdf = pd.DataFrame.from_dict(mtr_dic)
    outdf.set_index('Simulation',inplace=True)
    outdf.index.name = 'Simulation'
    
    outdf = outdf.astype(typedict)\
        .style.format(formatdict)
    
    state.mtr_df = outdf


def get_selected_data(state):
    # convert the selected date to datetime utc format
    state.st = pd.to_datetime(f'{state.st}T00:00:00Z')
    state.ed = pd.to_datetime(f'{state.ed}T23:59:00Z')
    # print(state.st,state.ed)
    # print(state.adf['time'].unique())
    state.adf['time'] = pd.to_datetime(state.adf['time'])
    state.sel_df = state.adf[(state.adf['time']>=state.st)&(state.adf['time']<=state.ed)]
    # print(state.sel_df['time'].unique())
    state.sel_df = state.sel_df[state.sel_df['Station'].isin(state.sel_station)]
    # group by time
    state.sel_df = state.sel_df[['time']+state.clist].groupby('time').mean().reset_index()
    # print(state.sel_df)


def select_station(state,cm):
    # Select the station
    mstations = ['Mean - all stations', 'Mean - regional stations', 'Mean - urban stations']
    state.sel_1s = cm.selectbox('Station:', mstations+state.stations)
    if state.sel_1s == 'Mean - all stations':
        state.sel_station = state.stations
    elif state.sel_1s == 'Mean - regional stations':
        state.sel_station = state.stations_regional
    elif state.sel_1s == 'Mean - urban stations':
        state.sel_station = state.stations_urban
    else:
        state.sel_station = [state.sel_1s]



def plot_station_map(state,cm):
    px.set_mapbox_access_token(open(".mapbox_token").read())
    fig = px.scatter_mapbox(state.sdf,
                        lat=state.sdf.lat,
                        lon=state.sdf.lon,
                        hover_name="stname",
                        color='type',
                        size='size',
                        size_max=10,
                        zoom=0)
    fig.update_layout(
            autosize=True,
            hovermode='x',
            # showlegend=True,
            # width=250,
            height=500,
            mapbox=dict(
                bearing=0,
                center=dict(
                    lat=56,
                    lon=11
                ),
                pitch=0,
                zoom=5,
                style='light'
            ),
            margin=dict(l=2, r=2, t=2, b=2),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="right",
                x=1.01,
                title=''
            )
        )
    cm.plotly_chart(fig, use_container_width=True)

def sidebar_configuration():
    st.sidebar.markdown("# Notes")
    
    st.sidebar.markdown('Four UBM simulations are available:')
    st.sidebar.markdown('1. eta_thor_dehm: driven by DEHM_THOR simulations and ETA meteorology')
    st.sidebar.markdown('2. ifs_cams_dehm: driven by DEHM_CAMS simulations and IFS meteorology')
    st.sidebar.markdown('3. ifs_cams_ensemble: driven by CAMS ensemble simulations and IFS meteorology')
    st.sidebar.markdown('4. wrf_novana_dehm: driven by DEHM_Novana simulations and WRF meteorology')

    st.sidebar.markdown("O\u2083 daily maximum 8-hour average is used for the comparison")
    st.sidebar.markdown(
        "Measurements are collected from CAMS, unit: $\u03BCg/m^3$", unsafe_allow_html=True)
    



def load_data(state):
    if 'sdf' not in state:
        load_stations(state)

    if 'adf' not in state:
        load_adf(state)


def load_stations(state):
    stfile = 'data/stations.csv'
    state.sdf = pd.read_csv(stfile)
    state.stations = state.sdf['stname'].values.tolist()
    state.stations_regional = state.sdf[state.sdf['type']=='regional']['stname'].values.tolist()
    state.stations_urban = state.sdf[state.sdf['type']=='urban']['stname'].values.tolist()

def load_adf(state):
    infile = 'data/CAMS2_72.csv'
    state.adf = pd.read_csv(infile)
    state.adf['time'] = pd.to_datetime(state.adf['date'])
    state.clist = ['obs']+state.adf['Version'].unique().tolist()
    # print(state.clist)
    state.adf = state.adf.pivot_table(index=['time','Station','obs'],columns='Version',values='mod').reset_index()
    

def calc_metrics(sim,obs):
    outdict = {}

    N = obs.shape[0]

    meansim = np.nanmean(sim)
    meanobs = np.nanmean(obs)
    sigmasim = np.nanstd(sim)
    sigmaobs = np.nanstd(obs)

    diff = sim - obs
    MB = np.nanmean(diff)
    outdict['MB'] = MB

    square_diff = np.square(diff)
    mean_square_diff = np.nanmean(square_diff)
    RMSE = np.sqrt(mean_square_diff)
    outdict['RMSE'] = RMSE

    addition = np.absolute(sim)+np.absolute(obs)
    division = np.where(addition==0, np.nan, np.true_divide(diff, addition))
    NMB = 2*np.nanmean(division)
    outdict['NMB'] = NMB

    FGE = 2*np.nanmean(np.absolute(division))
    outdict['FGE'] = FGE

    
    diffsim = sim - meansim
    diffobs = obs - meanobs
    multidiff = np.multiply(diffsim, diffobs)
    CORR = np.nanmean(multidiff)/(sigmasim*sigmaobs)
    outdict['Corr'] = CORR


    sum_square_diff = np.nansum(square_diff)
    sum_square_obs = np.nansum(np.square(diffobs))
    R2 = 1 - sum_square_diff/sum_square_obs

    # slope, intercept, r_value, p_value, std_err = stats.linregress(sim, obs)
    outdict['R2'] = R2

    # df = pd.DataFrame(outdict)
    return outdict


def initiate_state(state):
    if 'latlim' not in state:
        state.latlim, state.lonlim = [54.1,58.2], [7.1,15.9]
    if 'first_date' not in state:
        state.first_date = date(2023,7,1)
    if 'last_date' not in state:
        state.last_date = date(2024,2,1)
    if 'mtrs' not in state:
        state.mtrs = ['MB', 'NMB', 'RMSE', 'Corr']
    state.autoload = True

if __name__ == '__main__':
    run()
