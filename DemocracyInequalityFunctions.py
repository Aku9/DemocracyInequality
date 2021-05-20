from datetime import date
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm


class Country:
    def __init__(self, fiw_name=None, wid_name=None, wid_code=None):
        self.fiw_name = fiw_name
        self.wid_name = wid_name
        self.wid_code = wid_code


def match_wid_and_fiw_data(wid_df, fiw_df):
    # This can be a function
    country_names_list = []
    c = 0
    for i in range(0, len(fiw_df)):
        for j in range(0, len(wid_df)):
            if fiw_df['Country/Territory'][i] == wid_df['shortname'][j]:
                c = c + 1
                new_country = Country(fiw_df['Country/Territory'][i], wid_df['shortname'][j], wid_df['alpha2'][j])
                country_names_list.append(new_country)
    return country_names_list


def add_countries_with_mismatched_names(country_names_list):
    brunei = Country('Brunei', 'Brunei Darussalam', 'BN')
    country_names_list.append(brunei)

    congo = Country('Congo (Brazzaville)', 'Congo', 'CG')
    country_names_list.append(congo)

    dr_congo = Country('Congo (Kinshasa)', 'DR Congo', 'CD')
    country_names_list.append(dr_congo)

    eswatini = Country('Eswatini', 'Swaziland', 'SZ')
    country_names_list.append(eswatini)

    laos = Country('Laos', 'Loa PDR', 'LA')
    country_names_list.append(laos)

    macedonia = Country('North Macedonia', 'Macedonia', 'MK')
    country_names_list.append(macedonia)

    russia = Country('Russia', 'Russian Federation', 'RU')
    country_names_list.append(russia)

    south_korea = Country('South Korea', 'Korea', 'KR')
    country_names_list.append(south_korea)

    saint_kitts_and_nevis = Country('St. Kitts and Nevis', 'Saint Kitts and Nevis', 'KN')
    country_names_list.append(saint_kitts_and_nevis)

    saint_lucia = Country('St. Lucia', 'Saint Lucia', 'LC')
    country_names_list.append(saint_lucia)

    saint_vincent_and_the_grenadines = Country('St. Vincent and the Grenadines', 'Saint Vincent and the Grenadines',
                                               'VC')
    country_names_list.append(saint_vincent_and_the_grenadines)

    syria = Country('Syria', 'Syrian Arab Republic', 'SY')
    country_names_list.append(syria)

    gambia = Country('The Gambia', 'Gambia', 'GM')
    country_names_list.append(gambia)

    usa = Country('United States', 'USA', 'US')
    country_names_list.append(usa)

    vietnam = Country('Vietnam', 'Viet Nam', 'VN')
    country_names_list.append(vietnam)
    return country_names_list


def clean_up_country_list(country_names_list):
    country_names_list[116].wid_code = 'NA'  # The two character code for Namibia gets spuriously converted into NaN
    country_names_list.remove(
        country_names_list[87])  # Georgia the country gets spuriously matched to Georgia the state
    return country_names_list


def pull_value_from_df(df, variable, year, percentile='p0p100'):
    matched_codes = df[df['variable'] == variable]
    matched_percentiles = matched_codes[matched_codes['percentile'] == percentile]
    matched_years = matched_percentiles[matched_percentiles['year'] == year]
    df_value = matched_years['value']
    if not df_value.empty:
        value = df_value.values[0]
    else:
        value = None
    return value


def extract_economic_features_from_wid_df(wid_df):
    out_df = []
    columns = []
    variable_codes = [['anninc992i', ''],  # National Income Per Adult
                      ['agdpro992i', ''],  # GDP Per Adult
                      ['sptinc992j', 'p90p100'],  # Income Share (Top 10%)
                      ['sptinc992j', 'p50p90'],  # Income Share (Middle 40%)
                      ['sptinc992j', 'p0p50'],  # Income Share (Bottom 50%)
                      ['sptinc992j', 'p99p100'],  # Income Share (Top 1%)
                      ['anweal992i', ''],  # National Wealth Per Adult
                      ['wwealn999i', ''],  # National Income Ratio
                      ['shweal992j', 'p90p100'],  # Wealth Share (Top 10%)
                      ['shweal992j', 'p50p90'],  # Wealth Share (Middle 40%)
                      ['shweal992j', 'p0p50'],  # Wealth Share (Bottom 50%)
                      ['shweal992j', 'p99p100'],  # Wealth Share (Top 1%)
                      ['iquali999i', '']]  # Inequality Transparency

    columns.extend(['National Income Per Adult',
                    'GDP Per Adult',
                    'Income Share (Top 10%)',
                    'Income Share (Middle 40%)',
                    'Income Share (Bottom 50%)',
                    'Income Share (Top 1%)',
                    'National Wealth Per Adult',
                    'National Income Ratio',
                    'Wealth Share (Top 10%)',
                    'Wealth Share (Middle 40%)',
                    'Wealth Share (Bottom 50%)',
                    'Wealth Share (Top 1%)',
                    'Inequality Transparency'])

    variable_values = []
    for i in range(0, len(variable_codes)):
        code = variable_codes[i][0]
        percentile = variable_codes[i][1]
        if not percentile:
            percentile = 'p0p100'
        today = date.today()
        year = today.year
        value = None
        while not value:
            value = pull_value_from_df(wid_df, code, year, percentile)  # Per Adult National Income
            year = year - 1
            if year < 1500:
                value = float('nan')
        variable_values.append(value)
    out_df.extend(variable_values)

    return out_df, columns


def assemble_row(country_code, wid_values, wid_columns, fiw_name, fiw_values, fiw_columns):
    df_values = [fiw_name, country_code]
    columns = ['Country Name', 'Country Code']

    df_values.extend(wid_values)
    columns.extend(wid_columns)

    df_values.extend(fiw_values)
    columns.extend(fiw_columns)

    df_out = pd.DataFrame([df_values], columns=columns)
    return df_out


def plot_correlation_heatmap(corr):
    mask = np.tril(np.ones_like(corr, dtype=bool))
    _, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    g = sns.heatmap(corr, mask=mask, cmap=cmap, center=0, vmin=-1, vmax=1,
                    square=True, linewidths=.5, cbar_kws={"shrink": .5})
    g.axes.xaxis.set_ticks_position("top")
    plt.xticks(rotation=90)


def plot_scatter_and_model_fit(model_fit_dict, ax):
    # Load parameters and options
    x = model_fit_dict['x']
    y = model_fit_dict['y']
    w = model_fit_dict['w']
    predictions_wls = model_fit_dict['model_fit_values']
    xlabel = model_fit_dict['xlabel']
    ylabel = model_fit_dict['ylabel']
    str_loc_x = model_fit_dict['str_loc_x']
    str_loc_y = model_fit_dict['str_loc_y']

    # Make axes
    ax.tick_params(labelsize=18)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)

    # Scale weights for visualization
    w_plot = w * 60

    # Make text insert type 1: t and p values
    t = model_fit_dict['t']
    p = model_fit_dict['p']
    if t is not None and p is not None:
        if p < 0.001:
            p_str = 'p < 0.001'
        else:
            p_str = 'p = %.3f' % p
        t_str = 't = %.1f' % t
        tp_strs = t_str + '\n' + p_str

        ax.text(str_loc_x, str_loc_y, tp_strs,
                horizontalalignment='left',
                transform=ax.transAxes,
                fontsize=24)

    # Make text insert type 2: sentiment and polarization
    sentiment = model_fit_dict['sentiment']
    polarization = model_fit_dict['polarization']
    if sentiment is not None and polarization is not None:
        s_str = 'Sentiment = %.1f%%' % sentiment
        p_str = 'Polarization = %.2f' % polarization
        sp_strs = s_str + '\n' + p_str

        ax.text(str_loc_x, str_loc_y, sp_strs,
                horizontalalignment='left',
                transform=ax.transAxes,
                fontsize=24)

    # Plot data and model fit
    ax.scatter(x, y, s=w_plot)
    ax.plot(x, predictions_wls, color='red')


def specify_scatter_plot(df, x_factor, y_factor):
    scatter_plot_defaults = {'w': pd.Series(np.ones(len(df))), 'str_loc_x': .1, 'str_loc_y': .8, 'sentiment': None,
                             'polarization': None}

    scatter_plot_dict = scatter_plot_defaults.copy()
    x_frame = df[[x_factor]]
    y_frame = df[[y_factor]]
    xx = sm.add_constant(x_frame)  # adding a constant
    model = sm.WLS(y_frame, xx, missing='drop').fit()
    scatter_plot_dict['x'] = x_frame
    scatter_plot_dict['y'] = y_frame
    scatter_plot_dict['model_fit_values'] = model.predict(xx)
    scatter_plot_dict['p'] = model.pvalues[1]
    scatter_plot_dict['t'] = model.tvalues[1]
    scatter_plot_dict['xlabel'] = x_factor
    scatter_plot_dict['ylabel'] = y_factor

    return scatter_plot_dict


def count_missing_data(df_column):
    data_values = df_column.values
    missing_data_count = sum(np.isnan(data_values))
    return missing_data_count


def make_missing_data_df(econ_df):
    table1 = {'Covariate': [], 'Data Present': [], 'Data Present %': []}
    for column in econ_df:
        data_present = len(econ_df) - count_missing_data(econ_df[column])
        data_present_pct = data_present / len(econ_df) * 100
        table1['Covariate'].append(column)
        table1['Data Present'].append(data_present)
        table1['Data Present %'].append(data_present_pct)
    missing_data_df = pd.DataFrame.from_dict(table1)
    return missing_data_df


def run_model(x_frame, y_frame):
    xx = sm.add_constant(x_frame)  # adding a constant
    model = sm.OLS(y_frame, xx, missing='drop').fit()
    print_model = model.summary()
    print(print_model)
