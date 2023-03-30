import pymongo
from pymongo import MongoClient

import pandas as pd
import json
import numpy as np
import os
import seaborn as sns
import re

import tkinter as tk
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from pandastable import Table

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
from matplotlib.figure import Figure
import matplotlib.style as mstyle

matplotlib.use('TkAgg')


def open_collections():
    global inv_og_collection, insp_og_collection, vio_og_collection, inv_prep_collection, insp_prep_collection, vio_prep_collection
    client = pymongo.MongoClient("mongodb://localhost:27017/")
    db = client["VendorsDatabase"]
    inv_og_collection = db["InventoryCSVData"]  # Original data - Inventory 'inv' collection
    insp_og_collection = db["InspectionsCSVData"]  # Original data - Inspection 'insp' collection
    vio_og_collection = db["ViolationsCSVData"]  # Original data - Violations 'vio' collection
    inv_prep_collection = db["InventoryAnalysisData"]  # Cleaned data - Inventory 'inv' collection
    insp_prep_collection = db["InspectionsAnalysisData"]  # Cleaned data - Inspection 'insp' collection
    vio_prep_collection = db["ViolationsAnalysisData"]  # Cleaned data - Violations 'vio' collection


# OPEN CSV AND TRANSLATE TO JSON

def create_dataframe(csv_filename, json_filename):
    csv_filename = csv_filename  # CSV path name
    json_filename = json_filename  # JSON file name to be saved
    # Converting to pandas dataframe to be translated into a JSON format orientated by
    # columns. Pandas dataframe from this JSON file in a readable format
    csv_file = pd.DataFrame(pd.read_csv(csv_filename, sep=",", header=0, index_col=False))
    csv_file.to_json(json_filename, orient="columns", double_precision=0,
                     force_ascii=True, default_handler=None, indent=4)
    with open(json_filename) as data_file:
        data_dict = json.load(data_file)
    dataframe = pd.DataFrame.from_dict(data_dict, orient='columns')
    return dataframe


# SET GLOBAL DATAFRAME OPTIONS

def set_dataframe_options():
    pd.set_option('display.max_columns', None)  # Change 5 to None to to show all columns
    pd.set_option('display.max_rows', 30)  # Change 5 to None to to show all columns
    pd.set_option('precision', 0)  # Set default to no decimal places
    pd.set_option('display.float_format', lambda x: '%.0f' % x)  # Display floats to 0 decimal place
    pd.set_option('display.width', pd_width)


# CLEAN DATA

# Replace INACTIVE program status with NaN using the numpy 'Not a Number' value
def replace_inactive(dataframe):
    dataframe['PROGRAM STATUS'].replace('INACTIVE', np.nan, inplace=True)
    return dataframe


# After replacing 'INACTIVE' with NaN, remove any rows with an NaN value
# (this will also catch any rows that do not have full data i.e. no zip code
def remove_nulls(dataframe):
    dataframe.dropna(inplace=True)


# Remove any rows where all columns are duplicated
def remove_duplicates(dataframe):
    dataframe.drop_duplicates(subset=None, keep='first', inplace=True)


# Extract anything within parenthesis
def extract_parenthesis(string_in):
    pattern = re.compile(r"\(.*?\)")
    result_inner_split = pattern.search(string_in)
    if result_inner_split:
        result_inner = result_inner_split.group()
        result_outer = string_in.replace(str(result_inner), '')
        return result_outer, result_inner.replace('(', '').replace(')', '')
    else:
        results_packed = (string_in, '')
        return results_packed


# Use the above function to extract / split the PE DESCRIPTION column as required by the client
def extract_pe_description(dataframe):
    type_list = []
    number_list = []
    print(dataframe.head())
    for row in dataframe['PE DESCRIPTION']:
        des, number = extract_parenthesis(row)
        type_list.append(des)
        number_list.append(number)
    dataframe.insert(10, "TYPE OF SEATING", type_list)
    dataframe.insert(11, "NUMBER OF SEATS", number_list)
    dataframe.drop(columns="PE DESCRIPTION", inplace=True)
    print(dataframe.head())
    return dataframe


# Required for all data sets
def clean_all_data(dataframe):
    remove_nulls(dataframe)
    remove_duplicates(dataframe)
    return dataframe


# GET TAB NAME FOR TAB 1 FRAMES
# Returns the name of the tab to be passed into the Labelframe label on each tab
def get_tabname():
    global title_var
    title_var.set(data_tabs.notebook.tab(data_tabs.notebook.select(), "text"))
    return title_var


# CLEAN ORIGINAL DATA AND SHOW IN TAB 1 FRAME

# Clean original data as required and output as a pandastable within the tab's frame
def prepare_data_button(frame):
    # If statement to check which tab is selected (i.e. Inventory (0), Inspections (1) or Violations (2) and therefore
    # use and output the correct dataset
    global inv_dataframe, insp_dataframe, vio_dataframe, data_cleaned
    data_cleaned = True
    if data_tabs.notebook.index(data_tabs.notebook.select()) == 0:
        print()
        print("Before cleaning and preparinng: ")
        print(inv_dataframe[['PE DESCRIPTION', 'FACILITY CITY', 'FACILITY ADDRESS', 'Zip Codes']].head(15))
        inv_tab.cleaned_status_complete()
        cleaned = clean_all_data(inv_dataframe)
        extracted = extract_pe_description(cleaned)
        inv_dataframe = validate_zip_code(extracted)
        print()
        print("After cleaning and preparinng: ")
        print(inv_dataframe[['TYPE OF SEATING', 'NUMBER OF SEATS' , 'FACILITY CITY', 'FACILITY ADDRESS', 'ZIP CODE']].head(15))
        pt = Table(frame, dataframe=inv_dataframe, height=pd_height, width=pd_width, showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0, sticky='new')
        pt.show()

    elif data_tabs.notebook.index(data_tabs.notebook.select()) == 1:  # Inspections
        insp_tab.cleaned_status_complete()
        replaced = replace_inactive(insp_dataframe)
        extracted = extract_pe_description(replaced)
        cleaned = clean_all_data(extracted)
        insp_dataframe = validate_zip_code(cleaned)
        pt = Table(frame, dataframe=insp_dataframe, height=pd_height, width=pd_width, showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0, sticky='new')
        pt.show()
        pt = Table(seating_tab.data_frame, dataframe=insp_dataframe, height=pd_height, width=pd_width,
                   showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0, sticky='new')
        pt.show()
        pt = Table(zip_tab.data_frame, dataframe=insp_dataframe, height=pd_height, width=pd_width,
                   showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0, sticky='new')
        pt.show()

    else:  # Violations
        vio_tab.cleaned_status_complete()
        vio_dataframe = clean_all_data(vio_dataframe)
        pt = Table(frame, dataframe=vio_dataframe, height=pd_height, width=pd_width, showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0, sticky='new')
        pt.show()


# MEAN, MEDIAN AND MODES FOR TAB 2

# Mean inspection score per year for each type of seating and each zip code
def get_mean():
    global insp_dataframe
    # If statement to check which tab is selected (i.e. Type of Seating Analysis (0) or Zip Code Analysis (1) and
    # output the correct data.
    if mmm_tabs.notebook.index(mmm_tabs.notebook.select()) == 0:
        data = inspection_dataframe(insp_dataframe)
        mean = data.groupby(["TYPE OF SEATING", 'YEAR']).agg({'SCORE': ['mean']})
        mean.reset_index(inplace=True)
        mean.columns = ["TYPE OF SEATING", "YEAR", "MEAN INSPECTIONS SCORE"]
        print(mean.head(11))
        pt = Table(seating_tab.data_frame, dataframe=mean, height=pd_height, width=pd_width,
                   showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0, sticky='new')
        pt.show()
    else:
        data = inspection_dataframe(insp_dataframe)
        mean = data.groupby(["ZIP CODE", 'YEAR']).agg({'SCORE': ['mean']})
        mean.reset_index(inplace=True)
        mean.columns = ["ZIP CODE", "YEAR", "MEAN INSPECTIONS SCORE"]
        pt = Table(zip_tab.data_frame, dataframe=mean, height=pd_height, width=pd_width,
                   showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0, sticky='new')
        pt.show()


# Median inspection score per year for each type of seating and each zip code
def get_median():
    global insp_dataframe
    # If statement to check which tab is selected (i.e. Type of Seating Analysis (0) or Zip Code Analysis (1) and
    # output the correct data.
    if mmm_tabs.notebook.index(mmm_tabs.notebook.select()) == 0:
        data = inspection_dataframe(insp_dataframe)
        med = data.groupby(["TYPE OF SEATING", 'YEAR']).agg({'SCORE': ['median']})
        med.reset_index(inplace=True)
        med.columns = ["TYPE OF SEATING", "YEAR", "MEDIAN INSPECTION SCORE"]
        pt = Table(seating_tab.data_frame, dataframe=med, height=pd_height, width=pd_width,
                   showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0)
        pt.show()
    elif mmm_tabs.notebook.index(mmm_tabs.notebook.select()) == 1:
        data = inspection_dataframe(insp_dataframe)
        med = data.groupby(["ZIP CODE", 'YEAR']).agg({'SCORE': ['median']})
        med.reset_index(inplace=True)
        med.columns = ["ZIP CODES", "YEAR", "MEDIAN INSPECTION SCORE"]
        pt = Table(zip_tab.data_frame, dataframe=med, height=pd_height, width=pd_width,
                   showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0, sticky='new')
        pt.show()
    else:
        return


# Mode inspection score per year for each type of seating and each zip code
def get_mode():
    global insp_dataframe
    # If statement to check which tab is selected (i.e. Type of Seating Analysis (0) or Zip Code Analysis (1) and
    # output the correct data.
    if mmm_tabs.notebook.index(mmm_tabs.notebook.select()) == 0:
        data = inspection_dataframe(insp_dataframe)
        mode = data.groupby(["TYPE OF SEATING", 'YEAR'])['SCORE'].apply(pd.Series.mode).to_frame()
        mode.reset_index(inplace=True)
        mode.drop(columns='level_2', inplace=True)
        mode.columns = ["TYPE OF SEATING", "YEAR", "MODE INSPECTION SCORE"]
        pt = Table(seating_tab.data_frame, dataframe=mode, height=pd_height, width=pd_width,
                   showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0)
        pt.show()
    elif mmm_tabs.notebook.index(mmm_tabs.notebook.select()) == 1:
        data = inspection_dataframe(insp_dataframe)
        mode = data.groupby(["ZIP CODE", 'YEAR'])['SCORE'].apply(pd.Series.mode).to_frame()
        mode.reset_index(inplace=True)
        mode.drop(columns='level_2', inplace=True)
        mode.columns = ["ZIP CODES", "YEAR", "MODE INSPECTION SCORE"]
        pt = Table(zip_tab.data_frame, dataframe=mode, height=pd_height, width=pd_width,
                   showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=0, sticky='new')
        pt.show()
    else:
        return


def get_mean_button():
    # Check Inspection data has been loaded and cleaned. Mean calculation will pull from this. Else, show error popup.
    if load_cleaned_check_single(insp_tab):
        get_mean()
    else:
        create_popup(LoadErrorPopUp).run()
        return


def get_med_button():
    # Check Inspection data has been loaded and cleaned. Median calculation will pull from this. Else, show error popup.
    if load_cleaned_check_single(insp_tab):
        get_median()
    else:
        create_popup(LoadErrorPopUp).run()
        return


def get_mode_button():
    # Check Inspection data has been loaded and cleaned. Mode calculation will pull from this. Else, show error popup.
    if load_cleaned_check_single(insp_tab):
        get_mode()
    else:
        create_popup(LoadErrorPopUp).run()
        return


# Cut dataframe to only include required data for violation and correlation analysis
def inspection_dataframe(
        dataframe):
    # Following columns not required
    to_drop = ['OWNER ID', 'OWNER NAME', 'FACILITY ID',
               'FACILITY NAME', 'RECORD ID', 'PROGRAM NAME', 'PROGRAM STATUS',
               'PROGRAM ELEMENT (PE)', 'NUMBER OF SEATS',
               'FACILITY ADDRESS', 'FACILITY CITY', 'FACILITY STATE', 'FACILITY ZIP',
               'SERVICE CODE', 'SERVICE DESCRIPTION', 'GRADE',
               'SERIAL NUMBER', 'EMPLOYEE ID', 'Location',
               '2011 Supervisorial District Boundaries (Official)',
               'Census Tracts 2010', 'Board Approved Statistical Areas']
    dates_list = []
    year_list = []

    cut_data = dataframe.drop(columns=to_drop)

    # Translate ACTIVITY DATE entries to a readable format, i.e. datetime.
    for row in cut_data["ACTIVITY DATE"]:
        date = pd.to_datetime(row, format='%m/%d/%Y')
        dates_list.append(date)

    cut_data.insert(0, "DATETIME", dates_list)
    cut_data.drop(columns="ACTIVITY DATE", inplace=True)

    # Get the year from the newly formed datatime entries
    for date in cut_data["DATETIME"]:
        year = date.year
        year_list.append(year)

    # Add year column to dataframe
    cut_data.insert(0, "YEAR", year_list)
    cut_data.set_index(['YEAR'], inplace=True)

    return cut_data


# VIOLATION BAR CHART FOR TAB 3

# Return dataframe showing number of times each violation has been committed, its code and description
def barplot_dataframe(violations_dataframe):
    # Get the count of each unique violation code and output to a dataframe
    barplot_data = violations_dataframe['VIOLATION CODE'].value_counts().to_frame()
    # Add a violations description to new dataframe
    barplot_data["Violations Description"] = violations_dataframe["VIOLATION DESCRIPTION"]
    barplot_data.reset_index(inplace=True)
    # Rename columns to be easily referred to when plotting the bar graph
    barplot_data.columns = ['Violation Code', 'Count', 'Violation Description']
    # Sort by violation code (so violations code / description key can be easily readable)
    barplot_data.sort_values(by=['Violation Code'], inplace=True)
    print(barplot_data[['Violation Code', 'Count']].head())
    return barplot_data


# Create key to match violation code with its corresponding description
def violations_key():
    global vio_dataframe
    vio_dataframe.sort_values(by="VIOLATION CODE", inplace=True)

    code_df = vio_dataframe["VIOLATION CODE"]
    des_df = vio_dataframe["VIOLATION DESCRIPTION"]
    # Create list of unique violation codes
    codes_list = []
    for x in code_df.unique():
        codes_list.append(x)
    # Create list of unique violation descriptions
    des_list = []
    for x in des_df.unique():
        des_list.append(x)
    # Use two lists to create new dataframe mapping codes to descriptions.
    key_df = pd.DataFrame(list(zip(codes_list, des_list)),
                          columns=['CODE', 'DESCRIPTION'])

    return key_df


# Create bar graph to show number of times each violation has been committed.
def create_bar_graph():
    global vio_dataframe

    # Create violations count dataframe using cleaned Violations dataset
    vio_graph = barplot_dataframe(vio_dataframe)
    x_col = vio_graph['Violation Code']
    y_col = vio_graph['Count']

    sns.set_palette('Set2', n_colors=len(x_col))  # colours will cycle for as many violations codes

    # Create figure and add subplot. Add bar graph to seaborn barplot to subplot
    fig = Figure(figsize=(13.5, 6), dpi=100, tight_layout=True)
    ax = fig.add_subplot(111)
    graph = sns.barplot(x=x_col, y=y_col, data=vio_graph, ax=ax)

    ax.set_title('Number of establishments that have committed each type of violation', fontsize=11)
    ax.set_ylabel('Number of occurrences', fontsize=10)
    ax.set_xlabel('Violation code (see key for description)', fontsize=10)
    # Rotate x ticks 90 degrees to be more readable side by side
    graph.set_xticklabels(graph.get_xticklabels(), rotation=90, fontsize=7)
    plt.rc('xtick', direction='out', color='gray')
    plt.rc('ytick', direction='out', color='gray')

    # Annotation bars with their values (heights). To make it easier to read, any value under 100 (2-digits) is
    # presented horizontally, otherwise values are presented at 90 degrees to avoid them clashing.
    # Formatted to zero decimal places with a thousand comma.
    for p in graph.patches:
        if p.get_height() < 100 or p.get_height() > 100000:
            graph.annotate(format(p.get_height(), ',.0f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', size=6,
                           xytext=(0, 3),
                           textcoords='offset points')
        else:
            graph.annotate(format(p.get_height(), ',.0f'),
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', size=6,
                           xytext=(0, 3), rotation=90,
                           textcoords='offset points')
    return fig, graph


# Validate whether violations data has been loaded and cleaned using the complete/incomplete string variable
def create_bar_canvas():
    if load_cleaned_check_single(vio_tab):
        f, graph = create_bar_graph()
        canvas = FigureCanvasTkAgg(f, master=vioanalysis_tab.data_frame)
        toolbar = NavigationToolbar2Tk(canvas, vioanalysis_tab.data_frame)
        toolbar.update()
        canvas.draw()
        canvas.get_tk_widget().pack()
        return canvas
    else:
        create_popup(LoadErrorPopUp).run()


# Colours to cycle through
def get_colours_list(x_column):
    colours = ['black', 'maroon', 'coral', 'saddlebrown', 'darkorange', 'goldenrod', 'olive', 'darkolivegreen',
               'darkseagreen', 'darkgreen', 'lightseagreen', 'darkslategrey', 'darkcyan', 'deepskyblue', 'steelblue',
               'slategrey', 'midnightblue', 'slateblue', 'indigo', 'darkmagenta', 'mediumvioletred', 'palevioletred']

    div = (len(x_column) // len(colours)) + 1
    colours_list = colours * div
    colour_map = dict(zip(x_column, colours_list))
    return colour_map


# SCATTER PLOT FOR TAB 4

# Merge dataframe to be used when determining correlation between violation count per vendor and their zip code
def merge_dataframes(inspections, violations):
    # Columns not required
    to_drop = ['ACTIVITY DATE', 'OWNER ID', 'OWNER NAME', 'FACILITY NAME',
               'RECORD ID', 'PROGRAM NAME', 'PROGRAM STATUS',
               'PROGRAM ELEMENT (PE)', 'TYPE OF SEATING', 'NUMBER OF SEATS',
               'FACILITY ADDRESS', 'FACILITY STATE', 'FACILITY ZIP',
               'SERVICE CODE', 'SERVICE DESCRIPTION', 'SCORE', 'GRADE',
               'EMPLOYEE ID', 'Location',
               '2011 Supervisorial District Boundaries (Official)',
               'Census Tracts 2010', 'Board Approved Statistical Areas']

    # Merge inspections and violations dataframe
    cut_data = inspections.drop(columns=to_drop)  # Inspection Analysis Dataframe
    # Counts how many times each vendor commits a violation. Serial numbers are unique to each vendor, thus
    # (as duplicates have already been removed) each time a serial number occurs, it means they have
    # committed another violation.
    correlation_dataframe = violations['SERIAL NUMBER'].value_counts().to_frame()
    correlation_dataframe.reset_index(inplace=True)
    correlation_dataframe.columns = ['SERIAL NUMBER', 'NUMBER OF VIOLATIONS']
    combined_dataframe = pd.merge(cut_data, correlation_dataframe, on='SERIAL NUMBER')  # Merge dataframes
    print()
    print("Merged inspections/violations dataframe: ")
    print(combined_dataframe.head())

    return combined_dataframe


# Validate zip code (add 0's to front if missing to make it a valid 5-digit zip)
# Validate city and address name has no special (nbsp) characters in
def validate_zip_code(dataframe):
    zip_list = []
    for value in dataframe['Zip Codes']:
        number = int(value)  # convert float to int
        string = str(number)  # convert int to string
        fill = string.zfill(5)  # add 0's to start
        zip_list.append(fill)  # add to list to insert into dataframe

    dataframe.drop(columns="Zip Codes", inplace=True)
    dataframe["ZIP CODE"] = zip_list

    # Removing any nbsp
    pattern1 = re.compile(r"[^a-zA-Z0-9]+", flags=re.IGNORECASE)
    pattern2 = re.compile(r"[^a-zA-Z]+", flags=re.IGNORECASE)
    try:
        address = dataframe['FACILITY ADDRESS'].str.replace(pattern2, ' ')
        dataframe['FACILITY ADDRESS'] = address
    finally:
        city = dataframe['FACILITY CITY'].str.replace(pattern1, ' ')
        dataframe['FACILITY CITY'] = city
        return dataframe


# Get the mean number of violations committed per vendor for each city (zip codes grouped by city so
# the large data set can be visualised and analysed).
def get_merged_mean(data):
    mean = data.groupby(["FACILITY CITY"]).agg({'NUMBER OF VIOLATIONS': ['mean']})
    mean.reset_index(inplace=True)
    mean.columns = ["FACILITY CITY", "MEAN"]
    count = mean["FACILITY CITY"].count()
    # Create column with numbers 0 to length on city list for the number / city key.
    # Necessary as city labels are lengthy, so shortened to numbers on the graph
    x_range = (range(0, count))
    x_list = list(map(str, x_range))
    mean["X LABEL"] = x_list
    print()
    print("Mean number of violations per city: ")
    print(mean.head())
    return mean


# Merge inspections and violations dataframe and get the mean number of violations committed per vendor for each city
def corr_graph_dataframe():
    global insp_dataframe, vio_dataframe
    merged_data = merge_dataframes(insp_dataframe, vio_dataframe)
    corr_graph = get_merged_mean(merged_data)
    return corr_graph


# Manipulate the range of values used to generate output. Takes two entry values as parameters
# and returns a dataframe with only values in between entry range.
def select_range_dataframe(a, b):
    corr = corr_graph_dataframe()
    range_df = corr[(corr['MEAN'] >= a) & (corr['MEAN'] <= b)]
    return range_df


# Get lower bound value (a) from global entry box variable
def get_a():
    global lower_var
    a = lower_var.get()
    print("Lower bound = " + str(a))
    return float(a)


# Get upper bound value (b) from global entry box variable
def get_b():
    b = upper_var.get()
    print("Upper bound = " + str(b))
    return float(b)


# Closes existing canvas, figure and axes so as to overwrite graph and not append.
# Plot new graph using selected range dataframe
def update_scatter_graph(a, b):
    global fig, ax, canvas
    plt.cla()
    plt.clf()
    plt.close()

    data = select_range_dataframe(a, b)
    create_scatter_plot(data)


# Create scatter plot to determine whether there is a significant correlation between
# the number of violations committed per vendor and their zip code
def create_scatter_plot(data):
    global all_data
    fig, ax, canvas = create_scatter_canvas()
    x_col = data["FACILITY CITY"]
    y_col = data["MEAN"]
    x_label = data["X LABEL"]

    # Map and cycle through colours
    colour_map = get_colours_list(x_col)

    marker_size = 15
    # Plot matplotlib scatter plot on axes
    graph = ax.scatter(x_col, y_col, c=x_col.map(colour_map), s=marker_size)

    # To combat space issues, reduce number of labels shown as plot numbers increase
    if len(x_label) > 40:
        # If (when range selected), number of plots/labels is more than 40 but less than 120, show every other label
        if 40 < len(x_label) < 120:
            n = 2  # Keeps every 5th label
        else:
            # If number of plots/labels is more than or equal to 120, show every 5th label
            n = 5
        [l.set_visible(False) for (i, l) in enumerate(ax.xaxis.get_ticklabels()) if i % n != 0]
    else:
        # Less than 40, show all labels
        pass

    # For the show graph button (all cities)
    if all_data == True:
        # Show annotations when plot is hovered over
        annot = ax.annotate("", xy=(0, 0), xytext=(15, 10), textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"), fontsize=9)
        annot.set_visible(False)

        def update_annot(ind):

            pos = graph.get_offsets()[ind["ind"][0]]
            annot.xy = pos

            text = "{}, \n{}".format(" ".join([x_col[n] for n in ind["ind"]]),
                                     " ".join(["{:.2f}".format(y_col[n]) for n in ind["ind"]]))
            annot.set_text(text)
            annot.get_bbox_patch().set_alpha(0.8)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = graph.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)

    else:
        pass

    ax.set_aspect('auto')
    ax.grid('on')
    plt.xticks(x_col, x_label, fontsize=7)
    ax.set_title('Correlation between number of violations committed per vendor and their zip code (grouped by city)',
                 fontsize=11)
    ax.set_ylabel('Mean number of violations committed per zip code (grouped by city)', fontsize=10)
    ax.set_xlabel('City (see city key for reference)', fontsize=10)
    plt.rc('xtick', direction='out', color='gray')
    plt.rc('ytick', direction='out', color='gray')

    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.RIGHT)

    return canvas


def create_scatter_canvas():
    if load_cleaned_check_double(insp_tab, vio_tab):
        plt.cla()
        plt.clf()
        plt.close()
        mstyle.use('seaborn')
        fig, ax = plt.subplots(figsize=(13.5, 6), dpi=100, tight_layout=True)
        canvas = FigureCanvasTkAgg(fig, master=corranalysis_tab.data_frame)
        return fig, ax, canvas
    else:
        create_popup(LoadErrorPopUp).run()


# Create scatter graph when 'show graph (all cities)' is chosen (rather than show range).
# Check if data needed has been cleaned and prepared. If not, show error popup
def all_scatter_button():
    if load_cleaned_check_double(insp_tab, vio_tab):
        all_data_true()
        create_scatter_plot(
            select_range_dataframe(corr_graph_dataframe()["MEAN"].min(),
                                   corr_graph_dataframe()["MEAN"].max()))
    else:
        create_popup(LoadErrorPopUp).run()


def range_scatter_button():
    if load_cleaned_check_double(insp_tab, vio_tab):
        all_data_false()
        create_popup(GraphRangePopUp).run()
    else:
        create_popup(LoadErrorPopUp).run()


# Commentary of determining a corrlation. Gets the min and max values and their corresponding city names
def set_analysis_note():
    graph = corr_graph_dataframe()
    max_mean = graph["MEAN"].max()
    max_mean_str = "{:.2f}".format(max_mean)
    max_df = graph[(graph['MEAN'] == max_mean)]
    max_list = []
    for x in max_df["FACILITY CITY"]:
        max_list.append(x)
    if len(max_list) == 1:
        max_city_string = str(max_list)
    else:
        max_city_string = " and ".join(max_list)

    min_mean = graph["MEAN"].min()
    min_mean_str = "{:.2f}".format(min_mean)
    min_df = graph[(graph['MEAN'] == min_mean)]
    min_list = []
    for x in min_df["FACILITY CITY"]:
        min_list.append(x)
    if len(min_list) == 1:
        min_city_string = str(min_list)
    else:
        min_city_string = " and ".join(min_list)

    return max_city_string, max_mean_str, min_city_string, min_mean_str


# Displays labels next to city name so can be easily inferred from the graph (labels are too lengthy to have side by
# side)
def city_key():
    global lower_var, upper_var

    if lower_var.get() == "" or upper_var.get() == "":
        data = corr_graph_dataframe()
    else:
        data = select_range_dataframe(get_a(), get_b())

    num_col = data["X LABEL"]
    city_col = data["FACILITY CITY"]

    num_list = []
    for x in num_col:
        num_list.append(x)

    city_list = []
    for x in city_col:
        city_list.append(x)

    key_df = pd.DataFrame(list(zip(num_list, city_list)),
                          columns=['NUMBER CODE', 'CITY'])
    return key_df


def all_data_true():
    global all_data
    all_data = True
    return all_data


def all_data_false():
    global all_data
    all_data = False
    return all_data


# CREATE POPUP
# Create the required pop up (pop up class is passed as a parameter) and assign to the top level window
def create_popup(classtype):
    top_level = tk.Toplevel()
    popup = classtype(top_level)
    return popup


# Check if violations data has been cleaned/prepared and if it has, runs the violations code key pop up.
# If not, run error pop up
def create_vio_key_popup():
    if load_cleaned_check_single(vio_tab):
        create_popup(VioKeyPopUp).run()
    else:
        create_popup(LoadErrorPopUp).run()


# Check if inspections and violations data has been cleaned/prepared and if it has, runs the city key pop up.
# If not, run error pop up
def create_city_key_popup():
    if load_cleaned_check_double(insp_tab, vio_tab):
        create_popup(CorrKeyPopUp).run()
    else:
        create_popup(LoadErrorPopUp).run()


# ERROR CHECKS - LOADED AND/OR CLEANED
# Check the status of one tab's (passed as a parameter) loaded and cleaned status text variables
def load_cleaned_check_single(tab):
    global tab_var, datatype_var
    if tab.cleaned_var.get() == "COMPLETE":
        return True
    elif tab.loaded_var.get() == "COMPLETE" and tab.cleaned_var.get() == "INCOMPLETE":
        tab_var = "clean"
        if tab == insp_tab:
            datatype_var = "Inspections"
        else:
            datatype_var = "Violations"
        return False
    else:
        tab_var = "load and clean"
        if tab == insp_tab:
            datatype_var = "Inspections"
        else:
            datatype_var = "Violations"
        return False


# Check the status of two tabs' (passed as parameters) loaded and cleaned status text variables
def load_cleaned_check_double(tab1, tab2):
    global tab_var
    global datatype_var
    if tab1.cleaned_var.get() == "COMPLETE" and tab2.cleaned_var.get() == "COMPLETE":
        return True
    elif tab1.cleaned_var.get() == "COMPLETE" and tab2.loaded_var.get() == "COMPLETE" and tab2.cleaned_var.get() == "INCOMPLETE":
        tab_var = "clean"
        datatype_var = "Violations"
        return False
    elif tab1.cleaned_var.get() == "COMPLETE" and tab2.loaded_var.get() == "INCOMPLETE":
        tab_var = "load and clean"
        datatype_var = "Violations"
        return False
    elif tab1.loaded_var.get() == "COMPLETE" and tab1.cleaned_var.get() == "INCOMPLETE" and tab2.cleaned_var.get() == "COMPLETE":
        tab_var = "clean"
        datatype_var = "Inspections"
        return False
    elif tab1.loaded_var.get() == "INCOMPLETE" and tab2.cleaned_var.get() == "COMPLETE":
        tab_var = "load and clean"
        datatype_var = "Inspections"
        return False
    elif tab1.loaded_var.get() == "INCOMPLETE" and tab2.loaded_var.get() == "INCOMPLETE":
        tab_var = "load and clean"
        datatype_var = "Inspections and Violations"
        return False
    else:
        tab_var = "clean"
        datatype_var = "Inspections and Violations"
        return False


# SAVE TO MONGO DATABASE
# Check if status variables to decide where data should be saved
# (i.e. in CSV data collection or in prepared data collection)
# Overwrite anything already in the collection

def save_inventory():
    global inv_dataframe, inv_og_collection, inv_prep_collection, datatype_var, tab_var
    json_path = 'InventoryDatabase.json'
    if inv_tab.loaded_var.get() == "INCOMPLETE":
        if not close_window:
            datatype_var = 'Inventory'
            tab_var = 'load'
            create_popup(LoadErrorPopUp).run()
            return
        else:
            return
    else:
        data = inv_dataframe
        if inv_tab.cleaned_var.get() == "COMPLETE":
            collection = inv_prep_collection
        else:
            collection = inv_og_collection
    try:
        collection.delete_many({})
    finally:
        data.to_json(json_path, orient="records", double_precision=0,
                     force_ascii=True, default_handler=str, indent=4)
        with open(json_path) as json_file:
            file_data = json.load(json_file)

        collection.insert_many(file_data)
        print("Inventory Data saved to MongoDB database")
        return


def save_inspections():
    global insp_dataframe, insp_og_collection, insp_prep_collection, datatype_var, tab_var
    json_path = 'InspectionsDatabase.json'
    if insp_tab.loaded_var.get() == "INCOMPLETE":
        if not close_window:
            datatype_var = 'Inspections'
            tab_var = 'load'
            create_popup(LoadErrorPopUp).run()
            return
        else:
            return
    else:
        data = insp_dataframe
        if insp_tab.cleaned_var.get() == "COMPLETE":
            collection = insp_prep_collection
        else:
            collection = insp_og_collection
    try:
        collection.delete_many({})
    finally:
        data.to_json(json_path, orient="records", double_precision=0,
                     force_ascii=True, default_handler=None, indent=4)
        with open(json_path) as json_file:
            file_data = json.load(json_file)

        collection.insert_many(file_data)
        print("Inspections Data saved to MongoDB database")
        return


def save_violations():
    global vio_dataframe, vio_og_collection, vio_prep_collection, datatype_var, tab_var
    json_path = 'ViolationsDatabase.json'
    if vio_tab.loaded_var.get() == "INCOMPLETE":
        if not close_window:
            datatype_var = 'Violations'
            tab_var = 'load'
            create_popup(LoadErrorPopUp).run()
            return
        else:
            return
    else:
        data = vio_dataframe
        if vio_tab.cleaned_var.get() == "COMPLETE":
            collection = vio_prep_collection
        else:
            collection = vio_og_collection
    try:
        collection.delete_many({})
    finally:
        data.to_json(json_path, orient="records", double_precision=0,
                     force_ascii=True, default_handler=None, indent=4)
        with open(json_path) as json_file:
            file_data = json.load(json_file)

        collection.insert_many(file_data)
        print("Violations Data saved to MongoDB database")
        return


# CLOSING THE WINDOW
# Saves current state of data into the MongoDB database when Tkinter master window is closed.
def on_closing():
    global close_window
    if messagebox.askokcancel("Quit",
                              "Do you want to quit? "
                              "Current state of data will automatically be saved in the MongoDB database."):
        close_window = True
        print('Saving data to MongoDB...')
        save_inventory()
        save_inspections()
        save_violations()
        print("Exit.")
        root.destroy()


# MASTER WINDOW
class MasterWindow:
    def __init__(self, master):
        self.master = master

        master.title("Data Analysis Tool")
        master.geometry('{}x{}'.format(1440, 900))
        master.grid_propagate(False)


# HIGH-LEVEL TABS
class UpperTabs:
    def __init__(self, parent):
        self.parent = parent
        self.notebook = ttk.Notebook(self.parent)

        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Original Data")

        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="Inspection Score Analysis")

        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Violation Type Analysis")
        self.tab3.rowconfigure(0, minsize=50)
        self.tab3.rowconfigure(1, weight=1)

        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="Violations Per Zip Code Visualisation")
        self.tab4.rowconfigure(0, minsize=50)
        self.tab4.rowconfigure(1, weight=1)

        self.notebook.pack(expand=1, fill="both")


# TAB 1 - ORIGINAL DATA
class Tab1Tabs:
    def __init__(self, parent):
        self.parent = parent
        self.notebook = ttk.Notebook(self.parent)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab1.rowconfigure(1, weight=1)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab2.rowconfigure(1, weight=1)
        self.tab3 = ttk.Frame(self.notebook)
        self.tab3.rowconfigure(1, weight=1)

        self.notebook.add(self.tab1, text='Inventory Data')
        self.notebook.add(self.tab2, text='Inspection Data')
        self.notebook.add(self.tab3, text='Violations Data')

        self.notebook.pack(expand=1, fill="both")


class Tab1Frame:
    def __init__(self, parent, name):
        global title_var
        self.parent = parent
        self.name = name
        self.options_frame = ttk.LabelFrame(self.parent, text='Options', height=100, width=pd_width)
        self.options_frame.grid(column=0, row=0, padx=10, pady=5, sticky='new')
        self.options_frame.rowconfigure(0, weight=1)
        self.options_frame.rowconfigure(1, weight=1)
        self.options_frame.rowconfigure(2, weight=1)
        self.data_frame = ttk.LabelFrame(self.parent, text=name, height=pd_height, width=pd_width)
        self.data_frame.rowconfigure(0, weight=1)
        self.data_frame.columnconfigure(1, weight=1)
        self.data_frame.grid(column=0, row=1, padx=10, pady=5, sticky='n' + 's' + 'e' + 'w')
        self.notes_frame = ttk.LabelFrame(self.parent, text="Notes", height=100, width=pd_width)
        self.notes_frame.grid(column=0, row=2, padx=10, pady=5, sticky='sew')

        # status labels
        self.loaded_label = ttk.Label(self.options_frame, text="Loaded status: ").grid(column=5, row=0, padx=5, pady=5,
                                                                                       sticky='nw')
        self.loaded_var = StringVar()
        self.loaded_status = ttk.Label(self.options_frame, textvariable=self.loaded_var)
        self.loaded_status.grid(column=6, row=0, padx=5, pady=5, sticky='nw')
        self.loaded_var.set("INCOMPLETE")

        self.cleaned_label = ttk.Label(self.options_frame, text="Cleaned status:").grid(column=5, row=1, padx=5, pady=5,
                                                                                        sticky='nw')
        self.cleaned_var = StringVar()
        self.cleaned_status = ttk.Label(self.options_frame, textvariable=self.cleaned_var)
        self.cleaned_status.grid(column=6, row=1, padx=5, pady=5, sticky='nw')
        self.cleaned_var.set("INCOMPLETE")

        self.saved_label = ttk.Label(self.options_frame, text="Saved status:").grid(column=5, row=2, padx=5, pady=5,
                                                                                    sticky='nw')
        self.saved_var = StringVar()
        self.saved_status = ttk.Label(self.options_frame, textvariable=self.saved_var)
        self.saved_status.grid(column=6, row=2, padx=5, pady=5, sticky='nw')
        self.saved_var.set("INCOMPLETE")

        # buttons
        self.load_in_btn = ttk.Button(self.options_frame, text="Load initial data\n(CSV file)",
                                      command=lambda: [get_tabname(), create_popup(LoadInitialDataPopUp).run()],
                                      width=14)
        self.load_in_btn.grid(row=0, column=0, rowspan=2, padx=5, pady=10, sticky="nesw")

        self.clean_btn = ttk.Button(self.options_frame, text="Clean and prepare data",
                                    command=lambda: [self.clean_data_button()],
                                    width=16)
        self.clean_btn.grid(row=0, column=1, rowspan=2, padx=5, pady=10, sticky="nesw")

        self.save_btn = ttk.Button(self.options_frame, text="Save to database",
                                   command=lambda: [self.save_button()],
                                   width=14)  # TODO save to database (and when window is  closed)
        self.save_btn.grid(row=0, column=2, rowspan=2, padx=5, pady=10, sticky="nesw")

        self.options_frame.grid_columnconfigure(4, minsize=450)

        self.load_cl_btn = ttk.Button(self.options_frame, text="Load data\nfrom database",
                                      command=lambda: self.load_database_button(),
                                      width=14)
        self.load_cl_btn.grid(row=0, column=3, rowspan=2, padx=5, pady=10, sticky="nesw")

        # notes label
        self.not_set = ttk.Label(self.notes_frame,
                                 text="Please load data using the 'Load data' button in the options pane")
        self.not_set.grid(column=0, row=0, padx=10, pady=10, sticky='nw')

    def load_status_complete(self):
        self.not_set.grid_forget()
        self.loaded_var.set("COMPLETE")

    def cleaned_status_complete(self):
        self.not_set.grid_forget()
        self.cleaned_var.set("COMPLETE")

    def clean_data_button(self):
        global datatype_var, tab_var, data_cleaned
        if self.cleaned_var.get() == "COMPLETE":
            create_popup(AlreadyCleanedPopUp).run()
            return
        elif self.loaded_var.get() == "COMPLETE":
            prepare_data_button(self.data_frame)
            self.load_status_complete()
        else:
            datatype_var = data_tabs.notebook.tab(data_tabs.notebook.select(), "text").split(' ', 1)[0]
            tab_var = 'load'
            create_popup(LoadErrorPopUp).run()
            return

    # Get current tab number to decide whether to address inspection data, inventory data or violations data
    # Try and load from corresponding prepared data collection.
    # If empty, try and load from original CSV data collection.
    # If empty, show error as no data exists in the collection.
    # Translate found data to Pandas DataFrame and display in GUI
    def load_database_button(self):
        global inv_dataframe, insp_dataframe, vio_dataframe, inv_og_collection, inv_prep_collection, \
            insp_og_collection, insp_prep_collection, vio_og_collection, vio_prep_collection
        if data_tabs.notebook.index(data_tabs.notebook.select()) == 0:
            tab_name = inv_tab
            if inv_prep_collection.count_documents({}) > 0:
                inv_dataframe = pd.DataFrame(list(inv_prep_collection.find({}, {'_id': False})))
                inv_tab.loaded_var.set("COMPLETE")
                inv_tab.cleaned_var.set("COMPLETE")
                pt = Table(tab_name.data_frame, dataframe=inv_dataframe, height=500, width=500, showtoolbar=False,
                           showstatusbar=True)
            elif inv_og_collection.count_documents({}) > 0:
                inv_dataframe = pd.DataFrame(list(inv_og_collection.find({}, {'_id': False})))
                inv_tab.loaded_var.set("COMPLETE")
                pt = Table(tab_name.data_frame, dataframe=inv_dataframe, height=500, width=500, showtoolbar=False,
                           showstatusbar=True)
            else:
                create_popup(DatabaseEmptyPopUp).run()
                return
        elif data_tabs.notebook.index(data_tabs.notebook.select()) == 1:
            tab_name = insp_tab
            if insp_prep_collection.count_documents({}) > 0:
                insp_dataframe = pd.DataFrame(list(insp_prep_collection.find({}, {'_id': False})))
                insp_tab.loaded_var.set("COMPLETE")
                insp_tab.cleaned_var.set("COMPLETE")
                pt = Table(tab_name.data_frame, dataframe=insp_dataframe, height=500, width=500, showtoolbar=False,
                           showstatusbar=True)
            elif insp_og_collection.count_documents({}) > 0:
                insp_dataframe = pd.DataFrame(list(insp_og_collection.find()))
                insp_tab.loaded_var.set("COMPLETE")
                pt = Table(tab_name.data_frame, dataframe=insp_dataframe, height=500, width=500, showtoolbar=False,
                           showstatusbar=True)
            else:
                create_popup(DatabaseEmptyPopUp).run()
                return
        else:
            tab_name = vio_tab
            if vio_prep_collection.count_documents({}) > 0:
                vio_dataframe = pd.DataFrame(list(vio_prep_collection.find({}, {'_id': False})))
                vio_tab.loaded_var.set("COMPLETE")
                vio_tab.cleaned_var.set("COMPLETE")
                pt = Table(tab_name.data_frame, dataframe=vio_dataframe, height=500, width=500, showtoolbar=False,
                           showstatusbar=True)
            elif vio_og_collection.count_documents({}) > 0:
                vio_dataframe = pd.DataFrame(list(vio_og_collection.find()))
                vio_tab.loaded_var.set("COMPLETE")
                pt = Table(tab_name.data_frame, dataframe=vio_dataframe, height=500, width=500, showtoolbar=False,
                           showstatusbar=True)
            else:
                create_popup(DatabaseEmptyPopUp).run()
                return

        pt.grid(column=0, row=1, sticky='new')
        pt.show()

    def save_button(self):
        self.saved_var.set("COMPLETE")
        if data_tabs.notebook.index(data_tabs.notebook.select()) == 0:
            save_inventory()
        elif data_tabs.notebook.index(data_tabs.notebook.select()) == 1:
            save_inspections()
        else:
            save_violations()


# TAB 2 - MEAN, MEDIAN AND MODE OF INSPECTIONS DATA
class Tab2Tabs:
    def __init__(self, parent):
        self.parent = parent
        self.notebook = ttk.Notebook(self.parent)

        self.tab1 = ttk.Frame(self.notebook)
        self.tab1.rowconfigure(1, weight=1)
        self.tab2 = ttk.Frame(self.notebook)
        self.tab2.rowconfigure(1, weight=1)

        self.notebook.add(self.tab1, text='Type of Seating')
        self.notebook.add(self.tab2, text='Zip Codes')

        self.notebook.pack(expand=1, fill="both")


class Tab2Frame:
    def __init__(self, parent, name):
        self.parent = parent
        self.name = name
        # self.column = column
        # self.data_in = None
        # self.mean = None

        self.options_frame = ttk.LabelFrame(self.parent, text='Options', height=100, width=pd_width)
        self.options_frame.rowconfigure(0, weight=1)
        self.options_frame.rowconfigure(1, weight=1)
        self.options_frame.rowconfigure(2, weight=1)
        self.options_frame.grid(column=0, row=0, padx=10, pady=5, sticky='new')
        self.data_frame = ttk.LabelFrame(self.parent, text=name, height=pd_height, width=pd_width)
        self.data_frame.rowconfigure(0, weight=1)
        self.data_frame.columnconfigure(1, weight=1)
        self.data_frame.grid(column=0, row=1, padx=10, pady=5, sticky='n' + 's' + 'e' + 'w')
        self.notes_frame = ttk.LabelFrame(self.parent, text="Notes", height=100, width=pd_width).grid(column=0, row=2,
                                                                                                      padx=10, pady=5,
                                                                                                      sticky='sew')

        # buttons
        self.mean_btn = ttk.Button(self.options_frame, text="MEAN\ninspection score per year",
                                   command=lambda: get_mean_button(),
                                   width=24)
        self.mean_btn.grid(row=0, column=0, rowspan=2, padx=5, pady=10, sticky="nesw")

        self.median_btn = ttk.Button(self.options_frame, text="MEDIAN\ninspection score per year",
                                     command=lambda: get_med_button(),
                                     width=24)
        self.median_btn.grid(row=0, column=1, rowspan=2, padx=5, pady=10, sticky="nesw")

        self.mode_btn = ttk.Button(self.options_frame, text="MODE\ninspection score per year",
                                   command=lambda: get_mode_button(),
                                   width=24)
        self.mode_btn.grid(row=0, column=2, rowspan=2, padx=5, pady=10, sticky="nesw")


# TAB 3 - BAR GRAPH OF VIOLATIONS DATA
class Tab3Frame:
    def __init__(self, parent):
        self.parent = parent

        self.options_frame = ttk.LabelFrame(self.parent, text='Options', height=100, width=pd_width)
        self.options_frame.rowconfigure(0, weight=1)
        self.options_frame.rowconfigure(1, weight=1)
        self.options_frame.rowconfigure(2, weight=1)
        self.options_frame.grid(column=0, row=0, padx=10, pady=5, sticky='new')
        self.data_frame = ttk.LabelFrame(self.parent, text="Violations Analysis", height=pd_height,
                                         width=pd_width)
        self.data_frame.rowconfigure(0, weight=1)
        self.data_frame.columnconfigure(1, weight=1)
        self.data_frame.grid(column=0, row=1, padx=10, pady=5, sticky='n' + 's' + 'e' + 'w')

        self.notes_frame = ttk.LabelFrame(self.parent, text="Notes", height=50, width=pd_width)
        self.notes_frame.grid(column=0, row=2, padx=10, pady=5, sticky='sew')

        # notes_label = ttk.Label(self.notes_frame, text=)

        show_btn = ttk.Button(self.options_frame, text="Show graph",
                              command=lambda: create_bar_canvas(),
                              width=30)
        show_btn.grid(row=0, column=0, rowspan=2, padx=5, pady=10, sticky="nesw")

        key_btn = ttk.Button(self.options_frame, text="Violations descriptions key",
                             command=lambda: create_vio_key_popup(),
                             width=30)
        key_btn.grid(row=0, column=1, rowspan=2, padx=5, pady=10, sticky="nesw")


# TAB 4 - SCATTER PLOT OF INSPECTIONS AND VIOLATIONS DATA
class Tab4Frame:
    def __init__(self, parent):
        global corr_graph, all_data
        self.parent = parent

        self.options_frame = ttk.LabelFrame(self.parent, text='Options', height=100, width=pd_width)
        self.options_frame.rowconfigure(1, weight=1)
        # self.options_frame.grid_columnconfigure(3, minsize=350)
        self.options_frame.grid(column=0, row=0, padx=10, pady=5, sticky='new')
        self.data_frame = ttk.LabelFrame(self.parent, text="Correlation Analysis", height=pd_height,
                                         width=pd_width)
        self.data_frame.rowconfigure(0, weight=1)
        self.data_frame.columnconfigure(1, weight=1)
        self.data_frame.grid(column=0, row=1, padx=10, pady=5, sticky='n' + 's' + 'e' + 'w')
        self.notes_frame = ttk.LabelFrame(self.parent, text="Notes", height=60, width=pd_width)
        self.notes_frame.grid(column=0, row=2, padx=10, pady=5, sticky='sew')

        show_btn = ttk.Button(self.options_frame, text="Show graph (all cities)",
                              command=lambda: [all_scatter_button(), self.show_annotation()],
                              width=30)
        show_btn.grid(row=0, column=0, rowspan=2, padx=5, pady=10, sticky="nesw")

        key_btn = ttk.Button(self.options_frame, text="City key",
                             command=lambda: create_city_key_popup(),
                             width=30)
        key_btn.grid(row=0, column=2, rowspan=2, padx=5, pady=10, sticky="nesw")

        range_btn = ttk.Button(self.options_frame, text="Select range",
                               command=lambda: [range_scatter_button()],
                               width=30)
        range_btn.grid(row=0, column=1, rowspan=2, padx=5, pady=10, sticky="nesw")

        self.max_city_var = StringVar()
        self.max_var = StringVar()
        self.min_city_var = StringVar()
        self.min_var = StringVar()
        self.corr_label = None
        self.max_string = ""
        self.max_mean = ""
        self.min_string = ""
        self.min_mean = ""

    def show_annotation(self):
        try:
            self.corr_label.grid_forget()
        except AttributeError:
            pass
        finally:
            # Use returned variables from the set_analysis_note() function and output in label
            self.max_string, self.max_mean, self.min_string, self.min_mean = set_analysis_note()

            self.max_city_var.set(self.max_string)
            self.max_var.set(self.max_mean)

            self.min_city_var.set(self.min_string)
            self.min_var.set(self.min_mean)

            self.corr_label = ttk.Label(self.notes_frame,
                                        text=f"On average, vendors in {self.max_city_var.get()} committed "
                                             f"{self.max_var.get()} violations. It is "
                                             f"therefore appropriate to assume that vendors with this city's zip code "
                                             f"are more likely to commit violations.\nLikewise, vendors in "
                                             f"{self.min_city_var.get()} committed {self.min_var.get()} violations on "
                                             f"average, suggesting that vendors with this city's zip code tend "
                                             f"to commit less violations.", style='annot.TLabel',
                                        wraplength=pd_width)
            self.corr_label.grid(column=0, row=0, padx=5, pady=10, sticky='e')


# POP UPS
class LoadInitialDataPopUp:
    def __init__(self, parent):
        global title_var, dir
        self.parent = parent
        self.csv_filename = ""
        self.json_filename = ""
        self.inv_json_filename = 'INVENTORY.json'
        self.insp_json_filename = 'INSPECTIONS.json'
        self.vio_json_filename = 'VIOLATIONS.json'
        self.filename = ""

        parent.wm_title("LOAD DATA")
        parent.geometry('{}x{}'.format(400, 300))
        parent.columnconfigure(0, weight=1)

        self.label1 = ttk.Label(self.parent, textvariable=title_var)  # entry box label
        self.label1.grid(column=0, row=0, padx=10, pady=5, sticky='w' + 'e')
        self.label2 = ttk.Label(self.parent, text="Enter your CSV file name: ")  # entry box label
        self.label2.grid(column=0, row=1, padx=10, pady=10, sticky='w' + 'e')

        self.name_var = StringVar()  # entry box
        self.name_var.set("")
        self.entry = ttk.Entry(self.parent, textvariable=self.name_var, width=5)
        self.entry.grid(column=0, row=2, padx=10, pady=10, sticky='w' + 'e')

        self.error_label = ttk.Label(self.parent,
                                     text=f"Error - not a valid file name. "
                                          f"\nMake sure you have typed in the name correctly  and that the file "
                                          f"is saved in the following directory: \n{dir}",
                                     foreground='red', wraplength=350, justify=LEFT)

        self.okay_button = ttk.Button(self.parent, text="Load",
                                      command=lambda: self.check_csv_name())  # load button
        self.okay_button.grid(column=0, row=4, padx=10, pady=10, sticky='e')

    # Get string variable from user input
    def get_csv_name(self):

        self.filename = str(self.name_var.get())
        if self.filename.endswith('.csv'):
            return self.filename
        else:
            return f'{self.filename}.csv'

    # Check if inputted filename is in current directory
    def check_csv_name(self):
        if os.path.isfile(f'{os.getcwd()}/{self.get_csv_name()}'):
            try:
                self.error_label.grid_forget()
            finally:
                self.load_files()
        else:
            self.error_label.grid(column=0, row=3, padx=10, pady=10, sticky='w')

    def load_files(self):
        global inv_dataframe, insp_dataframe, vio_dataframe
        tab_number = data_tabs.notebook.index(data_tabs.notebook.select())
        if tab_number == 0:  # Inventory
            inv_tab.load_status_complete()  # Change loaded status variable
            inv_dataframe = create_dataframe(self.get_csv_name(), self.inv_json_filename) # Create dataframe
            print("Tab number = " + str(tab_number))
            print("Inventory Dataframe")
            print(inv_dataframe.head())
            frame = inv_tab.data_frame
            pt = Table(frame, dataframe=inv_dataframe, height=pd_height, width=pd_width, showtoolbar=False,
                       showstatusbar=True)  # Display as pandastable
            pt.grid(column=0, row=0, sticky='new')
            pt.show()
        elif tab_number == 1:  # Inspections
            insp_tab.load_status_complete()  # Change loaded status variable
            insp_dataframe = create_dataframe(self.get_csv_name(), self.insp_json_filename)  # Create dataframe
            print("Tab number = " + str(tab_number))
            print("Inspections Dataframe")
            print(insp_dataframe.head())
            frame = insp_tab.data_frame
            pt = Table(frame, dataframe=insp_dataframe, height=pd_height, width=pd_width, showtoolbar=False,
                       showstatusbar=True)  # Display as pandastable
            pt.grid(column=0, row=0, sticky='new')
            pt.show()
        else:  # Violations
            vio_tab.load_status_complete()  # Change loaded status variable
            vio_dataframe = create_dataframe(self.get_csv_name(), self.vio_json_filename)  # Create dataframe
            print("Tab number = " + str(tab_number))
            print("Violations Dataframe")
            print(vio_dataframe.head())
            frame = vio_tab.data_frame
            pt = Table(frame, dataframe=vio_dataframe, height=pd_height, width=pd_width, showtoolbar=False,
                       showstatusbar=True)  # Display as pandastable
            pt.grid(column=0, row=0, sticky='new')
            pt.show()
        self.parent.destroy()

    def run(self):
        self.parent.lift()
        self.parent.mainloop()


# Key for the violations bar graph
class VioKeyPopUp:
    def __init__(self, parent):
        self.parent = parent

        parent.wm_title("Violations Key")
        parent.geometry('{}x{}'.format(500, 500))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        label1 = ttk.Label(self.parent, text="Violations code key:")  # label
        label1.grid(column=0, row=0, padx=10, pady=5, sticky='nw')

        frame = ttk.Frame(self.parent, height=500, width=500)
        frame.grid(column=0, row=1, padx=10, pady=5, sticky='new')

        pt = Table(frame, dataframe=violations_key(), height=500, width=500, showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=1, sticky='new')
        pt.show()

        close_button = ttk.Button(self.parent, text="Close",
                                  command=lambda: self.parent.destroy())  # load button
        close_button.grid(column=0, row=2, padx=10, pady=10, sticky='w')

    def run(self):
        self.parent.lift()
        self.parent.mainloop()


# Key for the correlation scatter graph
class CorrKeyPopUp:
    def __init__(self, parent):
        self.parent = parent

        parent.wm_title("City Key")
        parent.geometry('{}x{}'.format(500, 500))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        label1 = ttk.Label(self.parent, text="City key:")  # label
        label1.grid(column=0, row=0, padx=10, pady=5, sticky='nw')

        frame = Frame(self.parent, height=500, width=500)
        frame.grid(column=0, row=1, padx=10, pady=5, sticky='new')

        pt = Table(frame, dataframe=city_key(), height=500, width=500, showtoolbar=False,
                   showstatusbar=True)
        pt.grid(column=0, row=1, sticky='new')
        pt.show()

        close_button = ttk.Button(self.parent, text="Close",
                                  command=lambda: self.parent.destroy())  # load button
        close_button.grid(column=0, row=2, padx=10, pady=10, sticky='w')

    def run(self):
        self.parent.lift()
        self.parent.mainloop()

# Data not loaded sufficiently for action
class LoadErrorPopUp:
    def __init__(self, parent):
        global tab_var, datatype_var
        self.parent = parent

        parent.wm_title("Error")
        parent.geometry('{}x{}'.format(420, 250))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        # Variables to display which data it is that need to be actioned and which actions are required to continue
        status_var = StringVar()
        status_var.set(tab_var)
        data_var = StringVar()
        data_var.set(datatype_var)

        label1 = ttk.Label(self.parent,
                           text=f"ERROR\n\nPlease {status_var.get()} your {data_var.get()} data first...")  # label
        label1.grid(column=0, row=0, padx=10, pady=5, sticky='nw')

        close_button = ttk.Button(self.parent, text="OK",
                                  command=lambda: self.parent.destroy())  # load button
        close_button.grid(column=0, row=1, padx=10, pady=10, sticky='e')

    def run(self):
        self.parent.lift()
        self.parent.mainloop()

# Data has already been cleaned and prepared
class AlreadyCleanedPopUp:
    def __init__(self, parent):
        self.parent = parent

        parent.wm_title("Error")
        parent.geometry('{}x{}'.format(420, 250))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        label1 = ttk.Label(self.parent,
                           text=f"Data has already been cleaned and prepared.")  # label
        label1.grid(column=0, row=0, padx=10, pady=5, sticky='nw')

        close_button = ttk.Button(self.parent, text="OK",
                                  command=lambda: self.parent.destroy())  # load button
        close_button.grid(column=0, row=1, padx=10, pady=10, sticky='e')

    def run(self):
        self.parent.lift()
        self.parent.mainloop()

# Database has no items in collection
class DatabaseEmptyPopUp:
    def __init__(self, parent):
        self.parent = parent

        parent.wm_title("Error")
        parent.geometry('{}x{}'.format(440, 230))
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(0, weight=1)

        label1 = ttk.Label(self.parent,
                           text=f"No data has been saved in the database.\nLoad/prepare/save your data.")  # label
        label1.grid(column=0, row=0, padx=10, pady=5, sticky='nw')

        close_button = ttk.Button(self.parent, text="OK",
                                  command=lambda: self.parent.destroy())  # load button
        close_button.grid(column=0, row=1, padx=10, pady=10, sticky='e')

    def run(self):
        self.parent.lift()
        self.parent.mainloop()

# Select range for correlation graph
class GraphRangePopUp:
    def __init__(self, parent):
        global lower_var, upper_var

        self.parent = parent

        parent.wm_title("Select data range")
        parent.geometry('{}x{}'.format(370, 280))
        parent.columnconfigure(0, weight=1)
        parent.columnconfigure(1, weight=1)
        parent.grid_rowconfigure(3, minsize=40)

        self.max_mean = int(corr_graph_dataframe()["MEAN"].max()) + 1

        self.title_label = ttk.Label(parent,
                                     text="Show cities where number of violations is between:")  # label
        self.title_label.grid(column=0, row=0, columnspan=2, padx=10, pady=5, sticky='e' + 'w')

        self.lower_label = ttk.Label(parent,
                                     text="Lower bound:")  # label
        self.lower_label.grid(column=0, row=1, columnspan=1, padx=10, pady=5, sticky='e' + 'w')

        # self.lower_var = IntVar()  # entry box
        self.lower_entry = ttk.Entry(parent, textvariable=lower_var, width=2)
        self.lower_entry.grid(column=1, row=1, columnspan=1, padx=10, pady=5, sticky='e' + 'w')

        # Dynamic maximum value to account for different data sets
        self.upper_label = ttk.Label(parent,
                                     text=f"Upper bound (maximum {self.max_mean}):")  # label
        self.upper_label.grid(column=0, row=2, columnspan=1, padx=10, pady=5, sticky='e' + 'w')

        self.upper_entry = ttk.Entry(parent, textvariable=upper_var, width=2)
        self.upper_entry.grid(column=1, row=2, columnspan=1, padx=10, pady=5, sticky='e' + 'w')

        self.max_mean = int(corr_graph_dataframe()["MEAN"].max()) + 1

        self.error1_label = Label(self.parent,
                                  text=f"Error: input not in range. \nRe-enter values (between 0 "
                                       f"and {self.max_mean})",
                                  fg='red', wraplength=350, justify=LEFT)
        self.error2_label = Label(self.parent,
                                  text=f"\nError: make sure your upper bound is greater than your lower bound",
                                  fg='red', wraplength=350, justify=LEFT)
        self.error3_label = Label(self.parent,
                                  text=f"\nError: please enter a number",
                                  fg='red', wraplength=350, justify=LEFT)

        self.close_button = ttk.Button(parent, text="OK",
                                       command=lambda: self.check_range())  # load button
        self.close_button.grid(column=1, row=4, columnspan=1, padx=10, pady=10, sticky='e')

    def run(self):
        self.parent.lift()
        self.parent.mainloop()

    # Check if values are within the data's range, that the lower bound is lesser
    # than the upper bound and that the input is a number
    def check_range(self):
        try:
            if get_a() > get_b():
                try:
                    self.error1_label.grid_forget()
                    self.error2_label.grid_forget()
                    self.error3_label.grid_forget()
                finally:
                    self.error2_label.grid(column=0, row=3, columnspan=2, padx=10, pady=5, sticky='e' + 'w')
                    return
            elif get_a() < 0 or get_b() > self.max_mean:
                try:
                    self.error1_label.grid_forget()
                    self.error2_label.grid_forget()
                    self.error3_label.grid_forget()
                finally:
                    self.error1_label.grid(column=0, row=3, columnspan=2, padx=10, pady=5, sticky='e' + 'w')
                    return
        except ValueError:
            try:
                self.error1_label.grid_forget()
                self.error2_label.grid_forget()
                self.error3_label.grid_forget()
            finally:
                self.error3_label.grid(column=0, row=3, columnspan=2, padx=10, pady=5, sticky='e' + 'w')
                return
        else:
            update_scatter_graph(get_a(), get_b())
            self.parent.destroy()
            return


if __name__ == '__main__':
    inv_dataframe, insp_dataframe, vio_dataframe, corr_graph = None, None, None, None
    fig, ax, canvas = None, None, None
    inv_og_collection, insp_og_collection, vio_og_collection = None, None, None
    inv_prep_collection, insp_prep_collection, vio_prep_collection = None, None, None

    all_data = False
    close_window = False
    data_cleaned = False

    pd_width = 1325
    pd_height = 480

    dir = os.getcwd() # Get the current working directory in which data sets must be saved
    tab_var = ""
    datatype_var = ""
    radio_num = -999 # Dummy value

    open_collections()
    set_dataframe_options()

    root = tk.Tk()

    # Ttk style conifigurations
    style = ttk.Style()

    style.theme_create('dataAnalysisProgram', settings={
        ".": {
            "configure": {
                "background": '#f6f9f9',  # All except tabs
                "font": 'black'
            }
        },
        "TNotebook": {
            "configure": {
                "background": '#e4f2f2',  # Your margin color
                "tabmargins": [10, 3, 10, 0],  # margins: left, top, right, separator
            }
        },
        "TNotebook.Tab": {
            "configure": {
                "background": '#9dd5d4',  # tab color when not selected
                "padding": [20, 4],
                "font": "white"
            },
            "map": {
                "background": [("selected", '#fff5c3')],  # Tab color when selected
                "expand": [("selected", [2, 2, 2, 0])]  # text margins
            }
        },
        "TButton": {
            "configure": {
                "background": '#3a8786',
                "foreground": "white",
                "padding": 6,
                "relief": "raised",
                "anchor": "center",
                'focuscolor': '#fff5c3'
            }
        },
        "TLabelframe": {
            "configure": {
                "relief": "groove",
            }
        },
        "TLabel": {
            "configure": {
                "background": '#f6f9f9',
                "anchor": "left"
            }
        }

    })

    style.theme_use('dataAnalysisProgram')
    style.configure('annot.TLabel', font=8)

    master_window = MasterWindow(root)

    main_tabs = UpperTabs(root)
    title_var = StringVar()
    upper_var = StringVar()
    lower_var = StringVar()
    upper_var.set("")  # dummy value
    lower_var.set("")  # dummy value

    data_tabs = Tab1Tabs(main_tabs.tab1)
    inv_tab = Tab1Frame(data_tabs.tab1, "Inventory Data")
    insp_tab = Tab1Frame(data_tabs.tab2, "Inspections Data")
    vio_tab = Tab1Frame(data_tabs.tab3, "Violations Data")

    mmm_tabs = Tab2Tabs(main_tabs.tab2)
    seating_tab = Tab2Frame(mmm_tabs.tab1, "Type of Seating Analysis")
    zip_tab = Tab2Frame(mmm_tabs.tab2, "Zip Codes Analysis")

    vioanalysis_tab = Tab3Frame(main_tabs.tab3)

    corranalysis_tab = Tab4Frame(main_tabs.tab4)

    # Call the on_closing() function when the main Tkinter window is closed
    root.protocol("WM_DELETE_WINDOW", on_closing)

    # Run the mainloop
    root.mainloop()
