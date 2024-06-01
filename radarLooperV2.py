# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 21:29:05 2023

This script was written by Mark De Bruin (UNL master's student). If you are going
to post a custom loop to twitter or elsewhere I would appreciate crediting, a lot
of hard work went into creating all of this. Not a ton of testing has been done, so
feel free to report back any bugs to me. Just send me the initial conditions
you entered.

Near the bottom is a manual data entry section to customize the loop.
"""

# Ensure that all packages are installed to your python environment before running.
# Many of these are not pre-installed to the environment.
import os
import pandas as pd
import time
import datetime
import warnings
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import numpy as np
import boto3
import botocore
from botocore.client import Config
from metpy.plots import add_timestamp, USCOUNTIES, USSTATES
from sklearn.neighbors import KDTree
import cv2
import re
from PIL import Image
import pyart
import nexradaws


# Some weird function I made to make sure every part of the time range will get downloaded        
def hourRound(iterTime):
    iterTimestamp = pd.Timestamp(iterTime)
    rounded = iterTimestamp.round('60min').to_pydatetime()
    if rounded <= iterTime:
        iterTime = rounded + datetime.timedelta(hours=1)
        return iterTime
    else:
        return rounded
    
# Function manages the available scan times in the AWS archive.
# Returns a dictionary with datetime keys on 1 hour intervals and values are
# the available minutes for that hour.
def minuteList(startTime, endTime, ID):
    roundEnd = hourRound(endTime)
    roundEnd = roundEnd - datetime.timedelta(hours=1)
    iterTime = startTime.replace(minute=0)
    finalList = {}
    finalList[iterTime] = []
    
    while iterTime != roundEnd:
        iterTime = iterTime + datetime.timedelta(hours=1)
        finalList[iterTime] = []
    
    for iterTime in list(finalList.keys()):
        
        # Access the archive and find available keys for the given prefix
        prefix = f'{iterTime:%Y}/{iterTime:%m}/{iterTime:%d}/{ID}/{ID}{iterTime:%Y%m%d_%H}'
        objects = []
        for obj in bucket.objects.filter(Prefix=prefix):
            objects.append(obj)
        
        for filetime in objects:
            if len(filetime.key) != 43:
                finalList[iterTime].append(int(filetime.key[31:33]))
            elif len(filetime.key) != 43 and len(filetime.key) != 39:
                print('BEWARE: New file type encountered in archive. Let chaos ensue!')
            
    # Just to take out any empty keys in the dictionary
    for key in list(finalList.keys()):
        if len(finalList[key]) == 0:
            del finalList[key]
    
    # Annoying mess of loops to prepare the final dictionary
    prevdif = 61
    minute = startTime.minute
    hour1list = list(finalList.values())[0]
    for goesmin in hour1list:
        dif = abs(minute-goesmin)
        if dif < prevdif:
            closestmin = goesmin
            prevdif = dif
    minIndexStart = hour1list.index(closestmin)
    hour1list = hour1list[minIndexStart::]
    hour1key = list(finalList.keys())[0]
    finalList[hour1key] = hour1list
    
    prevdif = 61
    hourElist = list(finalList.values())[-1]
    minute = endTime.minute
    for goesmin in hourElist:
            dif = abs(minute-goesmin)
            if dif < prevdif:
                closestmin = goesmin
                prevdif = dif
    minIndexEnd = hourElist.index(closestmin)
    hourElist = hourElist[:minIndexEnd:]
    hourEkey = list(finalList.keys())[-1]
    finalList[hourEkey] = hourElist
    
    return finalList

# Given a radar scan file, return a list of indices that correspond to all
# low level scans.
def sweepFinder(radar):
    els = [np.mean(radar.get_elevation(i)) for i in range(0,radar.nsweeps)]

    keepIdx = []
    good = False
    angle = 0.6
    while not good:
        prevIdx = -2
        for idx, tilt in enumerate(els):
            if tilt < angle and idx != prevIdx+1:
                keepIdx.append(idx)
                prevIdx = idx
        if len(keepIdx) > 0:
            good=True
        # In rare cases, lowest angle is >0.6
        angle+=0.1
    return keepIdx

# Downloads the radar scan file given a datetime and radar ID
def getRadar(startTime, ID, writePath):
    endTime = startTime + datetime.timedelta(minutes=1)
    scans = conn.get_avail_scans_in_range(startTime, endTime, ID)
    conn.download(scans[0], writePath)
    radar = pyart.io.read(os.path.join(writePath, str(scans[0])[40:63]))
    return radar

# Function does two main things. First it takes the radar file and converts it
# to lat/lon coordinates. Then it will plot it with the proper center point/bounds.
# Much of this was taken from metpy sample code.
def radarCoords(radar, lati, long, degSize, refPlot, sweep):
    
    if not refPlot:
        radarSwp = radar.extract_sweeps([sweep+1])
        display = pyart.graph.RadarDisplay(radar)
        nyq = radar.fields['velocity']['data'].max()
        velocity_dealiased = pyart.correct.dealias_region_based(radarSwp, vel_field='velocity',
                                                                nyquist_vel=nyq, centered=True)
        radarSwp.add_field('corrected_VEL', velocity_dealiased, replace_existing=True)
        velocity = radarSwp.fields['corrected_VEL']['data']
    else:
        radarSwp = radar.extract_sweeps([sweep])
        display = pyart.graph.RadarDisplay(radar)
        reflectivity = radarSwp.fields['reflectivity']['data']
    
    rlons = radarSwp.gate_longitude['data']
    rlats = radarSwp.gate_latitude['data']
    
    # Plot the data
    scanTime = datetime.datetime.strptime(display.generate_title('reflectivity',
                                                                 sweep=sweep)[14:33], '%Y-%m-%dT%H:%M:%S')
    fig = plt.figure(figsize=(12, 9), dpi=150, frameon=False)
    
    # Plot the data
    crs = ccrs.LambertConformal(central_longitude=long, central_latitude=lati)
    ax = fig.add_subplot(projection=crs)
    ax.set_facecolor('black')
    
    if refPlot:
        map1 = ax.pcolormesh(rlons, rlats, reflectivity, cmap='pyart_ChaseSpectral',
                             vmin=-10, vmax=75, transform=ccrs.PlateCarree(), zorder=3)
    else:
        map1 = ax.pcolormesh(rlons, rlats, velocity, vmin=-60, vmax=60, cmap='pyart_Carbone42',
                         transform=ccrs.PlateCarree(), zorder=3)
    ax.add_feature(USCOUNTIES, linewidth=0.5, edgecolor='white', zorder=2)
    ax.add_feature(USSTATES, linewidth=3, edgecolor='white', zorder=1)
    ax.set_extent([long - degSize, long + degSize, lati - degSize, lati + degSize])
    ax.set_aspect('equal', 'datalim')
    add_timestamp(ax, scanTime, y=0.02, high_contrast=True, fontsize='large', zorder=4)
    # Please leave my watermark thanks :)
    ax.annotate('Via Mark De Bruin, AWS, NOAA', xy=(0.02,0.96), xycoords='axes fraction',
                fontsize='x-large', color='white')
    
    plt.show()
    return fig

# Here's a fun one. If you hadn't noticed, there's a required text file to put in
# the CWD that has WSR88D locations. This function uses those coordinates to create
# a KD tree. The KDtree is used to find the nearest or 2nd nearest radar to the
# given lat/lon points (the present frame center basically)
def nearID(locArray, K):
    path = os.path.join(os.getcwd(), '88Dcoords.txt')
    df = pd.read_csv(path, delim_whitespace=True, header=0,
                     on_bad_lines='warn', usecols=[2,5,6])
    df['LAT'] = [float(row['LAT'][0:4])/100 for index, row in df.iterrows()]
    df['LONG'] = [float(row['LONG'][0:5])/-100 for index, row in df.iterrows()]
    df = df.loc[df['ICAO'] != '-'].reset_index()
    
    kd = KDTree(df[['LAT', 'LONG']].values, metric='euclidean')
    distances, indices = kd.query(locArray, k=K)
    
    flattened = indices[:,K-1][:]
    sites = [df['ICAO'][i] for i in flattened]
    return sites

# This function is a solution to a complex problem. You need a list of coordinates
# before knowing which radar IDs to use. But, you don't know the exact scan times
# (and therefore the proper amount to pan in each frame) until you have the IDs.

# To solve this, a lat/lon pan rate per minute is calculated, and perMinCoords is
# a lat/lon image center point for each minute of the duration of the loop.
def findPanRate(coordList, startTime, endTime):
    elapsed = endTime - startTime
    minutes = int(elapsed.total_seconds() / 60)
    latRate = round((coordList[1][0] - coordList[0][0]) / minutes, 3)
    lonRate = round((coordList[1][1] - coordList[0][1]) / minutes, 3)
    perMinCoords = np.array([])
    for n in range(0,minutes):
        pair = np.array([coordList[0][0] + (latRate * n), coordList[0][1] + (lonRate * n)])
        if n!=0:
            perMinCoords = np.vstack((perMinCoords, pair))
        else:
            perMinCoords = np.hstack((perMinCoords, pair))
    return perMinCoords, latRate, lonRate

# Takes the mixed dictionary of datetimes and minutes to all datetimes
def timeList2dt(timeList):
    newList = []
    for hour in timeList.keys():
        if len(timeList[hour]) == 0:
            continue
        else:
            for minute in timeList[hour]:
                fullTime = hour.replace(minute=minute)
                newList.append(fullTime)
    return newList

# Needed for the QCing process. I don't remember how this works.            
def timelineCut(masterTimeDict, startTime, endTime, IDs, perMinCoords):
    # Creates a 2D np array with a radar ID and matching time to query
    TL = pd.DataFrame(columns=['Scan_Time', 'Radar_ID', 'Center_Lat', 'Center_Lon'])
    n = 0
    # This is the place to fix empty radar data time periods
    for ID in IDs:
        iterTime = startTime + datetime.timedelta(minutes=n)
        if iterTime in masterTimeDict[ID]:
            TL.loc[n] = [iterTime, ID, perMinCoords[n][0], perMinCoords[n][1]]
        n+=1
    return TL.reset_index()

# Also needed for QCing process, but slightly different. Contains information
# about the presence of a scan time in the minute-by-minute timeline.
def timelineFull(masterTimeDict, startTime, endTime, IDs, perMinCoords):
    # Creates a 2D np array with a radar ID and matching time to query
    TL = pd.DataFrame(columns=['Scan_Time', 'Radar_ID', 'Center_Lat', 'Center_Lon', 'Present'])
    n = 0
    # This is the place to fix empty radar data time periods
    for ID in IDs:
        iterTime = startTime + datetime.timedelta(minutes=n)
        if iterTime in masterTimeDict[ID]:
            TL.loc[n] = [iterTime, ID, perMinCoords[n][0], perMinCoords[n][1], True]
        
        else:
            TL.loc[n] = [iterTime, ID, perMinCoords[n][0], perMinCoords[n][1], False]
        n+=1
    return TL.reset_index()

# Returns only scans that contain empty data from a given radar. Radar ID will
# later be replaced by 2nd nearest ID and similar scan times.
def emptyData(TL, k):
    goodTL = pd.DataFrame(columns=['Scan_Time', 'Radar_ID', 'Center_Lat',
                                   'Center_Lon', 'Present'])
    emptyTL = pd.DataFrame(columns=['Scan_Time', 'Radar_ID', 'Center_Lat',
                                    'Center_Lon', 'Present'])
    for row, column in TL.iterrows():
        rowSlice = TL['Present'].loc[row:row+k]
        if rowSlice.any() and TL['Present'][row]==True:
            goodTL.loc[row] = [TL['Scan_Time'][row], TL['Radar_ID'][row],
                               TL['Center_Lat'][row], TL['Center_Lon'][row],
                               TL['Present'][row]]
        
        elif not rowSlice.any():
            emptyTL.loc[row] = [TL['Scan_Time'][row], TL['Radar_ID'][row],
                                TL['Center_Lat'][row], TL['Center_Lon'][row],
                                TL['Present'][row]]
        
    return goodTL, emptyTL

# I would love to fix the efficiency on this to use less for loops. It was
# surprisingly complex but maybe some sort of numpy function already exists for this.
# An ID must be present for >k minutes, and if not is overwritten with another
# ID in the list.
def toleranceFunc(IDs, k):
    IDprev = IDs[0]
    problems = []
    consec = 1
    # This identifies all IDs that do not conform to the radar switch tolerance
    for idx, ID in enumerate(IDs):
        if ID == IDprev:
            consec+=1
            IDprev=ID
            continue
        elif consec >= k:
            consec=1
            IDprev=ID
            continue
        else:
            if idx-consec < 0:
                problems = problems+(list(range(idx-consec+1, idx)))
            else:
                problems = problems+(list(range(idx-consec, idx)))
            consec = 1
            IDprev=ID
            continue
        
    # Now get the problems at the end
    if consec < k:
        problems = problems+(list(range(idx-consec, idx+1)))
    
    badList = list(set(problems))
    badList.sort()
    
    if len(badList)==0:
        return IDs

    # This determines which stretch of bad IDs each index is in
    prev = badList[0]
    stretchList = []
    stretch = 1
    for idx in badList:
        if idx-prev <= 1:
            stretchList.append(stretch)
        else:
            stretch+=1
            stretchList.append(stretch)
        prev = idx
    
    # Now fix the ID list using stretchList and badList
    newIDs = IDs
    for idx, ID in enumerate(badList):
        if ID < k:
            setnum = stretchList[idx]
            lastInd = len(stretchList) - 1 - stretchList[::-1].index(setnum)
            backFix = badList[lastInd] + 1#set switch idx of badList + 1
            newIDs[ID] = IDs[backFix]
        else:
            setnum = stretchList[idx]
            frontInd = stretchList.index(setnum)
            frontFix = badList[frontInd] - 1#set switch idx of badList + 1
            newIDs[ID] = IDs[frontFix]

    return newIDs

# Thanks chatGPT
def images_to_gif(input_dir, output_file, duration, loop=0):
    """
    Convert a directory of images into a GIF.
    Args:
    input_dir (str): Path to the directory containing input images.
    output_file (str): Path to the output GIF file.
    duration (int): Duration (in milliseconds) for each frame.
    loop (int): Number of iterations. 0 means infinite loop.
    """
    images = [img for img in os.listdir(input_dir) if img.endswith(".png")]
    images.sort(key=lambda f: int(re.sub('\D', '', f)))
    
    imagesGIF = []
    for filename in images[0:60]:
        filepath = os.path.join(input_dir, filename)
        imagesGIF.append(Image.open(filepath))

    if not imagesGIF:
        print("No images found in the directory.")
        return

    imagesGIF[0].save(output_file, save_all=True, append_images=imagesGIF[1:], duration=duration, loop=loop)
    print(f"GIF saved to {output_file}")

#------------------------------------------------------------------------------    
#------------------------------------------------------------------------------

# Manual Data Entry
# Sample events below
'''
# December 2021
startTime = datetime.datetime(2021,12,11,00,39)
endTime = datetime.datetime(2021,12,11,6,00)
#ID = 'KDMX'
startlat = 36.2
startlon=-90.5
endlat = 37.6
endlon = -86.5
interval = 5
degSize = 0.8
radarSwitch = True

# Manual Data Entry
startTime = datetime.datetime(2023,4,4,21,00)
endTime = datetime.datetime(2023,4,5,1,30)
startlat = 40.5
startlon=-94
endlat = 41.8
endlon = -92.3
degSize = 0.8

startTime = datetime.datetime(2023,3,31,18,30)
endTime = datetime.datetime(2023,3,31,23,45)
startlat = 40.1
startlon=-94.4
endlat = 41.7
endlon = -90.5
degSize = 1.2

startTime = datetime.datetime(2023,6,11,21,30)
endTime = datetime.datetime(2023,6,12,2,30)
startlat = 37
startlon=-103.5
endlat = 36.5
endlon = -102.5
degSize = 0.8

startTime = datetime.datetime(2023,6,12,21,00)
endTime = datetime.datetime(2023,6,13,2,00)
startlat = 36
startlon=-103.7
endlat = 35.4
endlon = -103.4
degSize = 0.8
'''
# 2020 Derecho

# Beginnging and end bounds of the radar loop. Change the datetime object in
# YYYY,M,D,H,M format.
startTime = datetime.datetime(2020,8,10,11,26)
endTime = datetime.datetime(2020,8,10,21,45)

# Start/end coordinates
startlat = 42.8
startlon=-96.8
endlat = 41.5
endlon = -87.8

# Dictates window size in degree units. Plot width/height will be 2x degSize.
degSize = 1.8

# Tolerance is a tool to eliminate brief radar switches to make the loop more
# smooth. Essentially for the radar ID to be queued, it must be the nearest radar
# for > k minutes. Else a surrounding ID is used instead.
tolerance = True
k=25

# Short for max low level scans. Depending on radar VCP, multiple low level sweeps
# can be contained within a single scan file. If on, all lowest level sweeps
# are plotted.
maxLL = True

# And finally radar switching. If large distances are covered, have this on.
# If the loop pans over a small area, leave off and only one radar will be used.
radarSwitch = True

# Product to plot. True will plot reflectivity, false will plot velocity.
refPlot = True

# Controls the output format. True will write a GIF and false will write an
# mp4.
GIF = True

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

warnings.filterwarnings('ignore')
start_time = time.time()

s3 = boto3.resource('s3', config=Config(signature_version=botocore.UNSIGNED,
                                        user_agent_extra='Resource'))
bucket = s3.Bucket('noaa-nexrad-level2')

conn = nexradaws.NexradAwsInterface()

def main():
    
    # Controls the temporary image directory that is created.
    # Will empty the directory before writing.
    tempPath = os.path.join(os.getcwd(), 'TempImages_Radar')
    writePath = os.path.join(os.getcwd(), 'TempFiles_WSR88D')
    if os.path.isdir(tempPath) == False:
        os.mkdir(tempPath)
    for file in os.listdir(tempPath):
        os.remove(os.path.join(tempPath, file))
    
    # Controls the temporary radar file directory that is created.
    # Will empty the directory before writing.    
    if os.path.isdir(writePath) == False:
        os.mkdir(writePath)
    for file in os.listdir(writePath):
        os.remove(os.path.join(writePath, file))
    
    coordList = [[startlat, startlon], [endlat, endlon]]
    perMinCoords, latRate, lonRate = findPanRate(coordList, startTime, endTime)
    
    # When radarSwitch is off, use the midpoint coordinates to determine the
    # radar ID.
    if not radarSwitch:
        midpntLat = (startlat+endlat)/2
        midpntLon = (startlon+endlon)/2
        locArray = [[midpntLat, midpntLon]]
        IDs = []
        IDs.append(nearID(locArray, K=1)[0])
        IDs = [IDs[0] for i in range(1, len(perMinCoords)+1)]
        
        try:
            timeList = minuteList(startTime, endTime, IDs[0])
        except:
            IDs = nearID(perMinCoords, K=2)
    
    else:
        IDs = nearID(perMinCoords, K=1)
        if tolerance:
            IDs = toleranceFunc(IDs, k)
    
        # If only one radar ID is queued, but radar data is empty, use the 2nd
        # nearest radar to prevent an error. Full timeList then gets created later
        
        # Note that if the 1st and 2nd nearest radar data is missing, your loop is
        # screwed. It's unlikely enough that I didn't feel like programming the 
        # fix for this.
        try:
            timeList = minuteList(startTime, endTime, IDs[0])
        except:
            IDs = nearID(perMinCoords, K=2)
    
    uniqIDs = list(set(IDs))
    
    # masterTimeDict is a record of all existing scan times for each needed
    # radar ID during the loop duration
    masterTimeDict = {}
    for site in uniqIDs:
        timeList = minuteList(startTime, endTime, site)
        timeList = timeList2dt(timeList)
        masterTimeDict[site] = timeList
    
    # TL is the timeline dataframe. This is printed before loop downloads
    # begin. This variable is the controller of what exactly gets downloaded.
    TL = timelineFull(masterTimeDict, startTime, endTime, IDs, perMinCoords)
    
    # QC procedure
    # Turns out partially missing data is a complex problem. Mayfield Kentucky
    # case for example. A certain time range of the full event was missing.
    # This part of the procedure replaces the parts of the timeline with missing
    # radar data with the 2nd nearest radar ID/times.
    TL, emptyTL = emptyData(TL, k)
    if len(emptyTL) > k:
        emptyCoords = np.array([])
        n = 0
        for row, column in emptyTL.iterrows():
            pair = np.array([emptyTL['Center_Lat'][row], emptyTL['Center_Lon'][row]])
            if n != 0:
                emptyCoords = np.vstack((emptyCoords, pair))
            else:
                emptyCoords = np.hstack((emptyCoords, pair))
            n+=1

        IDs  = nearID(emptyCoords, K=2)
        IDs = toleranceFunc(IDs, k)
        uniqIDs = list(set(IDs))
        masterTimeDict = {}
        emptyTL = emptyTL.reset_index()
        coordList = [[emptyTL['Center_Lat'][0], emptyTL['Center_Lon'][0]],
                     [emptyTL['Center_Lat'][len(emptyTL)-1], emptyTL['Center_Lon'][len(emptyTL)-1]]]
        perMinCoords, latRate, lonRate = findPanRate(coordList, emptyTL['Scan_Time'][0],
                                                     emptyTL['Scan_Time'][len(emptyTL)-1])
        for site in uniqIDs:
            timeList = minuteList(emptyTL['Scan_Time'][0], emptyTL['Scan_Time'][len(emptyTL)-1], site)
            timeList = timeList2dt(timeList)
            masterTimeDict[site] = timeList
        # This is probably the most likely part of the code to break. I wrote this
        # a week ago and it's so complicated I already forget how it works.
        fixTL = timelineCut(masterTimeDict, emptyTL['Scan_Time'][0],
                            emptyTL['Scan_Time'][len(emptyTL)-1], IDs, perMinCoords)
        TL = pd.concat([TL, fixTL]).sort_values('Scan_Time').reset_index()

    # This section is an aid to help you perfect the loop. It previews the first
    # and last frame and prints the timeline.
    Nframes = len(TL)
    TL = TL.reset_index()
    frames = 'n'
    while frames != 'y':
        print(TL)
        print('Previewing First/Last Plots...')
        radar = getRadar(TL['Scan_Time'][0], TL['Radar_ID'][0], writePath)
        plot = radarCoords(radar, TL['Center_Lat'][0], TL['Center_Lon'][0],
                           degSize, refPlot, sweep=0)
        radar = getRadar(TL['Scan_Time'][len(TL)-1], TL['Radar_ID'][len(TL)-1], writePath)
        plot = radarCoords(radar, TL['Center_Lat'][len(TL)-1],
                           TL['Center_Lon'][len(TL)-1], degSize, refPlot, sweep=0)
        frames = input(f'Total Frames: {Nframes}. Proceed? (y/n)')
    
    count = 0
    for row, column in TL.iterrows():
        radar = getRadar(TL['Scan_Time'][row], TL['Radar_ID'][row], writePath)
        
        if maxLL:
            # Extra if statement to prevent break on last row
            if row!=len(TL)-1:
                sweeps = sweepFinder(radar)
                curCoord = (TL['Center_Lat'][row], TL['Center_Lon'][row])
                nextCoord = (TL['Center_Lat'][row+1], TL['Center_Lon'][row+1])
                dlat = (nextCoord[0]-curCoord[0]) / len(sweeps)
                dlon = (nextCoord[1]-curCoord[1]) / len(sweeps)
                # Iterates through the sweep list giving the function a new
                # index each time
                for idx, sweep in enumerate(sweeps):
                    plot = radarCoords(radar, TL['Center_Lat'][row]+(dlat*idx),
                                       TL['Center_Lon'][row]+(dlon*idx),
                                       degSize, refPlot, sweep)
                    plot.savefig(os.path.join(tempPath, f"{count}.png"),
                                 format='png', bbox_inches='tight')
                    count+=1
            else:
                plot = radarCoords(radar, TL['Center_Lat'][row],
                                   TL['Center_Lon'][row], degSize, refPlot, sweep=0)
                plot.savefig(os.path.join(tempPath, f"{count}.png"), format='png',
                             bbox_inches='tight')
        else:
            plot = radarCoords(radar, TL['Center_Lat'][row],
                               TL['Center_Lon'][row], degSize, refPlot, sweep=0)
            plot.savefig(os.path.join(tempPath, f"{count}.png"), format='png',
                         bbox_inches='tight')
            count+=1
        print(f'Download Progress: {row+1}/{Nframes}')
    print("--- %s seconds ---" % (time.time() - start_time))
    
    image_folder = tempPath
    if not GIF:
        # Write the video at the end. Not recursive so placed in main() for simplicity.
        video_name = f'{startTime.year}{startTime.month}{startTime.day}.mp4'
        images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
        images.sort(key=lambda f: int(re.sub('\D', '', f)))
        frame = cv2.imread(os.path.join(image_folder, images[1]))
        height, width, layers = frame.shape
        # 3rd input is video frame rate
        video = cv2.VideoWriter(video_name, 0, 15, (width,height))
        for image in images:
            video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.destroyAllWindows()
        video.release() 
        print('(Video format warning can be ingored)')
    
    else:
        output_file = f'{startTime.year}{startTime.month}{startTime.day}.gif'
        images_to_gif(image_folder, output_file, duration=50, loop=0)
    
    # Timeline dataframe is returned for troubleshooting.
    return TL
    
if __name__ == '__main__':
    TL = main()