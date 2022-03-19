import matplotlib.pyplot as plt
import numpy as np
import scipy
import pandas as pd
import matplotlib
import csv

from numba import float64
from scipy.cluster.vq import kmeans2, whiten, kmeans, vq
from sympy.physics.units import cm


def IOWork(filePath, initTime, endTime):
    """
    read csv file and get points of get on and get off.
    Args:
        filePath: str,the path of csv file
        initTime: the start time we screen
        endTime:  the end time we screen
    Return:
        actionPoint: dataframe,the points of get on and off
    """

    # read csv path

    # od=0 on
    # od=1 off
    head = ['id', 'ts', 'lon', 'lat', 'direct', 'speed', 'status', 'od']
    taxiIdBox = []  # id
    taxiBox = []  # all dict

    actionPoints = []  # save the results

    with open(filePath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        csvReader = csv.DictReader(f, fieldnames=head)
        for row in csvReader:
            # filter time
            if (row['ts'] > initTime) & (row['ts'] < endTime):
                # justice whether the id has showed before
                if row['id'] in taxiIdBox:  # has
                    index = taxiIdBox.index(row['id'])
                    if taxiBox[index]['status'] != row['status']:
                        if row['status'] == '重车':
                            row['od'] = 0
                        else:
                            row['od'] = 1
                        taxiBox[index] = row
                        actionPoints.append(row)
                else:  # first appear
                    taxiIdBox.append(row['id'])
                    taxiBox.append(row)
    return actionPoints


def main():
    actionPoints = IOWork('./Trajectory Data/20140501.csv', '2014-05-01 06:00:00', '2014-05-01 06:15:00')

    lon = []
    lat = []
    lonOn = []
    lonOff = []
    latOn = []
    latOff = []

    for i in actionPoints:
        lon.append(float64(i['lon']))
        lat.append(float64(i['lat']))
        if i['od'] == 0:
            lonOn.append(float64(i['lon']))
            latOn.append(float64(i['lat']))
        if i['od'] == 1:
            lonOff.append(float64(i['lon']))
            latOff.append(float64(i['lat']))

    lonArray = np.array(lon)
    latArray = np.array(lat)
    lonOnArray = np.array(lonOn)
    latOnArray = np.array(latOn)
    lonOffArray = np.array(lonOff)
    latOffArray = np.array(latOff)

    # connect array
    points = (np.stack((lonArray, latArray), axis=0)).T
    pointOn = (np.stack((lonOnArray, latOnArray), axis=0)).T
    pointOff = (np.stack((lonOffArray, latOffArray), axis=0)).T
    print(points.shape)

    # =========== K means ================
    # 1：根据K个中心将数据集按到中心值距离分簇
    # 2：将已分的数据集，根据平均向量再确定中心值
    # 3：重复1、2
    # 步骤，直至中心值不再移动（每次的差值与上次相同）

    # set cluster number
    n = 10

    # --------function one----------
    center, _ = kmeans(points, n)
    cluster, _ = vq(points, center)
    # --------function two----------
    _, label = kmeans2(points, n)

    _, labelOn = kmeans2(pointOn, n)
    _, labelOff = kmeans2(pointOff, n)

    # =========== paint ===================
    plt.subplot(231)
    plt.title('Display All Points')
    plt.xlim(114.1, 114.5)
    plt.ylim(30.3, 30.8)
    plt.scatter(lonArray, latArray, s=1, c='DarkBlue')

    plt.subplot(232)
    plt.title('Kmeans For All Points use function one')
    plt.xlim(114.1, 114.5)
    plt.ylim(30.3, 30.8)
    df = pd.DataFrame(points, index=cluster, columns=['lon', 'lat'])
    plt.scatter(df['lon'], df['lat'], s=1, c=df.index, cmap='hot')

    plt.subplot(233)
    plt.title('Kmeans For All Points use function two')
    plt.xlim(114.1, 114.5)
    plt.ylim(30.3, 30.8)
    df = pd.DataFrame(points, index=label, columns=['lon', 'lat'])
    plt.scatter(df['lon'], df['lat'], s=1, c=df.index, cmap='viridis')

    plt.subplot(234)
    plt.title('Kmeans For On Points')
    plt.xlim(114.1, 114.5)
    plt.ylim(30.3, 30.8)
    dfOn = pd.DataFrame(pointOn, index=labelOn, columns=['lon', 'lat'])
    plt.scatter(dfOn['lon'], dfOn['lat'], s=1, c=dfOn.index, cmap='inferno')

    plt.subplot(235)
    plt.title('Kmeans For Off Points')
    plt.xlim(114.1, 114.5)
    plt.ylim(30.3, 30.8)
    dfOff = pd.DataFrame(pointOff, index=labelOff, columns=['lon', 'lat'])
    plt.scatter(dfOff['lon'], dfOff['lat'], s=1, c=dfOff.index, cmap='spring')

    plt.show()


if __name__ == '__main__':
    main()
