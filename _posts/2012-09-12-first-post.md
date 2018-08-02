---
title: Chicago Divvy Bicycle
author: Ding Ma
layout: post
---

## Chicago Divvy Bicycle Project

[Divvy](https://www.divvybikes.com/) is a bicycle sharing system in the City of Chicago and two adjacent suburbs operated by Motivate for the Chicago Department of Transportation. It operates 5800 bicycles at 580 stations in an area bounded by 87th Street on the south, Central Street in Evanston on the north, Rainbow Beach Park near South Shore Drive on the east, and Harlem Avenue in Oak Park on the west.

In this blog, I proformed a series of data processing, visualization, and analysis. All the source code can be found on my Github page.

## Data
### Bicycle Data
You can download the Divvy Bicycle data from 2013 to 2017 [here](https://www.divvybikes.com/system-data).
In this data, each trip is anonymized and includes:
  * Trip start day and time
  * Trip end day and time
  * Trip start station
  * Trip end station
  * Rider type (Member, Single Ride, and Explore Pass)
  * If a Member trip, it will also include Member’s gender and year of birth

### Weather Data
In this project, I write a wrapper to download the historical weather of Chicago between 2013-2017 from [National Oceanic and Atmospheric Administration](https://www.noaa.gov).

## Visualization
  ### Stations at the end of 2017
  Here is an [interactive map](https://dingma129.github.io/assets/active_image/divvy/station_distribution.html) for station distribution. We can see that most of the stations are located in the downtown area.

  ### By Year
  Annual total trip counts were growing from 2014 to 2017. However, the average trip duration did not change at all.

<img src="https://dingma129.github.io/assets/figures/divvy/tripcountvsyear.png" width="800">

  ### By Month 
  During June, July, August and September, there were more trips. The average trip duration was also relatively longer during those four months. 

<img src="https://dingma129.github.io/assets/figures/divvy/tripcountvsmonth.png" width="800">

  ### By Day of Week
  There were more rides during weekdays comparing to weekends. But the average trip duration was longer during weekend. 

<img src="https://dingma129.github.io/assets/figures/divvy/tripcountvsdayofweek.png" width="800">

  ### By Hour
  If we take a closer look, there are particular patterns within a day.
     * During weekday, there were two peaks of trip counts. One happened at 7:00-8:59, and the other happened at 16:00-18:59.
     * During weekend, there was plenty of rides during 8:00-20:59 with no sharp peak.

<img src="https://dingma129.github.io/assets/figures/divvy/tripcountvshour.png" width="800">

## Analyses

### Analysis About Stations
Now I want to perform some analysis about the classification of types of all stations.
I introduced the following features for each station.
  * num_of_nearby_005: number of nearby stations with L1 metric of longitude and latitude being less than 0.005, which is roughly 1-2 blocks
  * num_of_nearby_01: number of nearby stations with L1 metric of longitude and latitude being less than 0.01
  * wkd_am_start_median: median of hourly departure trip counts between 7:00-8:59 on weekdays
  * wkd_am_end_median: median of hourly arrival trip counts between 7:00-8:59 on weekdays
  * wkd_pm_start_median: median of hourly departure trip counts between 16:00-18:59 on weekdays
  * wkd_pm_end_median: median of hourly arrival trip counts between 16:00-18:59 on weekdays
  * wke_start_median: median of hourly departure trip counts on weekends
  * wke_end_median: median of hourly arrival trip counts on weekends

There are some interesting correlations between those features. For example, the weekday morning arrival is strongly correlated to weekday afternoon departure. 

<img src="https://dingma129.github.io/assets/figures/divvy/amendvspmstart.png" width="400">

After rescaling the data to mean=0 and std=1, we perform a model selection with Gaussian Mixture Model(GMM) using BIC. The model selection concerns both the covariance type and the number of components in the model. From below, we can see that the best GMM model is the one with 5 components and covariance type being full.

<img src="https://dingma129.github.io/assets/figures/divvy/GMMbic.png" width="400">

Then we perform the GMM to classify the stations into 5 classes. An interactive map can be found [here](https://dingma129.github.io/assets/active_image/divvy/station_5types.html). 

Those 5 classes correspond to the following 5 cases:
  1. Work Area: mainly in downtown Chicago
  2. Living Area I: living areas close to downtown
  3. Living Area II: living areas not that close to downtown comparing to Living Area I
  4. Transition Centers
  5. Unpopular Area: mainly on the service boundary

### Analysis About Routes

After grouping all the stations into 5 classes, I want to see how differently do people travel using bicycles from different class of stations.
The most interesting 8 cases can be found in this interactive [map](https://dingma129.github.io/assets/active_image/divvy/wkd_8cases.html). In each case, I only plot the top 50 popular trips in order to get a clean figure. I get the following fun facts:
1. On weekday morning, people tend to ride from home to metro stations, take a metro, then continue riding to work.
    * The rides from living area (Living I and Living II) are mainly towards metro stations, while there are still some people riding directly from Living I towards Work Area.
    * The rides from Trainsition Centers are mainly towards Work Area, and most rides towards Work Area are coming from Trainsition Centers
2. On weekday afternoon, we can see exactly the opposite trends as in the morning.